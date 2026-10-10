"""PPO with self-play -- PLAN.md Phase 4, §5.3, §6.5.

Run: python3 rl/ppo.py [iterations] [--victory 3] [--envs 64] [--device cpu]

Three decisions here are specific to this game rather than to PPO, and each one
removes a class of bug the plan warns about.

**Only complete episodes are ever trained on.** Reward is terminal-only and
`gamma = 1.0` (§1.2 -- it makes the critic a literal win-probability predictor,
which is the deck-evaluator deliverable). So a transition's target does not
depend on *when* it is harvested, and there is no reason to bootstrap a
half-finished game. Per-env buffers carry across rollout boundaries and are
flushed only when the game ends. That deletes truncation-boundary handling
entirely -- the single most bug-prone part of a PPO implementation -- at the
cost of some transitions being one policy version stale, which is exactly what
PPO's ratio clipping already exists to handle (and the stored `logp` makes the
ratio correct regardless).

**Every seat keeps its own trajectory, and GAE runs inside it** (§5.3 gotchas
1-2). Between two of a seat's decisions the opponent moved, possibly several
times. Bootstrapping across that would have the critic predict its own value
from a position its opponent chose. Each seat's subsequence is treated as its
own episode with the opponent folded into the dynamics, so `V(s_{t+1})` is
always *that seat's* next decision.

**Rewards are written to both seats at once from the game result**, never
credited to whoever moved last (§5.3 gotcha 1).

When it does not learn, §4 Phase 4 gives the debug order: mask alignment, reward
sign and seat attribution, observation canonicalization, advantage
normalization, learning rate -- in that order, and it is one of the first three
about 90% of the time. Tests [1]-[5] in `rl/tests/test_env.py` and [1]-[3] in
`rl/tests/test_ppo.py` cover those first three, so a failure here should be
looked for further down the list than instinct suggests.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

from rl.config import Config
from rl.engine import actions as A
from rl.engine.cardtable import full_table
from rl.eval import report
from rl.nets import (RiftboundNet, count_params, load_weights,
                     shapes_growable, to_torch)
from rl.pool import Ladder, OpponentPool, elo_from_winrate
from rl.tests.fuzz import v0_pool
from rl.parallel import DealSpec
from rl.vec import BatchObs, VecRiftbound, batch


@dataclass
class HP:
    """§6.5, unchanged except where the note says why."""
    gamma: float = 1.0            # terminal-only reward; critic = win prob
    gae_lambda: float = 0.95
    lr: float = 3e-4
    n_envs: int = 64
    rollout: int = 2048           # transitions per update, not steps per env
    minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01        # applied to the NORMALIZED entropy (§6.3)
    vf_coef: float = 0.5
    # Which critic drives the advantage. False -- the honest one, seeing only
    # what the policy sees -- is the default; see `nets.RiftboundNet.values`
    # for why the privileged critic is an approximation rather than a free
    # variance reduction. Both are trained either way, so flipping this and
    # comparing `explained_var` / `explained_var_aux` settles it by measurement.
    privileged_critic: bool = False
    max_grad_norm: float = 0.5
    anneal_lr: bool = True
    # **BACKLOG W1.** Iterations over which the LR ramps linearly up from
    # 1/N of its scheduled value. A warm start (`--init`) restores the weights
    # but not Adam's moments, so the first full-size step jumped kl 0.3-0.8
    # from a policy that was already converged. 0 = off (the old behaviour).
    lr_warmup: int = 0

    # --- opponent pool (Phase 5) ------------------------------------------
    # `pool_frac` of envs play a sampled frozen past self; the rest are pure
    # self-play. Mixing rather than replacing: pure pool play loses the
    # sharpening that comes from facing your exact current strength, pure
    # self-play is what produced the 8.8-point drift against greedy.
    pool_frac: float = 0.5
    pool_every: int = 10          # snapshot the learner every N iterations
    # **BACKLOG #22, cheap version.** Checkpoints pinned into the pool for the
    # whole run, never evicted and never below `pin_floor` of the sample mass.
    # Part of the HP so a resume refuses a changed opponent set: the pool's
    # composition is the independent variable of a diversity run, and resuming
    # with a different one would splice two experiments, which is the mistake
    # `resume.pt` started carrying hyperparameters to prevent.
    pool_init: tuple = ()
    pool_capacity: int = 20
    pfsp_p: float = 2.0
    # BACKLOG #21. Sample (opponent, learner deck) JOINTLY from a per-matchup
    # win record instead of sampling the opponent from a per-opponent average
    # and the deck independently. OFF by default so an unflagged run reproduces
    # today's sampling exactly -- see `pool.OpponentPool` for the scheme and
    # `MatchupDealer` for the half that has to live next to the deal.
    matchup_pfsp: bool = False
    # BACKLOG D2/D2b. Every env slot plays Bo3 MATCHES (`Bo3Dealer`): the
    # between-games decisions are real decisions, and games 2-3 carry what the
    # match has shown. OFF by default -- an unflagged run deals Bo1 exactly as
    # it always did.
    bo3: bool = False


# ---------------------------------------------------------------------------
# Rollout storage
# ---------------------------------------------------------------------------

def slice_obs(b: BatchObs, i: int) -> dict:
    """One env's observation, detached from the batch."""
    return {
        "zones": {k: v[i].copy() for k, v in b.zones.items()},
        "zone_mask": {k: v[i].copy() for k, v in b.zone_mask.items()},
        "globals": b.globals[i].copy(),
        "legal_actions": b.legal_actions[i].copy(),
        "action_mask": b.action_mask[i].copy(),
        "privileged": None if b.privileged is None else b.privileged[i].copy(),
    }


def _subset(b: BatchObs, slots: list[int]) -> BatchObs:
    """A BatchObs restricted to some env slots, for a per-opponent forward."""
    return BatchObs(
        zones={k: v[slots] for k, v in b.zones.items()},
        zone_mask={k: v[slots] for k, v in b.zone_mask.items()},
        globals=b.globals[slots],
        legal_actions=b.legal_actions[slots],
        action_mask=b.action_mask[slots],
        to_move=b.to_move[slots],
        n_legal=b.n_legal[slots],
        privileged=None if b.privileged is None else b.privileged[slots],
    )


def stack_obs(rows: list[dict], device: str) -> dict:
    keys = rows[0]["zones"].keys()
    b = BatchObs(
        zones={k: np.stack([r["zones"][k] for r in rows]) for k in keys},
        zone_mask={k: np.stack([r["zone_mask"][k] for r in rows]) for k in keys},
        globals=np.stack([r["globals"] for r in rows]),
        legal_actions=np.stack([r["legal_actions"] for r in rows]),
        action_mask=np.stack([r["action_mask"] for r in rows]),
        to_move=np.zeros(len(rows), np.int8),
        n_legal=np.zeros(len(rows), np.int16),
        privileged=(None if rows[0]["privileged"] is None
                    else np.stack([r["privileged"] for r in rows])),
    )
    return to_torch(b, device)


@dataclass
class Step:
    obs: dict
    action: int
    logp: float
    value: float
    seat: int
    adv: float = 0.0
    ret: float = 0.0


@dataclass
class Rollout:
    steps: list[Step] = field(default_factory=list)
    episodes: int = 0
    ep_len: list[int] = field(default_factory=list)
    ep_turns: list[int] = field(default_factory=list)
    truncated: int = 0
    seat0_wins: int = 0
    # Episodes a seat lost by looping (`env._check_sterile_loop`). A detector
    # nobody can see is a detector nobody trusts, so it is printed every
    # iteration next to `trunc`: the pair separates "games are running long"
    # from "a policy found a livelock and is sitting in it".
    loops: int = 0
    # Episodes discarded because the board ran out of permanent ROWS
    # (`state.BoardOverflow`). Printed with `loop=` because the two are the same
    # question -- is the policy doing something the engine cannot represent --
    # and because a rising rate is the signal to raise `MAX_PERMS` or to build
    # the loop shortcut (BACKLOG item 9) rather than to keep discarding games.
    overflows: int = 0
    # Episodes against a FROZEN pool opponent, and how many the learner won.
    # `pool.record` already tracks this per entry, but it decays at 0.98 for
    # PFSP's benefit -- a moving estimate is what sampling wants and exactly
    # what a measurement does not. These two are the plain per-iteration count,
    # and in exploiter mode they are the headline number of the whole run.
    opp_games: int = 0
    opp_wins: int = 0


def finish_episode(pending: list[Step], rewards, hp: HP) -> None:
    """Assign returns and advantages, one seat's subsequence at a time.

    `rewards` is indexed by seat and both entries are filled from the game
    result -- the seat that moved last has no special status here, which is the
    whole point (§5.3 gotcha 1).
    """
    for seat in (0, 1):
        own = [s for s in pending if s.seat == seat]
        if not own:
            continue
        r_terminal = float(rewards[seat])
        adv = 0.0
        for t in reversed(range(len(own))):
            last = t == len(own) - 1
            # Terminal-only reward: every delta but the last is pure value
            # bootstrap from THIS seat's next decision, never the opponent's.
            next_v = 0.0 if last else own[t + 1].value
            r = r_terminal if last else 0.0
            delta = r + hp.gamma * next_v - own[t].value
            adv = delta + hp.gamma * hp.gae_lambda * adv
            own[t].adv = adv
            own[t].ret = adv + own[t].value


# See `Trainer.update`: larger MPS minibatches get wrong gradients (torch 2.6).
MPS_CHUNK = 256


class Trainer:
    def __init__(self, table, cfg: Config, deal_fn, hp: HP,
                 device: str = "cpu", seed: int = 0,
                 update_device: str | None = None, workers: int = 0,
                 deal_spec=None) -> None:
        self.table, self.cfg, self.hp, self.device = table, cfg, hp, device
        self.deal_fn = deal_fn
        # **BACKLOG P2.** `workers > 0` runs the games in that many processes
        # (`rl/parallel.py`); the deal is then rebuilt inside each worker from
        # `deal_spec`, and `deal_fn` is only what main uses for itself. Both
        # classes seed per slot, so the same seed deals the same games.
        self.workers = int(workers)
        if self.workers > 0:
            from rl.parallel import ParallelVec
            if deal_spec is None:
                raise ValueError("workers > 0 needs a deal_spec (DealSpec): "
                                 "the deal is built inside each worker")
            if hp.matchup_pfsp:
                raise ValueError("--matchup-pfsp cannot run with --workers: "
                                 "its dealer takes per-deal choices from the "
                                 "main process's opponent pool (take())")
            self.vec = ParallelVec(table, cfg, hp.n_envs, deal_spec,
                                   seed0=seed, workers=self.workers)
        else:
            self.vec = VecRiftbound(table, cfg, hp.n_envs, deal_fn, seed0=seed)
        self.net = RiftboundNet(
            self.vec.shapes(),
            privileged_critic=hp.privileged_critic).to(device)
        # **BACKLOG P1.** The update can run on another device (the Apple GPU:
        # 4.3x faster, `docs/PARALLEL.md`) while acting stays on `device`, where
        # one small forward per env step is cheaper than a transfer. `upd_net`
        # owns the optimizer; `net` is what everything else reads and saves.
        # `update()` syncs net -> upd_net on entry and back on exit, so every
        # path that loads weights into `net` (--init, resume) needs no change.
        self.upd_device = update_device or device
        self.upd_net = (self.net if self.upd_device == device
                        else copy.deepcopy(self.net).to(self.upd_device))
        self.opt = torch.optim.Adam(self.upd_net.parameters(), lr=hp.lr, eps=1e-5)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.obs = self.vec.reset()
        self.pending: list[list[Step]] = [[] for _ in range(hp.n_envs)]
        self.global_step = 0

        # Opponent pool. `env_opp[i]` is the frozen opponent assigned to slot i
        # (None = pure self-play) and `learner_seat[i]` which side the learner
        # is on there -- alternated, because a learner that always sits in seat
        # 0 would never train the going-second game, and victory 3 has a large
        # first-player edge (measured 64% self-play seat 0).
        self.pool = OpponentPool(capacity=hp.pool_capacity, p=hp.pfsp_p,
                                 matchup=hp.matchup_pfsp)
        self.rng = np.random.default_rng(seed)
        self.env_opp: list = [None] * hp.n_envs
        self.learner_seat = [i % 2 for i in range(hp.n_envs)]
        self._opp_cache: dict[int, RiftboundNet] = {}
        # Which decklist the learner is piloting in slot i, and both seats'
        # lists. `None` unless the deal is a `MatchupDealer`, and `pool.record`
        # ignores a `None` key, so this is inert in every other mode.
        self.env_deck: list = [None] * hp.n_envs
        self.env_names: list = [None] * hp.n_envs
        # **Drain the queue the `reset` above just filled.** Those n deals
        # happened before this object existed, so nothing has collected their
        # choices; leaving them queued would offset every later pop by n_envs
        # and mislabel every record. The choices themselves are uniform here
        # (the pool is empty at construction), but the ALIGNMENT matters.
        for i in range(hp.n_envs):
            pay = self._take_deal()
            if pay is not None:
                self.env_names[i] = pay["names"]
                self.env_deck[i] = pay["names"][self.learner_seat[i]]

    def _opp_net(self, entry) -> RiftboundNet:
        """Instantiate (and cache) a frozen opponent."""
        key = entry.uid             # never id(entry): see `Entry.uid` (B2)
        net = self._opp_cache.get(key)
        if net is None:
            net = RiftboundNet(
                self.vec.shapes(),
                privileged_critic=self.hp.privileged_critic).to(self.device)
            net.load_state_dict(entry.state_dict)
            net.eval()
            if len(self._opp_cache) > self.hp.pool_capacity + 2:
                # Keep what is still in use: an evicted entry can still be
                # seated in a slot whose episode has not ended.
                live = ({e.uid for e in self.pool.entries}
                        | {o.uid for o in self.env_opp if o is not None})
                self._opp_cache = {k: v for k, v in self._opp_cache.items()
                                   if k in live}
            self._opp_cache[key] = net
        return net

    def _take_deal(self):
        """The `MatchupDealer`'s choice for the episode just dealt, if any."""
        take = getattr(self.deal_fn, "take", None)
        return take() if callable(take) else None

    def _assign(self, i: int) -> None:
        """Pick who slot i plays next episode, and which seat the learner takes.

        With `--matchup-pfsp` the pair was already chosen jointly, at the deal
        (see `MatchupDealer`), and this collects it. Without it -- or on the two
        paths that call `_assign` with nothing queued, `--exploit` priming and
        the resume re-pick -- the opponent is sampled here as it always was,
        conditioned on the already-dealt deck when there is a matrix to condition
        on.
        """
        pay = self._take_deal()
        if pay is not None:
            self.env_names[i] = pay["names"]
            self.learner_seat[i] = int(pay["seat"])
            self.env_deck[i] = pay["names"][self.learner_seat[i]]
            self.env_opp[i] = pay["entry"]
            return
        self.learner_seat[i] = int(self.rng.integers(2))
        if self.env_names[i] is not None:
            self.env_deck[i] = self.env_names[i][self.learner_seat[i]]
        self.env_opp[i] = (self.pool.sample(self.rng, deck_key=self.env_deck[i])
                           if self.pool.entries
                           and self.rng.random() < self.hp.pool_frac else None)

    # -- collect ---------------------------------------------------------

    def close(self) -> None:
        """Stop worker processes, if any. Safe to call twice."""
        close = getattr(self.vec, "close", None)
        if close is not None:
            close()

    # **Torch threads are NOT capped during collect with workers** (P2,
    # measured, docs/PARALLEL.md). Lockstep never overlaps the two: workers
    # sit idle while main runs the forwards, so a cap only slows the forwards
    # (workers 8, rollout 4096: 1 thread 8.6s, 2 7.8s, 4 7.4s, 8 7.3s).
    @torch.no_grad()
    def collect(self) -> Rollout:
        hp, out = self.hp, Rollout()
        while len(out.steps) < hp.rollout:
            # Which slots is the learner acting in? The rest are driven by a
            # frozen opponent and produce no training data -- recording them
            # would teach the learner to imitate an old checkpoint.
            mine = [i for i in range(hp.n_envs)
                    if self.env_opp[i] is None
                    or int(self.obs.to_move[i]) == self.learner_seat[i]]
            idx_np = np.zeros(hp.n_envs, np.int64)

            if mine:
                t = to_torch(self.obs, self.device)
                idx, logp, _, value = self.net.act(t, generator=self.gen)
                idx_np[mine] = idx.cpu().numpy()[mine]
                logp_np = logp.cpu().numpy()
                val_np = value.cpu().numpy()
                for i in mine:
                    self.pending[i].append(Step(
                        obs=slice_obs(self.obs, i), action=int(idx_np[i]),
                        logp=float(logp_np[i]), value=float(val_np[i]),
                        seat=int(self.obs.to_move[i])))

            # Group the remaining slots by opponent so each frozen net runs one
            # batched forward rather than one per slot.
            others: dict[int, list[int]] = {}
            for i in range(hp.n_envs):
                if i not in mine:
                    others.setdefault(-1 if self.env_opp[i] is None
                                     else self.env_opp[i].uid, []).append(i)
            for _, slots in others.items():
                opp = self._opp_net(self.env_opp[slots[0]])
                sub = to_torch(_subset(self.obs, slots), self.device)
                with torch.no_grad():
                    oidx, *_ = opp.act(sub, generator=self.gen)
                idx_np[slots] = oidx.cpu().numpy()
                if getattr(self.env_opp[slots[0]], "old_payment", False):
                    # D6: a pinned checkpoint that never saw the payment choice
                    # plays the engine's old default (option 0) there.
                    for i in slots:
                        if self.obs.legal_actions[i, 0, A.A_PAY_SPLIT] == 1.0:
                            idx_np[i] = 0

            self.obs, rewards, done, infos = self.vec.step(idx_np)
            self.global_step += hp.n_envs

            for i in np.flatnonzero(done):
                if self.env_opp[i] is not None:
                    won = rewards[i][self.learner_seat[i]] > 0
                    self.pool.record(self.env_opp[i], learner_won=won,
                                     deck_key=self.env_deck[i])
                    out.opp_games += 1
                    out.opp_wins += int(won)
                # The next game of a Bo3 match keeps its seats and opponent:
                # the learner is the same player, carrying its own history.
                if not self.vec.continuing(int(i)):
                    self._assign(i)
                finish_episode(self.pending[i], rewards[i], hp)
                ep = infos[i]["episode"]
                out.steps.extend(self.pending[i])
                out.episodes += 1
                out.ep_len.append(len(self.pending[i]))
                out.ep_turns.append(ep["turns"])
                out.truncated += int(ep["truncated"])
                out.loops += int(ep.get("loop_loser", -1) >= 0)
                out.overflows += int(bool(ep.get("board_overflow")))
                # **From the reward, not from `winner`.** A truncated game is
                # decided on points by 408.2.b, so `winner` stays -1 on a game
                # seat 0 genuinely won -- the seat balance diagnostic would have
                # drifted low exactly when truncations start appearing, which is
                # at victory 8.
                out.seat0_wins += int(rewards[i][0] > 0)
                self.pending[i] = []
        return out

    # -- update ----------------------------------------------------------

    def update(self, roll: Rollout, frac_done: float, warm: float = 1.0) -> dict:
        hp, dev = self.hp, self.upd_device
        net = self.upd_net
        if net is not self.net:
            net.load_state_dict(self.net.state_dict())
        steps = roll.steps
        n = len(steps)

        if hp.anneal_lr:
            # Cosine to 10% rather than to 0: a run that ends with a dead
            # learning rate cannot recover if it is extended.
            lr = hp.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac_done)))
        else:
            lr = hp.lr
        lr *= warm
        for g in self.opt.param_groups:
            g["lr"] = lr

        obs = stack_obs([s.obs for s in steps], dev)
        act = torch.as_tensor([s.action for s in steps], device=dev)
        old_logp = torch.as_tensor([s.logp for s in steps], dtype=torch.float32,
                                   device=dev)
        adv_all = torch.as_tensor([s.adv for s in steps], dtype=torch.float32,
                                  device=dev)
        ret_all = torch.as_tensor([s.ret for s in steps], dtype=torch.float32,
                                  device=dev)

        stats = {"pg": 0.0, "vf": 0.0, "vf_aux": 0.0, "ent": 0.0,
                 "kl": 0.0, "clipfrac": 0.0}
        nb = 0
        net_dev_is_mps = torch.device(dev).type == "mps"
        for _ in range(hp.update_epochs):
            order = torch.randperm(n, generator=self.gen).to(dev)
            # Even splits, not a fixed stride. A stride leaves a remainder
            # minibatch that can hold a single transition, whose advantage
            # std is undefined -- that produced a NaN policy loss which then
            # spread to every parameter through one optimizer step, with no
            # error and no crash.
            for mb in torch.tensor_split(order, hp.minibatches):
                if mb.numel() < 2:
                    continue
                # Normalized per minibatch, which is what CleanRL does and what
                # the §4 debug list means by "advantage normalization".
                a_mb = adv_all[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std(unbiased=False) + 1e-8)

                # **On MPS the minibatch is fed in chunks** and the gradients
                # summed, which is the same update. Measured 2026-10-09 (torch
                # 2.6): a 3,192-transition minibatch got card_enc WEIGHT
                # gradients with norm ~1e7 and cosine 0.003 to the CPU's (biases
                # and every other layer exact), so each step was effectively
                # random and a smoke run diverged by iteration 9. Chunks of 512
                # and 256 matched the CPU in every trial. On the CPU the chunk is
                # the whole minibatch, so CPU runs are unchanged.
                chunk = MPS_CHUNK if net_dev_is_mps else mb.numel()
                self.opt.zero_grad(set_to_none=True)
                acc = {"pg": 0.0, "vf": 0.0, "vf_aux": 0.0, "ent": 0.0,
                       "kl": 0.0, "clipfrac": 0.0}
                for pos in torch.split(torch.arange(mb.numel(), device=dev), chunk):
                    ch = mb[pos]
                    w = ch.numel() / mb.numel()
                    sub = {
                        "zones": {k: v[ch] for k, v in obs["zones"].items()},
                        "zone_mask": {k: v[ch] for k, v in obs["zone_mask"].items()},
                        "globals": obs["globals"][ch],
                        "actions": obs["actions"][ch],
                        "action_mask": obs["action_mask"][ch],
                    }
                    if "privileged" in obs:
                        sub["privileged"] = obs["privileged"][ch]

                    logp, ent, value, value_aux = net.evaluate(sub, act[ch])
                    ratio = (logp - old_logp[ch]).exp()
                    a = a_mb[pos]

                    pg = torch.max(-a * ratio,
                                   -a * ratio.clamp(1 - hp.clip_coef,
                                                    1 + hp.clip_coef)).mean()
                    vf = 0.5 * (value - ret_all[ch]).pow(2).mean()
                    # The spare critic is fitted to the same returns so the two
                    # can be compared head to head, and it reaches the trunk
                    # through a detached input, so this term cannot move the
                    # shared representation -- only its own weights. See
                    # `nets.values`.
                    vf_aux = 0.5 * (value_aux - ret_all[ch]).pow(2).mean()
                    loss = (pg - hp.ent_coef * ent.mean()
                            + hp.vf_coef * vf + hp.vf_coef * vf_aux)
                    # A single NaN reaches every parameter through one Adam step
                    # and the run continues producing plausible-looking output
                    # forever after. Cheap to check, impossible to notice later.
                    assert torch.isfinite(loss), (
                        f"non-finite loss (pg={pg.item()}, vf={vf.item()}, "
                        f"ent={ent.mean().item()}, minibatch of {mb.numel()})")
                    (loss * w).backward()

                    with torch.no_grad():
                        acc["pg"] += w * pg.item()
                        acc["vf"] += w * vf.item()
                        acc["vf_aux"] += w * vf_aux.item()
                        acc["ent"] += w * ent.mean().item()
                        # Schulman's low-variance KL estimator.
                        d = old_logp[ch] - logp
                        acc["kl"] += w * ((d.exp() - 1) - d).mean().item()
                        acc["clipfrac"] += w * ((ratio - 1).abs()
                                                > hp.clip_coef).float().mean().item()

                nn.utils.clip_grad_norm_(net.parameters(), hp.max_grad_norm)
                self.opt.step()
                for k in acc:
                    stats[k] += acc[k]
                nb += 1

        if net is not self.net:
            self.net.load_state_dict(net.state_dict())
        for k in stats:
            stats[k] /= max(1, nb)
        stats["lr"] = lr
        stats["explained_var"] = explained_variance(
            np.array([s.value for s in steps]), np.array([s.ret for s in steps]))
        return stats


def explained_variance(pred: np.ndarray, target: np.ndarray) -> float:
    """1 - Var(target - pred) / Var(target). The critic's health in one number:
    0 means it is no better than predicting the mean, 1 means perfect."""
    var = target.var()
    return float("nan") if var == 0 else float(1 - (target - pred).var() / var)


# ---------------------------------------------------------------------------

# Derived from SPECS, not listed. This was a hand-maintained tuple, which meant
# every spell transcribed into the DSL had to be remembered here too or it
# would never appear in a training deck -- a silent omission that looks like
# nothing at all, since the card simply never gets drawn. The DSL is the single
# source of truth for "the engine can play this".
def _spell_pool() -> tuple[str, ...]:
    from rl.engine.cardtable import full_table
    from rl.engine.effects import SPECS
    t = full_table()
    # **A spec may name a card the table does not have**, and that is normal
    # rather than an error: a set under `data/upcoming/` is scripted before it
    # is released, so its `SPECS` entries exist while its rows do not (and
    # appear the moment `RIFTBOUND_UPCOMING` is set or the set ships). This
    # walks SPECS -> table, the one direction that can miss, so it has to skip
    # rather than raise -- `id_of` used to KeyError and take the whole training
    # pool down with it the first time a card was scripted early.
    known = set(t.names)
    return tuple(sorted(n for n in SPECS
                        if n in known and t.is_type(t.id_of(n), "Spell")))


SPELLS = _spell_pool()


# Measured from the 29 decklists in `decks/`, not assumed:
#   main deck   39 cards, every single deck
#   copies      at most 3 of any one card
#   spells      28% - 67% of the main deck, median 49% (33 decks)
#
# An earlier version of this file used 30 cards and called a 30% spell rate
# "well above what a real decklist would run". Both were wrong, and the second
# was wrong in the opposite direction -- 30% is near the LOW end of the real
# range. The project owner's Lillia lists run 54%.
MAIN_DECK_SIZE = 39
MAX_COPIES = 3
SPELL_RATE_RANGE = (0.28, 0.67)

# Copies per DISTINCT card, measured over the same 30 lists: 17.8% of a real
# deck's distinct cards are singletons, 30.4% are 2-ofs and 51.8% are 3-ofs --
# 16.7 distinct cards in 39, with more than half of them maxed out. A deck is
# not a sample of a pool, it is a short list of cards you want to see every
# game, and that redundancy is what `obs.py`'s `decks` zone conditions on:
# draw consistency, what the trash implies about what is left, and archetype.
# A near-singleton deck misrepresents all three.
COPY_WEIGHTS = (0.178, 0.304, 0.518)


def _engine_rev() -> str:
    """The engine commit a checkpoint was trained against.

    Without it, a checkpoint is unreproducible: find a rules bug in three days
    and there is no way to tell which runs were trained against the broken rule.
    `"<sha>-dirty"` when the tree has uncommitted changes, because that is the
    honest answer and a silent sha would be worse than none.
    """
    import subprocess
    try:
        root = Path(__file__).resolve().parents[1]
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=10)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def v1_deal(table, deck_size: int = MAIN_DECK_SIZE,
            spell_rate: tuple[float, float] | float = SPELL_RATE_RANGE):
    """Decks built to the real constraints, with the implemented spells.

    `spell_rate` may be a fixed fraction or a (lo, hi) range sampled per deal.
    A range is the default because the real decks span 28-62% and an agent
    trained on one density would be tuned for one archetype.

    **The card pool used to be what limited realism, and no longer is.** With
    only three spells implemented the 3-copy limit capped a deck at 9 spells --
    23% of 39, below even the lowest real deck -- so this function silently
    fell short of whatever rate it was asked for. The DSL now holds enough
    spells that the measured output spans the real range: mean 48%, max 67%
    against a corpus of 28-67%, median 49%. `deal_stats` still reports the
    actual distribution rather than the requested one, because that gap is
    exactly the sort of thing that comes back.
    """
    spells = [table.id_of(n) for n in SPELLS]
    units = v0_pool(table)
    bfs = [c for c in range(table.n) if table.is_type(c, "Battlefield")][:2]

    def _build(rng, want_spells: int) -> list[int]:
        """Fill a deck the way a list is WRITTEN: distinct cards, then copies.

        The old version drew `deck_size` cards one at a time from the whole
        pool and rejected a 4th copy, so a repeat was an accident -- with 233
        spells and 106 units to choose from, decks came out nearly singleton
        (36.5 distinct of 39, against 16.7 for the real lists). Copies are
        chosen deliberately instead, from the measured distribution.
        """
        deck: list[int] = []

        def take(pool, n):
            """Add exactly `n` cards as 1-3 copies each of distinct cards."""
            if n <= 0:
                return
            assert MAX_COPIES * len(pool) >= n, "pool too small to fill"
            # Without replacement, so each card is chosen at most once and its
            # copy count is decided in one place. The LAST card takes whatever
            # is left of `n`, because the deck has to come out at exactly
            # `deck_size` -- that truncation is the only reason the output
            # holds slightly more 1- and 2-ofs than the corpus does.
            order = rng.permutation(np.asarray(pool, np.int64))
            i = 0
            while n > 0:
                c, i = int(order[i]), i + 1
                k = min(n, 1 + int(rng.choice(MAX_COPIES, p=COPY_WEIGHTS)))
                deck.extend([c] * k)
                n -= k

        take(spells, want_spells)
        take(units, deck_size - len(deck))
        return deck

    def deal(seed):
        rng = np.random.default_rng(seed)
        decks = []
        for _ in range(2):
            rate = (rng.uniform(*spell_rate)
                    if isinstance(spell_rate, tuple) else spell_rate)
            decks.append(_build(rng, int(round(deck_size * rate))))
        runes = [[int(rng.integers(6)) for _ in range(12)] for _ in range(2)]
        # No decklist, so no Champion Legend (103.1). A random pool has no
        # domain identity to take one from, and a game without one is legal.
        return decks, runes, bfs, [-1, -1]
    return deal


def deck_pool_deal(table, min_coverage: float = 0.0, seed_decks=None):
    """Sample a matchup from the REAL decklists each episode.

    This is what makes domain identity learnable. `v1_deal` builds decks by
    sampling the implemented pool at random, which produces 6-domain soup with
    an unrelated rune deck -- cards are frequently uncastable and no archetype
    signal exists to condition on. Real decks commit: a rune deck of 8 Order /
    4 Body *is* the deck's identity, and it is what makes Fury play like Fury
    and Mind play like Mind.

    The network can already represent this -- cards carry a 6-dim domain
    multi-hot and the globals carry per-domain rune counts for both seats -- so
    the missing piece was never the architecture, it was the training
    distribution. Sampling a different matchup per episode is also what stops
    the policy overfitting to one archetype, which is the failure the project
    owner asked about: swap the deck and it should not suddenly play worse.
    """
    from rl.decks import load_all, matchup
    # `latest_only`: eight iterations of one Lillia list were eight independent
    # draws, so 46% of the training pool was a single archetype and 71% of
    # matchups contained it. That is exactly the overfitting this function's
    # docstring says it exists to prevent -- swap the deck and the policy should
    # not suddenly play worse -- and it was being manufactured by the loader.
    pool = [d for d in (seed_decks or load_all(table, latest_only=True))
            if d.coverage >= min_coverage]
    assert pool, f"no decks at coverage >= {min_coverage:.0%}"

    def deal(seed):
        rng = np.random.default_rng(seed)
        i, j = rng.integers(len(pool)), rng.integers(len(pool))
        # WHICH of its three battlefields each deck presents is sampled too
        # (486.5). It used to be each deck's first, always -- so two thirds of
        # every decklist's battlefields never appeared in training at all, and
        # the policy could not learn to play on ground it had never seen. Free
        # to vary now that battlefields are distinguishable in the observation;
        # before that they all encoded identically and this would have changed
        # nothing.
        picks = (int(rng.integers(3)), int(rng.integers(3)))
        return matchup(pool[int(i)], pool[int(j)], picks=picks)
    return deal


class Bo3Dealer:
    """Every env slot plays real Bo3 matches (486), back to back (D2/D2b).

    A match is two decks for its whole length, in fixed seats; between its
    games the players sideboard, pick battlefields and choose the play order as
    real decisions (`engine/pregame.py`), and each game is dealt with what the
    last one left behind -- burned battlefields, swapped lists, what each seat
    has SEEN of the other. The reward stays per GAME (the project owner's call):
    every game is its own episode, scored as one always was.

    The vector env asks `for_slot(slot, finished_state, seed)` when a slot needs
    its next game, which is where the finished game is booked. Called as a plain
    `deal_fn(seed)` -- evaluation, anything that wants one game -- it deals an
    ordinary Bo1 matchup, so a Bo3 run is still measured on the same footing
    as every earlier one.
    """

    def __init__(self, table, pool, min_coverage: float = 0.0) -> None:
        from rl.decks import present
        self.table = table
        self.pool = [d for d in pool if d.coverage >= min_coverage]
        assert self.pool, f"no decks at coverage >= {min_coverage:.0%}"
        self._present = present
        self._bo1 = deck_pool_deal(table, min_coverage, seed_decks=self.pool)
        self.ctx: dict = {}
        self.decks: dict = {}
        self.matches = 0
        self.games = [0, 0, 0]          # games played as game 1, 2, 3

    def __call__(self, seed):
        return self._bo1(seed)

    def continuing(self, slot: int) -> bool:
        """Is `slot`'s current game a later game of a match already under way?
        The trainer keeps the learner's seat and opponent fixed for those."""
        ctx = self.ctx.get(slot)
        return ctx is not None and ctx.game > 1

    def for_slot(self, slot: int, finished, seed: int):
        from rl.engine.pregame import MatchCtx
        ctx = self.ctx.get(slot)
        if ctx is not None and finished is not None:
            ctx.record(finished, self.table)
        if ctx is None or ctx.over:
            if ctx is not None:
                self.matches += 1
            rng = np.random.default_rng(seed)
            a = self.pool[int(rng.integers(len(self.pool)))]
            b = self.pool[int(rng.integers(len(self.pool)))]
            bfs = [[self._present(d, k, o) for k in range(3)]
                   for d, o in ((a, b), (b, a))]
            ctx = MatchCtx(best_of=3, designated=int(rng.integers(2)),
                           bf_reg=bfs, main=[list(a.main), list(b.main)],
                           side=[list(a.sideboard), list(b.sideboard)])
            self.ctx[slot], self.decks[slot] = ctx, (a, b)
        a, b = self.decks[slot]
        self.games[min(ctx.game, 3) - 1] += 1
        runes = [(a.runes * 3)[:12], (b.runes * 3)[:12]]
        return (None, runes, None, [a.legend, b.legend],
                [a.champion, b.champion]), ctx


class MatchupDealer:
    """`deck_pool_deal`, except the opponent and the learner's deck are chosen
    TOGETHER from the pool's matchup matrix (BACKLOG #21).

    **Why this has to live at the deal and not at `Trainer._assign`.** The
    opponent is the Trainer's choice; the decklists are the deal's. But
    `VecRiftbound.step` auto-resets a finished slot -- it calls the deal for the
    NEXT episode before `collect` has even looked at `done`, let alone called
    `_assign`. So by the time the Trainer picks an opponent, the decks are
    already dealt and unchangeable. Anything that samples the pair jointly has
    to do it inside the deal, which is why the choice is made here and the
    Trainer *collects* it.

    **The handoff is a FIFO and the alignment is exact.** `_fresh` is called in
    ascending slot order (`reset` over all slots, then one call per finished
    slot inside a `step`), and `collect` walks `np.flatnonzero(done)` in that
    same ascending order, calling `_assign` once per finished slot. One push per
    pop, same order. The two places that call `_assign` outside that loop (the
    `--exploit` priming and the resume re-pick) run with the queue already
    drained, and `Trainer._assign` falls back to the conditional
    p(opponent | deck) there rather than popping something that belongs to
    another slot. `test_ppo [11]` checks the alignment against the decks the
    envs were actually dealt, because an off-by-one here would mislabel every
    record while looking like it worked.

    Unbound (`pool is None`), or with an empty pool, this is `deck_pool_deal`
    with the same uniform deck choice -- which is what makes the first rollout
    of a flagged run identical to an unflagged one.
    """

    def __init__(self, table, decks, min_coverage: float = 0.0) -> None:
        from rl.decks import matchup as _mk
        self._mk = _mk
        self.pool = [d for d in decks if d.coverage >= min_coverage]
        assert self.pool, f"no decks at coverage >= {min_coverage:.0%}"
        self.names = [d.name for d in self.pool]
        self.opp_pool = None          # set by `bind`; the OpponentPool
        self.opp_rng = None           # the Trainer's own np Generator
        self.pool_frac = 0.0
        self.queue: deque = deque()
        self.self_play_deals = 0      # diagnostics, asserted in the tests
        self.joint_deals = 0

    def bind(self, opp_pool, rng, pool_frac: float) -> None:
        self.opp_pool, self.opp_rng, self.pool_frac = opp_pool, rng, pool_frac

    def __call__(self, seed):
        rng = np.random.default_rng(seed)
        entry, learner = None, None
        if (self.opp_pool is not None and self.opp_pool.entries
                and self.opp_rng.random() < self.pool_frac):
            # A frozen-opponent episode: this is the only kind with a matchup
            # record, so it is the only kind whose deck is steered.
            entry, learner = self.opp_pool.sample_cell(self.opp_rng, self.names)
            self.joint_deals += 1
        else:
            self.self_play_deals += 1
        if learner is None:
            i = int(rng.integers(len(self.pool)))
        else:
            i = self.names.index(learner)
        other = int(rng.integers(len(self.pool)))
        # Which SEAT the learner's deck goes on is uniform, exactly as
        # `_assign`'s own coin was; the Trainer takes this seat instead of
        # re-rolling, so the deck it thinks the learner holds is the one dealt.
        seat = int(rng.integers(2))
        a, b = (i, other) if seat == 0 else (other, i)
        picks = (int(rng.integers(3)), int(rng.integers(3)))
        self.queue.append({"names": (self.names[a], self.names[b]),
                           "entry": entry, "seat": seat})
        return self._mk(self.pool[a], self.pool[b], picks=picks)

    def take(self):
        """One queued choice, FIFO. `None` when the queue is empty, which is a
        legitimate state (see the class docstring), not an error."""
        return self.queue.popleft() if self.queue else None


def select_decks(table, names: str | None, min_coverage: float = 0.0):
    """The training deck pool, optionally narrowed to `names`.

    This is what makes generalist and specialist the same code path:

      * `None`        -> all 30 lists. The generalist.
      * `"a,b,c"`     -> a narrow generalist over three archetypes.
      * `"lillia"`    -> one deck, so every game is a MIRROR match. The
                         cheapest specialist, and an honest one: the net only
                         ever pilots that list.

    Matching is by substring, case-insensitive, so `lillia-fae-fawn` or just
    `fae` both work. An unmatched name is an error rather than an empty pool --
    a typo that silently trained on nothing would look exactly like a bad run.

    **The mirror match is not the specialist you probably want.** It never faces
    another archetype, and reading the opponent is most of the skill. The better
    one -- pilot deck D against the whole field, and take gradient only from D's
    seat -- needs the deal to tell the Trainer which seat it put D on; see
    `PLAN.md`. `learner_seat` and `collect`'s `mine` filter already do the
    credit-masking half.
    """
    from rl.decks import load_all
    pool = [d for d in load_all(table, latest_only=True)
            if d.coverage >= min_coverage]
    assert pool, f"no decks at coverage >= {min_coverage:.0%}"
    if not names:
        return pool
    out, missing = [], []
    for want in (x.strip().lower() for x in names.split(",") if x.strip()):
        hit = [d for d in pool if want in d.name.lower()]
        if not hit:
            missing.append(want)
        out.extend(hit)
    assert not missing, (
        f"no deck matches {missing}; available:\n  "
        + "\n  ".join(sorted(d.name for d in pool)))
    # de-duplicate, keep order
    seen, uniq = set(), []
    for d in out:
        if d.name not in seen:
            seen.add(d.name); uniq.append(d)
    return uniq


def deal_stats(table, deal_fn, n: int = 500) -> dict:
    """What the deal function actually produces, as opposed to what it was asked
    for. Worth checking whenever the card pool changes."""
    spells = {table.id_of(x) for x in SPELLS}
    fracs, sizes, copies, distinct = [], [], [], []
    # Redundancy, not just legality. `max_copies` alone says a deck obeyed the
    # 3-copy limit; it cannot tell a real list from 39 singletons, which is the
    # shape `v1_deal` used to produce. Compare against the corpus: 16.7
    # distinct of 39, 51.8% of them 3-ofs.
    at = {1: 0, 2: 0, 3: 0}
    for i in range(n):
        for deck in deal_fn(i)[0]:
            sizes.append(len(deck))
            fracs.append(sum(c in spells for c in deck) / max(1, len(deck)))
            counts: dict[int, int] = {}
            for c in deck:
                counts[c] = counts.get(c, 0) + 1
            copies.append(max(counts.values()))
            distinct.append(len(counts))
            for k in counts.values():
                at[min(k, 3)] += 1
    tot = max(1, sum(at.values()))
    return {"deck_size": sorted(set(sizes)),
            "spell_frac_mean": float(np.mean(fracs)),
            "spell_frac_max": float(np.max(fracs)),
            "max_copies": int(np.max(copies)),
            "distinct_mean": float(np.mean(distinct)),
            "distinct_range": (int(np.min(distinct)), int(np.max(distinct))),
            "copy_frac": tuple(round(at[k] / tot, 3) for k in (1, 2, 3))}


# Deliberately still 30, not MAIN_DECK_SIZE. Every Phase 1-4 number was
# measured with 30-card decks, and `fuzz.make_game`'s golden outcome pins that.
# Changing it would silently invalidate the whole gate suite. Recorded as a
# known caveat instead: the v0 results describe a 30-card game, which is not
# the real one.
def v0_deal(table, deck_size: int = 30):
    """The Phase 1-3 deal, so training results stay comparable to the gates."""
    pool = v0_pool(table)
    bfs = [c for c in range(table.n) if table.is_type(c, "Battlefield")][:2]

    def deal(seed):
        rng = np.random.default_rng(seed)
        decks = [[int(rng.choice(pool)) for _ in range(deck_size)]
                 for _ in range(2)]
        runes = [[int(rng.integers(6)) for _ in range(12)] for _ in range(2)]
        return decks, runes, bfs, [-1, -1]
    return deal


def _cfg_fingerprint(cfg) -> dict:
    """Every Config field, so a checkpoint records WHICH GAME it was trained in.

    v9-exploiter was trained on the units-only game against a champion trained
    with spells, and nothing caught it: `ck["shapes"]` is the observation layout,
    which is identical either way, so the one assertion on the load path could
    not see it. A rules difference is invisible to a tensor shape.
    """
    from dataclasses import asdict
    return asdict(cfg)


# Config fields that describe the MATCH rather than the game inside an episode.
# Nothing in a rollout reads them -- both mechanics they govern (Bo3 battlefield
# choice, sideboarding) happen BETWEEN games -- so a difference here cannot make
# two checkpoints incomparable, and treating it as fatal would reject every
# checkpoint written before the fields existed.
_CFG_MATCH_ONLY = frozenset({"best_of", "sideboarding",
                             # D6 adds a DECISION (which runes pay), not a rule:
                             # the game an older checkpoint played is unchanged,
                             # and it plays the old default there
                             # (`Entry.old_payment`).
                             "choose_payment"})


def _cfg_diff(old: dict, new: dict) -> dict:
    """Differences that could actually change an episode.

    Two kinds of non-difference are filtered out, and both bit immediately when
    `best_of` was added:

    * **Match-level fields** (`_CFG_MATCH_ONLY`) -- not read inside an episode.
    * **Fields the old checkpoint predates**, i.e. absent from its fingerprint and
      sitting at the dataclass default here. A field that did not exist when a
      checkpoint was written is not a change in the game it played; reporting it
      as one makes every Config addition retroactively invalidate every
      checkpoint. A missing field whose value is NOT the default is still
      reported, because then something really was chosen.
    """
    from dataclasses import fields as _dc_fields

    from rl.config import Config as _C
    defaults = {f.name: f.default for f in _dc_fields(_C)}
    out = {}
    for k in set(old) | set(new):
        if k in _CFG_MATCH_ONLY:
            continue
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if k not in old and b == defaults.get(k):
            continue
        out[k] = (a, b)
    return out


def _check_cfg(ck, cfg, what: str, strict: bool) -> None:
    """Compare a checkpoint's training config against this run's.

    `strict` for --exploit: a frozen opponent MUST play the same game, or the
    measurement is meaningless. Looser for --init, where the victory-score
    curriculum (3 -> 5 -> 8) changes the config ON PURPOSE -- but `units_only` is
    an error either way, because a policy that has never been allowed to cast a
    spell is not a warm start for a game with spells.
    """
    old = ck.get("cfg")
    if old is None:
        print(f"     NOTE: {what} predates config fingerprinting, so the game it "
              f"was trained in cannot be verified. Check it by eval.")
        return
    new = _cfg_fingerprint(cfg)
    diff = _cfg_diff(old, new)
    if not diff:
        print(f"     {what} config: identical")
        return
    lines = "\n".join(f"       {k}: ckpt {a!r} -> here {b!r}"
                      for k, (a, b) in sorted(diff.items()))
    if "units_only" in diff or strict:
        raise SystemExit(
            f"\n{what} was trained on a DIFFERENT GAME:\n{lines}\n\n"
            f"  This is what invalidated v9-exploiter. If it is deliberate, say so\n"
            f"  by matching the flags; --real-decks implies spells.")
    print(f"     {what} config differs (allowed for --init):\n{lines}")


def _hold_wake_lock() -> None:
    """Keep macOS awake for as long as THIS process lives (BACKLOG B5).

    Idle sleep froze or throttled v11 session 0, v13 and v14 -- 47% of v14's
    wall clock -- and "remember to type `caffeinate` in front of the launch"
    failed three times running. `-w <pid>` ties the assertion to this process,
    so it ends with the run however the run ends, crash included. A no-op off
    macOS or where the binary is missing.
    """
    import os
    import shutil
    import subprocess
    exe = shutil.which("caffeinate") if sys.platform == "darwin" else None
    if exe is None:
        return
    subprocess.Popen([exe, "-dimsu", "-w", str(os.getpid())],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    print("wake lock: caffeinate -dimsu held for this process "
          "(--no-caffeinate to skip)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("iterations", nargs="?", type=int, default=100)
    p.add_argument("--victory", type=int, default=3)
    p.add_argument("--spells", action="store_true",
                   help="enable the DSL spell pool (units_only=False). IMPLIED "
                        "by --real-decks, which is what you almost always want: "
                        "real decklists are full of spells and without this they "
                        "are dead cards in hand.")
    p.add_argument("--units-only", action="store_true",
                   help="force the v0 units-only game even with --real-decks. "
                        "Only for reproducing a pre-DSL run; it makes a large "
                        "fraction of every real decklist unplayable.")
    p.add_argument("--real-decks", action="store_true",
                   help="train on sampled real decklists, so domain "
                        "identity is learnable")
    p.add_argument("--min-coverage", type=float, default=0.6)
    p.add_argument("--decks", default=None,
                   help="comma-separated deck names (substring match) to train "
                        "on. One name = mirror-match specialist; omit for all "
                        "30 = generalist. Requires --real-decks.")
    p.add_argument("--holdout", type=int, default=0,
                   help="reserve N decks the learner never trains on, and report "
                        "on them separately at the end. This is how you measure "
                        "GENERALIZATION rather than assuming it: a v5 checkpoint "
                        "scored 54.5%% vs greedy on its own 30 decks and 48.0%% "
                        "once 17 unseen ones were added.")
    p.add_argument("--per-deck", type=int, default=0,
                   help="after training, report the win rate broken down by the "
                        "deck the learner piloted, with this many deals each. "
                        "This is what decides whether specialists are needed.")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    p.add_argument("--update-device", default=None, metavar="DEV",
                   help="run the PPO update on DEV (e.g. mps, the Apple GPU) "
                        "while acting stays on --device. Measured 4.3x faster "
                        "on the update, which is ~1/3 of an iteration "
                        "(BACKLOG P1, docs/PARALLEL.md). Default: --device")
    p.add_argument("--workers", type=int, default=0, metavar="N",
                   help="run the env games in N worker processes (BACKLOG P2, "
                        "docs/PARALLEL.md). 0 = the serial vector env in this "
                        "process. Same seed, same games either way. Not with "
                        "--matchup-pfsp.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=100)
    p.add_argument("--out", default="rl/runs/ppo")
    p.add_argument("--init", default=None,
                   help="checkpoint to warm-start from -- this is how the "
                        "victory-score curriculum (§1.3) is annealed 3 -> 5 -> 8")
    p.add_argument("--exploit", default=None, metavar="CKPT",
                   help="EXPLOITER MODE (BACKLOG #12). Pin the opponent pool to "
                        "this one frozen checkpoint and never snapshot the "
                        "learner into it, so every episode is learner-vs-champion "
                        "and `vs_opp=` in the log is a direct exploitability "
                        "measurement. Forces --pool-frac 1.0 and disables "
                        "pool snapshots. Pair with --init <same ckpt> to start "
                        "from a known 50%%, which is what makes the climb readable.")
    p.add_argument("--no-snapshots", action="store_true",
                   help="do not keep the opponent-pool snapshots on disk. They "
                        "are ~4MB each, one per --pool-every iterations, and they "
                        "are what rl/ladder.py needs to tell whether a run is "
                        "still improving. Kept by default for that reason.")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore an existing resume.pt and start over. Without "
                        "this, a run whose --out already holds resume.pt PICKS UP "
                        "where it stopped, so the same command can be run in as "
                        "many short sessions as you like.")
    p.add_argument("--no-choose-payment", action="store_true",
                   help="let the engine pick which runes pay a Power cost "
                        "(the pre-2026-10-07 rule) instead of asking the policy")
    p.add_argument("--no-caffeinate", action="store_true",
                   help="do not hold a macOS wake lock for the run. By default a "
                        "training run starts `caffeinate -dimsu -w <pid>`, which "
                        "keeps the machine awake exactly as long as this process "
                        "lives (BACKLOG B5: idle sleep cost three runs).")
    p.add_argument("--max-hours", type=float, default=None, metavar="H",
                   help="stop cleanly after H hours, whatever `iterations` says. "
                        "Sizing a run by iteration count needs a throughput "
                        "guess, and this machine throttles under sustained load "
                        "-- v8-postfix was sized off 296 st/s, settled at 119, "
                        "and a 4h ask took 7h23m. Set `iterations` optimistically "
                        "and let the clock end it. Exploiter runs are ~2x more "
                        "env-steps per iteration than the rollout suggests, "
                        "because the frozen champion makes half the moves.")
    p.add_argument("--lr", type=float, default=None,
                   help=f"Adam learning rate (default {HP.lr}). A warm-started "
                        "run restarts the anneal with fresh optimizer state, so "
                        "the first update takes a full-size step from a policy "
                        "that was already converged -- measured kl 0.46 against "
                        "a healthy ~0.03. Lower it for any fine-tune.")
    p.add_argument("--lr-warmup", type=int, default=0, metavar="N",
                   help="ramp the LR linearly over the first N iterations "
                        "(BACKLOG W1). A warm start's fresh Adam otherwise takes "
                        "a full-size first step: kl 0.3-0.8 against ~0.03. "
                        "Counted on the absolute iteration, so a resume past N "
                        "is unaffected. Default 0 (off).")
    p.add_argument("--ent-coef", type=float, default=None,
                   help=f"entropy bonus on the normalized entropy "
                        f"(default {HP.ent_coef})")
    p.add_argument("--pool-frac", type=float, default=None,
                   help="fraction of envs facing a frozen pool opponent "
                        f"(default {HP.pool_frac}); --exploit forces 1.0")
    p.add_argument("--bo3", action="store_true",
                   help="BO3 MATCHES (BACKLOG D2/D2b). Each env slot plays real "
                        "best-of-3 matches: sideboarding, a secret battlefield "
                        "pick (burned after a won game) and the loser's choice "
                        "of play order are learned decisions; games 2-3 know "
                        "what earlier games showed. Reward stays per game. Pair "
                        "with --init to grow a pre-Bo3 checkpoint into it.")
    p.add_argument("--matchup-pfsp", action="store_true",
                   help="MATCHUP-AWARE PFSP (BACKLOG #21). Keep the PFSP win "
                        "record per (frozen opponent, deck the learner is "
                        "piloting) instead of per opponent, and sample the pair "
                        "JOINTLY toward cells the learner is losing. The "
                        "per-opponent average is what hid v10's ~45 Elo: pinning "
                        "the pool to one opponent was the only lever that has "
                        "ever measurably worked, and it worked by deleting the "
                        "averaging. Requires --real-decks (a procedural deal has "
                        "no deck identity to key on). OFF by default.")
    p.add_argument("--pool-init", default=None, metavar="A.pt,B.pt",
                   help="comma-separated checkpoints PINNED into the opponent "
                        "pool for the whole run (BACKLOG #22). Never evicted and "
                        "never below the pin floor, because PFSP drives an "
                        "opponent it beats toward zero sample share -- which is "
                        "how a population forgets what it used to beat. v13 "
                        "showed a pool of self-snapshots has a FLAT difficulty "
                        "surface, which makes any PFSP weighting a no-op")
    p.add_argument("--no-anneal", action="store_true",
                   help="hold the learning rate at --lr instead of cosine-"
                        "annealing it to 0.1x over `iterations`. The anneal is "
                        "the remaining suspect for policies that stop moving: "
                        "v12 doubled the entropy bonus and still decayed to "
                        "kl 0.0006 by iteration 259 of 300, where the effective "
                        "rate is ~1/7 of base, so no entropy term can buy back "
                        "a step size that is nearly zero.")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    table = full_table()
    cfg = Config().at_victory_score(a.victory)
    # **`--real-decks` implies spells.** This cost a whole 4-hour run: v9-exploiter
    # was launched with --real-decks but WITHOUT --spells, so it trained on the
    # units-only game while its frozen champion had been trained with spells. The
    # champion was crippled by the config, the learner adapted to the cheaper
    # game, and `vs_opp` climbed to 57% while the checkpoint actually got worse.
    # The tell was visible in the very first eval -- vs_greedy 55.0% at iteration
    # 20, against the champion's 70.5% -- an instant drop no amount of drift
    # explains. Nothing in the log said which game was being played.
    if a.spells or (a.real_decks and not a.units_only):
        cfg = dc_replace(cfg, units_only=False)
    cfg = dc_replace(cfg, choose_payment=not a.no_choose_payment)
    if a.units_only and a.spells:
        p.error("--units-only and --spells contradict each other")
    hp = HP(n_envs=a.envs, rollout=a.rollout)
    if a.lr is not None:
        hp.lr = a.lr
    if a.ent_coef is not None:
        hp.ent_coef = a.ent_coef
    if a.pool_frac is not None:
        hp.pool_frac = a.pool_frac
    if a.no_anneal:
        hp.anneal_lr = False
    hp.lr_warmup = max(0, a.lr_warmup)
    # Normalized onto HP so the resume check compares the same shape it stored.
    a.pool_init = tuple(x.strip() for x in (a.pool_init or "").split(",")
                        if x.strip())
    hp.pool_init = a.pool_init
    if a.pool_init and a.exploit:
        p.error("--pool-init and --exploit contradict: --exploit pins the pool "
                "to ONE frozen champion and must never gain an entry")
    if a.matchup_pfsp:
        if not a.real_decks:
            p.error("--matchup-pfsp needs --real-decks: the matchup key is the "
                    "name of the decklist the learner is piloting, and the "
                    "procedural deals have no decklists")
        hp.matchup_pfsp = True
    if a.bo3:
        if not a.real_decks:
            p.error("--bo3 needs --real-decks: a match is two registered lists, "
                    "with battlefields and sideboards the procedural deals lack")
        if a.matchup_pfsp:
            p.error("--bo3 and --matchup-pfsp both own the deal; pick one")
        hp.bo3 = True
    if a.exploit:
        # Exploiter mode: the pool holds exactly one entry, the champion, and
        # never gains another. `pool_every = 0` is the load-bearing half -- a
        # learner snapshotted into its own opponent pool is ordinary self-play
        # again, and the measurement would quietly stop meaning anything.
        hp.pool_frac, hp.pool_every = 1.0, 0
    if a.real_decks:
        pool = select_decks(table, a.decks, a.min_coverage)
        held: list = []
        if a.holdout > 0:
            # Deterministic in `--seed`, so the split is reproducible and a
            # later run can be compared against this one honestly.
            _r = np.random.default_rng(a.seed)
            idx = _r.permutation(len(pool))
            held = [pool[int(i)] for i in idx[:a.holdout]]
            pool = [pool[int(i)] for i in idx[a.holdout:]]
            print(f"     holdout: training on {len(pool)}, holding back "
                  f"{len(held)} -> " + ", ".join(d.name for d in held))
        if a.decks:
            print(f"     decks: {len(pool)} -- "
                  + ", ".join(d.name for d in pool)
                  + ("  (MIRROR match)" if len(pool) == 1 else ""))
        spec = (None if hp.matchup_pfsp
                else DealSpec("bo3" if hp.bo3 else "pool", pool, a.min_coverage))
    else:
        spec = DealSpec("v1" if a.spells else "v0")
    # Built from the same spec the workers build from (P2), so main's own deal
    # -- eval, `deal_stats` -- is the deal the envs play.
    deal = (MatchupDealer(table, pool, a.min_coverage) if spec is None
            else spec.build(table))
    if a.workers > 0 and hp.matchup_pfsp:
        p.error("--matchup-pfsp cannot run with --workers: its dealer takes "
                "each deal's (opponent, deck) choice from the main process's "
                "opponent pool, and the deal happens in the workers")

    if a.update_device == "mps" and not torch.backends.mps.is_available():
        p.error("--update-device mps: MPS is not available on this machine")
    tr = Trainer(table, cfg, deal, hp, device=a.device, seed=a.seed,
                 update_device=a.update_device, workers=a.workers,
                 deal_spec=spec)
    if isinstance(deal, MatchupDealer):
        # Bound only now, because the pool and the RNG belong to the Trainer.
        # Until this line the dealer is `deck_pool_deal`, which is what makes
        # the construction-time `reset` above uniform and harmless.
        deal.bind(tr.pool, tr.rng, hp.pool_frac)
        print(f"     MATCHUP PFSP: joint (opponent, deck) sampling over "
              f"{hp.pool_capacity} x {len(deal.names)} = "
              f"{hp.pool_capacity * len(deal.names)} cells")
    if a.init:
        ck = torch.load(a.init, map_location=a.device, weights_only=False)
        # The observation layout is baked into the first and last layers, so a
        # checkpoint from a different encoder would load into the wrong shapes
        # or, worse, the right shapes with different meanings.
        assert shapes_growable(ck["shapes"], tr.vec.shapes()), (
            f"checkpoint was trained on a different observation layout:\n"
            f"  ckpt {ck['shapes']}\n  here {tr.vec.shapes()}")
        _check_cfg(ck, cfg, f"--init {a.init}", strict=False)
        if load_weights(tr.net, ck, tr.vec.shapes()):
            print(f"     grown from an older observation layout -- plays "
                  f"exactly as it did; the new inputs start at zero weight")
        print(f"warm-started from {a.init} (iter {ck.get('iter')}, "
              f"eval {ck.get('eval')})")

    # --- pinned opponents (BACKLOG #22) ----------------------------------
    # v13 measured its own pool and found the difficulty surface FLAT: chi2/dof
    # 1.02 by deck, 0.51 by opponent. That was guaranteed -- the pool was 20
    # snapshots of its own run 10 iterations apart, i.e. 20 near-copies. PFSP
    # reweights a difficulty surface, so on a flat one it is a no-op by
    # construction, and that is the whole explanation of v13's null.
    #
    # Pinning genuinely different policies is the cheapest way to make the
    # surface non-flat. The ladder spans ~46 Elo across the lineage (v8 +87,
    # v7 +91, v10 +133), which is real variance a flat self-snapshot pool cannot
    # contain.
    if a.pool_init:
        for path in a.pool_init:
            pck = torch.load(path, map_location=a.device, weights_only=False)
            assert shapes_growable(pck["shapes"], tr.vec.shapes()), (
                f"pinned opponent {path} was trained on a different observation "
                f"layout:\n  ckpt {pck['shapes']}\n  here {tr.vec.shapes()}")
            # Strict, like --exploit: a pinned opponent playing a different GAME
            # (spells off, a different victory score) would make `vs_opp`
            # meaningless, and `ck["shapes"]` is byte-identical either way -- the
            # lesson v9 cost four hours to learn.
            _check_cfg(pck, cfg, f"--pool-init {path}", strict=True)
            _pn = RiftboundNet(tr.vec.shapes(),
                               privileged_critic=hp.privileged_critic).to(a.device)
            load_weights(_pn, pck, tr.vec.shapes())
            _pn.eval()
            _nm = Path(path).parent.name or Path(path).stem
            if pck.get("cfg") is None:
                # v7/v8/v10 predate config fingerprinting, so `_check_cfg` can
                # only print a note. For THESE three the verification exists and
                # is stronger than a stored config: the ladder rated them in this
                # exact game (victory 8, spells, real decks) at +91/+87/+133. Say
                # so at launch rather than let a passing check look like proof --
                # v9-exploiter is in this same directory and WAS trained on
                # another game.
                print(f"     NOTE: {_nm} predates config fingerprinting. Verify "
                      f"by eval (the ladder did) -- a units-only checkpoint "
                      f"would load here without complaint.")
            _pe = tr.pool.pin(_pn, int(pck.get("iter", 0)), name=_nm)
            _pe.old_payment = not getattr(_pn, "knows_payment", True)
        print(f"     PINNED OPPONENTS: {len(a.pool_init)} "
              f"({', '.join(Path(x).parent.name for x in a.pool_init)}) "
              f"-- never evicted, each held at >= {tr.pool.pin_floor:.0%} "
              f"of the sample mass")

    if a.exploit:
        ck = torch.load(a.exploit, map_location=a.device, weights_only=False)
        assert shapes_growable(ck["shapes"], tr.vec.shapes()), (
            f"the frozen champion was trained on a different observation "
            f"layout:\n  ckpt {ck['shapes']}\n  here {tr.vec.shapes()}")
        _check_cfg(ck, cfg, f"--exploit {a.exploit}", strict=True)
        _champ = RiftboundNet(tr.vec.shapes(),
                              privileged_critic=hp.privileged_critic).to(a.device)
        load_weights(_champ, ck, tr.vec.shapes())
        _champ.eval()
        tr.pool.add(_champ, int(ck.get("iter", 0)), name="champion")
        # `_assign` only runs when an episode ENDS, so without this the first
        # rollout would be pure self-play -- the one stretch of the run whose
        # numbers would be wrong, and the easiest kind of wrong to miss.
        for _i in range(hp.n_envs):
            tr._assign(_i)
        print(f"EXPLOITER: every episode is learner vs frozen {a.exploit} "
              f"(iter {ck.get('iter')}); pool is pinned to it and never grows. "
              f"`vs_opp=` is the measurement.")
        if a.init != a.exploit:
            print("     note: --init differs from --exploit, so the run does "
                  "NOT start from a known 50%.")

    print(f"PPO: victory={a.victory} envs={hp.n_envs} rollout={hp.rollout} "
          f"device={a.device} params={count_params(tr.net):,}")
    print(f"     game: {'UNITS ONLY (no spells)' if cfg.units_only else 'units + spells'}"
          f"  decks: {'real decklists' if a.real_decks else ('procedural v1' if a.spells else 'procedural v0')}")
    print(f"     exit criteria: >=90% vs random, >=65% vs greedy")
    if not hp.anneal_lr:
        # Printed because a run's log is the only record of which knob it tested,
        # and "the lr was flat" is invisible in the per-iteration line.
        print(f"     LR ANNEAL OFF: lr held at {hp.lr:g} for the whole run "
              f"(default cosines to {0.1 * hp.lr:g})")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- resume ------------------------------------------------------------
    # **The optimizer state is the point.** Restarting from `last.pt` with
    # `--init` restores the WEIGHTS but gives Adam fresh moments, which takes a
    # full-size step from an already-converged policy -- measured kl 0.46 against
    # a healthy 0.03. A resume that did that would not be a continuation, it
    # would be a new run that happens to start in the same place.
    start_it = 0
    best = -1.0
    vs_opp_ewma: float | None = None
    eval_ewma: float | None = None
    resume_path = out / "resume.pt"
    if resume_path.exists() and not a.no_resume:
        rs = torch.load(resume_path, map_location=a.device, weights_only=False)
        assert rs["shapes"] == tr.vec.shapes(), "resume.pt has a different layout"
        _check_cfg(rs, cfg, "resume.pt", strict=True)
        tr.net.load_state_dict(rs["net"])
        tr.opt.load_state_dict(rs["opt"])
        tr.global_step = int(rs["global_step"])
        tr.rng.bit_generator.state = rs["np_rng"]
        tr.gen.set_state(rs["torch_gen"])
        # **Hyperparameters are not restored, only compared.** They come from the
        # command line, so a resume that omitted `--ent-coef 0.02` would silently
        # continue at the 0.01 default and destroy a single-variable experiment
        # mid-flight, with nothing in the log to show it. Learned the hard way an
        # hour after building resume: v12 exists to test exactly that one knob.
        old_hp = rs.get("hp")
        if old_hp:
            from dataclasses import asdict as _asdict
            now_hp = _asdict(hp)
            drift = {k: (old_hp[k], now_hp[k]) for k in old_hp
                     if k in now_hp and old_hp[k] != now_hp[k]}
            if drift:
                raise SystemExit(
                    "\nresume.pt was written with DIFFERENT hyperparameters:\n"
                    + "\n".join(f"       {k}: was {a!r}, now {b!r}"
                                 for k, (a, b) in sorted(drift.items()))
                    + "\n\n  Pass the same flags as the original launch, or"
                      " --no-resume to start over.\n  Resuming with a changed"
                      " knob makes the run two experiments spliced together.")
            print(f"     hyperparameters: match the original launch")
        start_it = int(rs["iter"])
        best = float(rs["best"])
        vs_opp_ewma = rs.get("vs_opp_ewma")
        eval_ewma = rs.get("eval_ewma")
        # **Rebuild the opponent pool from the snapshots on disk.** The pool
        # itself is memory-only, so a resume used to restart it EMPTY -- which
        # silently turns a pool run into pure self-play until `pool_every`
        # refills it, and over a 20-entry capacity that is most of a session.
        # The tell is `vs_opp` vanishing from the log, because no env has a
        # frozen opponent at all. Now that snapshots are persisted they are
        # exactly what the pool needs; `--exploit` is skipped because its pool is
        # pinned to the champion and must never gain an entry.
        if hp.pool_every and not a.exploit:
            snaps = sorted((out / "snapshots").glob("iter*.pt"))
            for sp in snaps[-hp.pool_capacity:]:
                sc = torch.load(sp, map_location=a.device, weights_only=False)
                if sc.get("shapes") != tr.vec.shapes():
                    continue
                _n = RiftboundNet(tr.vec.shapes(),
                                  privileged_critic=hp.privileged_critic)
                _n.load_state_dict(sc["net"])
                _n.eval()
                tr.pool.add(_n, int(sc.get("iter", 0)))
            if snaps:
                print(f"     pool rebuilt from {len(tr.pool.entries)} snapshot(s) "
                      f"on disk ({snaps[0].name}..{snaps[-1].name})")
            else:
                print(f"     NOTE: no snapshots on disk, so the opponent pool "
                      f"restarts EMPTY -- this resumed run is pure self-play "
                      f"until iteration {((start_it // hp.pool_every) + 1) * hp.pool_every}")
        # **Re-attach the win records, flagged runs only.** The matchup matrix is
        # the state the sampler reads, and a resume that dropped it would restart
        # every cell at the 0.5 prior -- a 300-iteration run resumed twice would
        # spend most of its life sampling uniformly, with nothing in the log to
        # say so. Outside the snapshot branch on purpose, so an EXPLOITER resume
        # (whose single entry is added above, from --exploit) keeps its matrix
        # too. Gated on the flag because restoring the per-opponent record is
        # itself a behaviour change and an unflagged run must reproduce today
        # exactly. By NAME, so a pruned or reordered snapshot set degrades to
        # the prior instead of mislabelling a record.
        if hp.matchup_pfsp and tr.pool.entries:
            _hit = tr.pool.load_records(rs.get("pool_records") or {})
            print(f"     matchup matrix: records re-attached to {_hit} of "
                  f"{len(tr.pool.entries)} pool entries")
        # Envs are fresh, so the handful of episodes in flight when it stopped
        # are lost. Re-pick opponents and seats for every slot.
        for _i in range(hp.n_envs):
            tr._assign(_i)
        print(f"RESUMED from {resume_path} at iteration {start_it} "
              f"({tr.global_step:,} steps already done, best={best:.4f})")
    elif resume_path.exists():
        print(f"--no-resume: ignoring {resume_path} and starting over")

    if not a.no_caffeinate:
        _hold_wake_lock()
    t0 = time.time()
    # **`--max-hours` is budgeted in COMPUTE time, not wall-clock.** macOS put
    # this machine to sleep inside three consecutive training runs: v14 lost
    # 3,279s frozen plus ~4,079s to DarkWake throttling -- 47% of its wall clock
    # -- and `time.time()` counts every second of it against the budget, so the
    # run stopped at iteration 112 of 300 having done barely half the work it was
    # asked for. `CLOCK_UPTIME_RAW` does not advance while the system is asleep,
    # so "4 hours" now means four hours of actually running. It is not a
    # substitute for `caffeinate -dimsu` in front of the launch, which is what
    # stops the sleep happening at all; this only stops the budget being eaten by
    # it.
    t0_compute = time.clock_gettime(time.CLOCK_UPTIME_RAW)
    # Steps THIS SESSION. `tr.global_step` is cumulative across resumes, so
    # dividing it by this session's elapsed time reports a throughput that was
    # never achieved -- it read 1286 st/s against a real 330 in testing.
    step0 = tr.global_step
    # **Per-iteration timing, because the cumulative average hid a 4x cliff for
    # days across three runs.** `st/s` below is (steps this session)/(elapsed
    # this session), so a sudden collapse decays into the average instead of
    # showing up as one: v14's log read "119 st/s" when its healthy rate was 275
    # and its post-sleep rate was 70. The instantaneous number is the one that
    # shows a problem while the run is still going.
    _it_steps, _it_time = tr.global_step, t0_compute
    _it_history: list[float] = []
    stopped_early = False

    def _save_resume(it: int) -> None:
        """Written via a temp file: a crash mid-write must not eat the resume."""
        tmp = out / "resume.pt.tmp"
        from dataclasses import asdict as _asdict
        torch.save({"net": tr.net.state_dict(), "opt": tr.opt.state_dict(),
                    "shapes": tr.vec.shapes(), "cfg": _cfg_fingerprint(cfg),
                    "hp": _asdict(hp),
                    "iter": it, "global_step": tr.global_step, "best": best,
                    "vs_opp_ewma": vs_opp_ewma, "eval_ewma": eval_ewma,
                    # The pool's NETS come back from <out>/snapshots; its win
                    # record does not exist on disk anywhere else, and with
                    # --matchup-pfsp that record is the matrix the sampler runs
                    # on. Tensor-free, keyed by entry name (BACKLOG #21).
                    "pool_records": tr.pool.records(),
                    "np_rng": tr.rng.bit_generator.state,
                    "torch_gen": tr.gen.get_state(),
                    "engine": _engine_rev()}, tmp)
        tmp.replace(resume_path)

    it = start_it
    for it in range(start_it + 1, a.iterations + 1):
        if a.max_hours and (time.clock_gettime(time.CLOCK_UPTIME_RAW)
                            - t0_compute) >= a.max_hours * 3600.0:
            print(f"\n--max-hours {a.max_hours} reached at iteration {it - 1} "
                  f"of {a.iterations}; stopping cleanly.", flush=True)
            stopped_early = True
            # Save HERE as well as at evals: evals are 20 iterations apart, so a
            # clock stop between them would otherwise throw away up to 20
            # iterations of real work.
            _save_resume(it - 1)
            print(f"     resume.pt written at iteration {it - 1}. "
                  f"Rerun the same command to continue.", flush=True)
            break
        roll = tr.collect()
        warm = min(1.0, it / hp.lr_warmup) if hp.lr_warmup else 1.0
        stats = tr.update(roll, (it - 1) / max(1, a.iterations), warm)  # absolute
        dt = time.time() - t0
        # Instantaneous rate for THIS iteration, plus a loud note when one takes
        # far longer than the run's own median -- the signature of system sleep
        # or thermal throttling, which is otherwise invisible until the run ends
        # short. Median rather than mean because an eval iteration is
        # legitimately several times a plain one and would drag a mean around.
        _now_c = time.clock_gettime(time.CLOCK_UPTIME_RAW)
        _d_t, _d_s = _now_c - _it_time, tr.global_step - _it_steps
        _inst = _d_s / _d_t if _d_t > 0 else 0.0
        _stall = ""
        if _it_history:
            _med = sorted(_it_history)[len(_it_history) // 2]
            if _d_t > 3.0 * _med:
                _stall = (f"  <-- STALL: this iteration took {_d_t:.0f}s against "
                          f"a median of {_med:.0f}s. Lost ~{_d_t - _med:.0f}s to "
                          f"system sleep or thermal throttling; launch under "
                          f"`caffeinate -dimsu`")
        _it_history.append(_d_t)
        _it_steps, _it_time = tr.global_step, _now_c
        # The learner's share against the frozen pool. In exploiter mode the
        # pool is one entry so this is exactly "how far past 50% has it got",
        # which is the whole run; otherwise it is a useful aside and is only
        # printed when some envs actually faced a frozen self.
        vs_opp = (roll.opp_wins / roll.opp_games) if roll.opp_games else None
        if vs_opp is not None:
            vs_opp_ewma = (vs_opp if vs_opp_ewma is None
                           else 0.8 * vs_opp_ewma + 0.2 * vs_opp)
        vs_opp_str = ("" if vs_opp is None else
                      f"| vs_opp {vs_opp:.1%} (ewma {vs_opp_ewma:.1%}, "
                      f"n={roll.opp_games}) ")
        print(f"[{it:4d}/{a.iterations}] steps={tr.global_step:>8,} "
              f"eps={roll.episodes:>4} len={np.mean(roll.ep_len):5.1f} "
              f"turns={np.mean(roll.ep_turns):4.1f} "
              f"seat0={roll.seat0_wins / max(1, roll.episodes):.0%} "
              f"trunc={roll.truncated / max(1, roll.episodes):.1%} "
              f"loop={roll.loops / max(1, roll.episodes):.1%} "
              f"ovf={roll.overflows / max(1, roll.episodes):.1%} | "
              f"ent={stats['ent']:.3f} kl={stats['kl']:.4f} "
              f"clip={stats['clipfrac']:.2f} ev={stats['explained_var']:+.2f} "
              f"{vs_opp_str}| {_inst:.0f} st/s"
              f" (avg {(tr.global_step - step0) / dt:.0f})"
              + _stall, flush=True)

        if hp.pool_every and it % hp.pool_every == 0:
            tr.pool.add(tr.net, it)
            # **Persist it.** The pool has always been memory-only, so every run
            # so far generated 20-50 intermediate policies and threw all of them
            # away, leaving two checkpoints. That is why "did it plateau?" was
            # unanswerable on v7 and v8 -- the evidence existed and was discarded.
            # `rl/ladder.py` turns these into rungs. ~4MB each.
            if not a.no_snapshots:
                snaps = out / "snapshots"
                snaps.mkdir(parents=True, exist_ok=True)
                torch.save({"net": tr.net.state_dict(),
                            "shapes": tr.vec.shapes(), "iter": it,
                            "cfg": _cfg_fingerprint(cfg),
                            "engine": _engine_rev()},
                           snaps / f"iter{it:05d}.pt")

        if a.eval_every and it % a.eval_every == 0:
            r = report(tr.net, table, cfg, deal, n=a.eval_games, device=a.device)
            flag = "  <-- PHASE 4 EXIT MET" if r["pass"] else ""
            lock = (f"  livelocked {r['livelocks']}" if r.get("livelocks")
                    else "")
            print(f"       eval: vs_random {r['vs_random']:.1%}  "
                  f"vs_greedy {r['vs_greedy']:.1%}  "
                  f"elo(greedy=0) {elo_from_winrate(r['vs_greedy']):+.0f}"
                  f"{lock}{flag}\n       {tr.pool.summary()}", flush=True)
            # In exploiter mode the objective is beating the CHAMPION, not
            # beating greedy, so selecting `best.pt` on vs_random+vs_greedy
            # would hand back the checkpoint that best solved a different
            # problem. The smoothed rate is used rather than this iteration's,
            # which at n~=200 swings several points on noise alone.
            #
            # **And SMOOTH the ordinary case too.** `vs_greedy` is saturated
            # (MEASURING.md), so across a run it is a flat noisy series -- v11
            # gave 14 evals with mean 71.2% and sd 2.7. Selecting the max over 14
            # such draws is selecting noise: it picked iteration 20 of 280, and
            # an eval at n=60 is +/-6pp, so the max is biased high by over a
            # standard deviation. Every downstream use of `best.pt` -- the next
            # run's `--init`, `play.py`, the ladder -- then inherits a lucky
            # sample. An ewma over the eval series costs nothing and removes most
            # of that bias.
            raw = r["vs_random"] + r["vs_greedy"]
            eval_ewma = raw if eval_ewma is None else 0.6 * eval_ewma + 0.4 * raw
            score = (vs_opp_ewma if (a.exploit and vs_opp_ewma is not None)
                     else eval_ewma)
            if score > best:
                best = score
                torch.save({"net": tr.net.state_dict(), "shapes": tr.vec.shapes(),
                            "iter": it, "eval": r, "engine": _engine_rev(),
                            "vs_opp": vs_opp_ewma, "exploit": a.exploit,
                            "selected_on": score, "eval_ewma": eval_ewma,
                            "cfg": _cfg_fingerprint(cfg)},
                           out / "best.pt")
            # **Also roll `last.pt` forward at every eval, not just at the end.**
            # A multi-hour run that dies overnight would otherwise leave only
            # `best.pt` (which lags, by design) and nothing from the last hours.
            # Written via a temp file so a crash mid-write cannot corrupt it.
            _tmp = out / "last.pt.tmp"
            torch.save({"net": tr.net.state_dict(), "shapes": tr.vec.shapes(),
                        "iter": it, "eval": r, "engine": _engine_rev(),
                        "cfg": _cfg_fingerprint(cfg)}, _tmp)
            _tmp.replace(out / "last.pt")
            _save_resume(it)
    torch.save({"net": tr.net.state_dict(), "shapes": tr.vec.shapes(),
                "iter": (it - 1 if stopped_early else a.iterations),
                "stopped_early": stopped_early, "vs_opp": vs_opp_ewma,
                "engine": _engine_rev(), "cfg": _cfg_fingerprint(cfg)},
               out / "last.pt")
    if not stopped_early:
        # Finished. Leaving resume.pt behind would make the same command look
        # like it had work left, and silently re-save without training.
        resume_path.unlink(missing_ok=True)
        print(f"reached iteration {a.iterations}; resume.pt removed.")
    tr.close()          # worker processes (P2); a no-op when serial

    if a.per_deck and a.real_decks:
        from rl.eval import per_deck, print_per_deck
        print(f"\nper-deck breakdown vs greedy "
              f"({a.per_deck} deals x 2 seats x {len(pool)} decks)")
        print_per_deck(per_deck(tr.net, table, cfg, pool, n=a.per_deck,
                                device=a.device))
        if held:
            print(f"\nHELD-OUT decks -- never trained on ({len(held)}):")
            print_per_deck(per_deck(tr.net, table, cfg, held, n=a.per_deck,
                                    device=a.device))
    return 0


if __name__ == "__main__":
    sys.exit(main())
