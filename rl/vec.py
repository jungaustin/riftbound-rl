"""Synchronous vector env and observation batching -- PLAN.md §5.4.

A Python engine at ~4.5k decisions/sec is the bottleneck, so the only thing that
matters here is amortizing network inference across many games. This runs `n`
independent episodes in lockstep and stacks their observations into batched
arrays; PLAN.md's target layout is 8 worker processes x 32 envs each, and this
is the inner 32.

**Every env is at a decision point for exactly one seat**, and which seat that
is varies across the batch. That is fine and it is the point: one shared network
plays both seats, observations are canonicalized (`obs.py`), so a batch row does
not need to know whose turn it is. `to_move` is returned anyway, because the
trainer needs it to route the transition into the right per-seat buffer.

**Auto-reset.** When an episode ends, its slot is immediately refilled with a
fresh game and the returned observation belongs to the *new* episode. The
finished episode's outcome is reported in the same `step` via `rewards` and
`done`. This is the standard vector contract and it is also the standard place
to introduce the reward-attribution bug (§5.3 gotcha 1), so: `rewards[i]` is
indexed by **seat**, both entries are filled at once, and neither has anything
to do with whoever happened to act last in slot `i`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.config import Config
from rl.engine.cardtable import CardTable
from rl.env import RiftboundEnv
from rl.obs import Encoder, Obs


@dataclass(frozen=True)
class BatchObs:
    """`n` observations stacked. Field names match `Obs` so a net can take either."""
    zones: dict[str, np.ndarray]      # {name: [n, slots, row_dim]}
    zone_mask: dict[str, np.ndarray]  # {name: [n, slots]}
    globals: np.ndarray               # [n, GLOBAL_DIM]
    legal_actions: np.ndarray         # [n, A_max, act_dim]
    action_mask: np.ndarray           # [n, A_max]
    to_move: np.ndarray               # [n] int8
    n_legal: np.ndarray               # [n] int16
    privileged: np.ndarray | None     # [n, priv_dim], critic-only

    def __len__(self) -> int:
        return int(self.globals.shape[0])


def batch(obs: list[Obs]) -> BatchObs:
    keys = obs[0].zones.keys()
    return BatchObs(
        zones={k: np.stack([o.zones[k] for o in obs]) for k in keys},
        zone_mask={k: np.stack([o.zone_mask[k] for o in obs]) for k in keys},
        globals=np.stack([o.globals for o in obs]),
        legal_actions=np.stack([o.legal_actions for o in obs]),
        action_mask=np.stack([o.action_mask for o in obs]),
        to_move=np.array([o.to_move for o in obs], np.int8),
        n_legal=np.array([o.n_legal for o in obs], np.int16),
        privileged=(None if obs[0].privileged is None
                    else np.stack([o.privileged for o in obs])),
    )


def slot_seed(seed0: int, slot: int, n_envs: int, k: int) -> int:
    """The seed of slot `slot`'s `k`-th episode (BACKLOG P2).

    Per slot, not a global counter: a global counter hands out seeds in the
    order games FINISH, which differs between the serial loop and the worker
    processes (`rl/parallel.py`). Keyed on the slot, the two deal identical
    games, which is what `test_parallel [1]` checks. Every seed is still used
    exactly once within a run: slot + n_envs * k is a bijection onto 0, 1, 2...
    """
    return seed0 + slot + n_envs * k


class VecRiftbound:
    """`n_envs` games advanced in lockstep, with auto-reset.

    `deal_fn(seed) -> (decks, rune_decks, battlefields[, legends[, champions]])`
    is called once per
    episode. Making it a callable rather than fixed decklists is what lets the
    deck-evaluator work (§1.1) reuse this loop unchanged: sample a matchup per
    episode and the same machinery trains a policy that generalizes across
    decks instead of memorizing one.
    """

    def __init__(self, table: CardTable, cfg: Config, n_envs: int, deal_fn,
                 seed0: int = 0, auto_pass: bool = True,
                 privileged: bool = True, check: bool = False) -> None:
        self.cfg = cfg
        self.deal_fn = deal_fn
        self.enc = Encoder(table, cfg, privileged=privileged)
        self.envs = [RiftboundEnv(table, cfg, encoder=self.enc,
                                  auto_pass=auto_pass, check=check)
                     for _ in range(n_envs)]
        self.seed0 = seed0
        # Episodes dealt per slot, for `slot_seed`.
        self._dealt = [0] * n_envs
        self.episodes = 0
        self.truncations = 0

    def _seed(self, slot: int) -> int:
        s = slot_seed(self.seed0, slot, len(self.envs), self._dealt[slot])
        self._dealt[slot] += 1
        return s

    def continuing(self, slot: int) -> bool:
        """Is `slot`'s current game a later game of a match under way (Bo3)?
        The trainer keeps the learner's seat and opponent fixed for those."""
        cont = getattr(self.deal_fn, "continuing", None)
        return bool(cont(slot)) if cont is not None else False

    def _fresh(self, env: RiftboundEnv, slot: int) -> Obs:
        seed = self._seed(slot)
        # A match dealer (Bo3) needs to know WHICH slot and how its last game
        # ended, because the next game of a match continues from it.
        for_slot = getattr(self.deal_fn, "for_slot", None)
        if for_slot is not None:
            args, ctx = for_slot(slot, env.state, seed)
            return env.reset(seed, *args, match=ctx)
        return env.reset(seed, *self.deal_fn(seed))

    def reset(self) -> BatchObs:
        return batch([self._fresh(e, i) for i, e in enumerate(self.envs)])

    def step(self, action_indices) -> tuple[BatchObs, np.ndarray, np.ndarray,
                                            list[dict]]:
        """Advance every env by one decision.

        Returns `(obs, rewards[n, 2], done[n], infos)`. On a slot that finished,
        `obs` is already the first observation of its replacement episode.
        """
        n = len(self.envs)
        rewards = np.zeros((n, 2), np.float32)
        done = np.zeros(n, bool)
        infos: list[dict] = []
        out: list[Obs] = []

        for i, env in enumerate(self.envs):
            r = env.step(int(action_indices[i]))
            info = {"action": r.info["action"]}
            if r.done:
                rewards[i] = r.rewards
                done[i] = True
                self.episodes += 1
                self.truncations += int(r.truncated)
                info["episode"] = env.summary()
                out.append(self._fresh(env, i))
            else:
                out.append(r.obs)
            infos.append(info)

        return batch(out), rewards, done, infos

    def shapes(self) -> dict:
        return self.enc.shapes()

    @property
    def truncation_rate(self) -> float:
        """§5.3 gotcha 7 -- a first-class metric. A rising rate means the agent
        found a way to stall, and that eats a training run quietly."""
        return self.truncations / max(1, self.episodes)
