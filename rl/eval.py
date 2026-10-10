"""Evaluation -- PLAN.md Phase 4 exit and §7.

Everything here is **seat-swapped on paired seeds**: each deal is played twice,
once with the network on each side. Without that, a win rate is contaminated by
whatever first-player edge the position has (measured: random-vs-random sits at
50.0% only because it is paired this way, and the first player wins 53.7% of
random games), and by which side happened to be dealt the better opening.

The opponents are the Phase 2 baselines, unchanged, so Phase 4's numbers are
directly comparable to the ones the engine was validated with.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from rl.agents.greedy import greedy_agent
from rl.config import Config
from rl.engine import actions as A_
from rl.engine import game
from rl.engine.state import BoardOverflow
from rl.env import RiftboundEnv
from rl.nets import to_torch
from rl.obs import Encoder
from rl.vec import batch


@torch.no_grad()
def net_choice(net, obs, device: str, deterministic: bool,
               generator=None) -> int:
    from rl.nets import untrained_choice
    forced = untrained_choice(net, obs)
    if forced is not None:
        return forced
    t = to_torch(batch([obs]), device)
    idx, *_ = net.act(t, deterministic=deterministic, generator=generator)
    return int(idx.item())


# Games abandoned as livelocked, this process. Surfaced in the eval line so a
# nonzero count is visible rather than buried in a log file.
LIVELOCKS: list[dict] = []


def _dump_livelock(env, table, seed: int, net_seat: int, steps: int) -> None:
    """Record enough to reproduce an abandoned game, and say so once."""
    s = env.state
    info = {
        "seed": seed, "net_seat": net_seat, "steps": steps,
        "turn": int(s.turn), "phase": int(s.phase),
        "showdown_bf": int(s.showdown_bf), "showdown_step": int(s.showdown_step),
        "n_chain": int(s.n_chain), "n_trig": int(s.n_trig),
        "priority": int(s.priority), "passes": int(s.passes),
        "points": s.points.tolist(),
        "hands": [int(x) for x in s.n_hand],
        "chain": [table.names[int(s.chain[i, 0])] for i in range(int(s.n_chain))],
        "legal": [repr(a) for a in env.legal[:12]],
    }
    LIVELOCKS.append(info)
    if len(LIVELOCKS) == 1:
        print(f"  !! livelocked game abandoned: {info}", flush=True)


@torch.no_grad()
def play_one(net, opponent, table, cfg, enc: Encoder, seed: int, net_seat: int,
             deal_fn, device: str = "cpu", deterministic: bool = False,
             generator=None) -> int:
    """One game. Returns the winning seat, or -1."""
    env = RiftboundEnv(table, cfg, encoder=enc)
    obs = env.reset(seed, *deal_fn(seed))
    # `game.play_game` caps steps; this loop did not, so a livelock here hung
    # a training run silently for half an hour instead of raising. It cost more
    # to notice than to fix.
    steps = 0
    while obs is not None:
        steps += 1
        if steps > 5000:
            # **A livelock is a bug, but killing the run loses the bug.** This
            # used to raise, which took down a 400-iteration training run at
            # iteration 80 and threw away the only copy of the policy that
            # caused it -- the surviving checkpoint was two evals older and
            # did not reproduce. So: dump everything needed to reproduce, count
            # it, and let the run continue. `duel` already treats a -1 as a
            # game nobody won.
            _dump_livelock(env, table, seed, net_seat, steps)
            return -1
        if obs.to_move == net_seat:
            i = net_choice(net, obs, device, deterministic, generator)
        else:
            act = opponent(env.state, table, cfg, obs.to_move, env.legal)
            i = env.legal.index(act)
        obs = env.step(i).obs
    return int(env.state.winner)


def duel(net, opponent_factory, table, cfg, deal_fn, n: int = 200,
         device: str = "cpu", seed0: int = 100_000,
         deterministic: bool = False) -> float:
    """Network win rate over `n` deals, each played from both seats."""
    enc = Encoder(table, cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed0)
    wins = games = 0
    for i in range(n):
        for net_seat in (0, 1):
            opp = opponent_factory(np.random.default_rng(seed0 + i))
            w = play_one(net, opp, table, cfg, enc, seed0 + i, net_seat,
                         deal_fn, device, deterministic, gen)
            if w >= 0:
                wins += int(w == net_seat)
            games += 1
    return wins / games


def per_deck(net, table, cfg, pool, opponent_factory=None, n: int = 20,
             device: str = "cpu", seed0: int = 500_000) -> list[dict]:
    """Win rate broken down by the deck the LEARNER piloted.

    **`report()` cannot answer the question this does.** It returns one
    aggregate over every matchup, which cannot tell "decent at all 30 decks"
    from "strong at 27 and hopeless at 3" -- and that difference is exactly the
    specialist-versus-generalist decision. A generalist whose per-deck rates are
    flat needs no specialists; one with three bad decks needs three fine-tunes,
    not thirty agents.

    The learner pilots `d` from BOTH seats against a sampled opponent deck, for
    the reason `duel` does: at victory 3 the first-player edge is ~64%, so an
    unpaired per-deck number is dominated by which seat it happened to sit in.

    `n` is deals per deck, so the cost is `2 * n * len(pool)` games. The
    confidence interval is reported because with n=20 a 10-point gap between two
    decks is not yet a finding.
    """
    from rl.decks import matchup
    if opponent_factory is None:
        opponent_factory = greedy_agent
    enc = Encoder(table, cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed0)
    rows = []
    for d in pool:
        wins = games = 0
        for i in range(n):
            for net_seat in (0, 1):
                rng = np.random.default_rng(seed0 + i)
                foe = pool[int(rng.integers(len(pool)))]
                a, b = (d, foe) if net_seat == 0 else (foe, d)
                picks = (int(rng.integers(3)), int(rng.integers(3)))
                deal = (lambda _s, a=a, b=b, pk=picks:
                        matchup(a, b, picks=pk))
                w = play_one(net, opponent_factory(rng), table, cfg, enc,
                             seed0 + i, net_seat, deal, device, False, gen)
                if w >= 0:
                    wins += int(w == net_seat)
                games += 1
        wr = wins / max(1, games)
        ci = 1.96 * math.sqrt(max(wr * (1 - wr), 1e-9) / max(1, games))
        rows.append({"deck": d.name, "win_rate": wr, "ci": ci, "games": games})
    rows.sort(key=lambda r: r["win_rate"])
    return rows


def print_per_deck(rows: list[dict]) -> None:
    """The table, worst deck first -- that is the one that would need help."""
    if not rows:
        return
    lo, hi = rows[0], rows[-1]
    print(f"  {'deck':<34} {'win rate':>9}  {'+/-':>6}  games")
    print("  " + "-" * 62)
    for r in rows:
        print(f"  {r['deck']:<34} {r['win_rate']:>8.1%}  "
              f"{r['ci']:>5.1%}  {r['games']:>5d}")
    spread = hi["win_rate"] - lo["win_rate"]
    # Is the spread bigger than the noise? If not, per-deck differences are not
    # yet measurable and specialists cannot be justified from this data.
    noise = lo["ci"] + hi["ci"]
    verdict = ("REAL -- the worst decks are genuinely worse served"
               if spread > noise else
               "within noise -- no per-deck weakness is measurable yet")
    print(f"\n  spread {spread:.1%} (worst {lo['deck']} -> best {hi['deck']}), "
          f"noise +/-{noise:.1%}\n  -> {verdict}")


def per_deck_baseline(table, cfg, pool, n: int = 30, seed0: int = 700_000) -> list[dict]:
    """Deck strength with the POLICY TAKEN OUT: greedy pilots both sides.

    **This is what makes `per_deck` interpretable.** That function reports the
    learner's win rate per deck, which conflates two different things -- "the
    agent is bad with this deck" and "this deck is bad against the field". A
    4% deck could be either. Running the same measurement with a fixed heuristic
    on both sides isolates the second, so the difference is the first.

    No network, so this is cheap and -- unlike anything measured on a checkpoint
    -- it does not go stale when the policy or the observation changes. It is a
    property of the decks and the engine.
    """
    from rl.decks import matchup
    rows = []
    for d in pool:
        wins = games = 0
        for i in range(n):
            for seat in (0, 1):
                rng = np.random.default_rng(seed0 + i)
                foe = pool[int(rng.integers(len(pool)))]
                a, b = (d, foe) if seat == 0 else (foe, d)
                picks = (int(rng.integers(3)), int(rng.integers(3)))
                decks, runes, bfs, legends, champs = matchup(a, b, picks=picks)
                st = game.new_game(table, cfg, decks, runes, bfs, seed=seed0 + i,
                                   legends=legends, champions=champs)
                agents = [greedy_agent(np.random.default_rng(seed0 + i + 7 * k))
                          for k in range(2)]
                try:
                    out = game.play_game(table, cfg, st, agents)
                    w = int(out.get("winner", -1))
                except BoardOverflow:
                    # This helper drives `play_game` directly rather than
                    # through `RiftboundEnv`, so the guard that turns a
                    # board-row overflow into a truncation is not in the path --
                    # and greedy piloting BOTH seats reaches `MAX_PERMS` more
                    # often than either eval opponent does, because it plays
                    # every token-maker it can afford. Resolve it the way
                    # `env.final_rewards` does, by 408.2.b, so an overflow costs
                    # one game instead of the whole measurement.
                    r = A_.outcome(st, cfg)
                    w = 0 if r[0] > 0 else (1 if r[1] > 0 else -1)
                if w >= 0:
                    wins += int(w == seat)
                games += 1
        wr = wins / max(1, games)
        ci = 1.96 * math.sqrt(max(wr * (1 - wr), 1e-9) / max(1, games))
        rows.append({"deck": d.name, "win_rate": wr, "ci": ci, "games": games})
    rows.sort(key=lambda r: r["win_rate"])
    return rows


def report(net, table, cfg, deal_fn, n: int = 200, device: str = "cpu") -> dict:
    """Both Phase 4 exit numbers: >=90% vs random, >=65% vs greedy."""
    net.eval()
    vs_random = duel(net, game.random_agent, table, cfg, deal_fn, n, device)
    vs_greedy = duel(net, greedy_agent, table, cfg, deal_fn, n, device)
    net.train()
    return {
        "vs_random": vs_random,
        "vs_greedy": vs_greedy,
        "livelocks": len(LIVELOCKS),
        "pass": vs_random >= 0.90 and vs_greedy >= 0.65,
    }
