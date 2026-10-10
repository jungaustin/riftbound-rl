"""Baseline head-to-head -- PLAN.md Phase 2 exit criteria.

Two questions, both of which must be answered before any training happens:

  1. **Is the mirror fair?** random vs random, seat-swapped on paired seeds,
     should sit at 50% +/- 2. A skew here is a seat asymmetry bug, and it would
     be indistinguishable from "the agent learned something" later.
  2. **Does the engine reward sane play?** Greedy should beat random by >=80%.
     If it does not, either the engine is wrong or the heuristic is -- and this
     is far cheaper to discover here than after a training run.

Seat-swapping on paired seeds is what makes these numbers mean anything: both
seats see the identical deal, so any residual gap is structural rather than
variance in what was dealt.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rl.agents.greedy import greedy_agent
from rl.config import Config
from rl.engine import game
from rl.engine.cardtable import full_table
from rl.tests.fuzz import make_game, v0_pool


def duel(table, cfg, make_a, make_b, n: int, seed0: int = 0) -> float:
    """Win rate of A over B, playing each deal twice with seats swapped."""
    wins = games = 0
    for i in range(n):
        for swap in (0, 1):
            rng = np.random.default_rng(seed0 + i)
            s = make_game(table, cfg, seed0 + i)
            agents = [make_a(rng), make_b(rng)]
            if swap:
                agents = agents[::-1]
            r = game.play_game(table, cfg, s, agents)
            if r["winner"] >= 0:
                a_seat = 1 if swap else 0
                wins += (r["winner"] == a_seat)
            games += 1
    return wins / games


def main(n=500, victory=8):
    table = full_table()
    cfg = Config().at_victory_score(victory)
    print(f"baselines at victory_score={victory}, {n} deals x 2 seats each")

    mirror = duel(table, cfg, game.random_agent, game.random_agent, n)
    ok1 = abs(mirror - 0.5) <= 0.02
    print(f"  random vs random : {mirror:.1%}   "
          f"{'ok' if ok1 else 'FAIL -- expected 50% +/- 2, seat asymmetry?'}")

    g = duel(table, cfg, greedy_agent, game.random_agent, n)
    ok2 = g >= 0.80
    print(f"  greedy vs random : {g:.1%}   "
          f"{'ok' if ok2 else 'FAIL -- expected >=80%, engine or heuristic wrong'}")

    gm = duel(table, cfg, greedy_agent, greedy_agent, n)
    print(f"  greedy vs greedy : {gm:.1%}   (sanity: should also be ~50%)")
    return ok1 and ok2


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    v = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    sys.exit(0 if main(n, v) else 1)
