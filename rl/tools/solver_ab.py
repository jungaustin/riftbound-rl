"""Does the #28 solver convert the forced wins the policy throws away?

    PYTHONPATH="$PWD" python3 rl/tools/solver_ab.go.py [n_control]

**Why not a plain 300-game duel.** The effect being measured is ~3.8pp, and 300
games resolves ±5.6pp -- the measurement would be unable to see its own target.
`MEASURING.md` has the sizing table; detecting 3.8pp needs ~700 games, which is
hours.

**So pair it instead.** The solver can only change a game that contained a
thrown-away forced win, and `lethal_probe` already identified exactly those: 31 of
400. Replaying those 31 with and without the solver, on identical seeds, removes
the 369 games where nothing can change and which contribute only noise. The
fleet-wide effect then follows arithmetically: flips / 400.

**And a control set, because a solver that gains 15 and loses 10 elsewhere is
worthless.** The same paired comparison over games the probe found NO miss in. The
solver should be inert there -- it either proves a win or defers -- so any movement
is a bug in it, not a benefit.

Both conditions reseed the sampler per game, so the two runs make identical
choices until the solver actually overrides something. After that they diverge,
which is the effect.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.config import Config
from rl.engine.cardtable import full_table
from rl.engine.state import BoardOverflow
from rl.env import RiftboundEnv
from rl.eval import net_choice
from rl.obs import Encoder
from rl.ppo import HP, Trainer, deck_pool_deal, select_decks
from rl.solver import Solver

CKPT = "rl/runs/v11-pool/last.pt"
SEED0 = 900_000


def play(net, table, cfg, deal, enc, game: int, solver: Solver | None) -> int:
    """Self-play one game; returns the winner (-1 undecided). Seat 0 may solve."""
    gen = torch.Generator(device="cpu").manual_seed(SEED0 + game)
    env = RiftboundEnv(table, cfg, encoder=enc)
    obs = env.reset(SEED0 + game, *deal(SEED0 + game))
    try:
        while obs is not None:
            seat = int(obs.to_move)
            idx = net_choice(net, obs, "cpu", False, gen)
            if solver is not None and seat == 0:
                idx = solver.choice(env.state, table, cfg, seat, env.legal, idx)
            obs = env.step(idx).obs
        return int(env.state.winner)
    except BoardOverflow:
        return -1


def main() -> int:
    n_control = int(sys.argv[1]) if len(sys.argv) > 1 else 31
    table = full_table()
    cfg = dc_replace(Config().at_victory_score(8), units_only=False)
    pool = select_decks(table, None, 0.6)
    deal = deck_pool_deal(table, 0.6, seed_decks=pool)
    enc = Encoder(table, cfg)
    tr = Trainer(table, cfg, deal, HP(n_envs=2, rollout=16), device="cpu", seed=0)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    tr.net.load_state_dict(ck["net"])
    tr.net.eval()

    probe = json.load(open("rl/runs/lethal_probe.json"))["per_game"]
    missed = [r["game"] for r in probe if r["missed"]]
    clean = [r["game"] for r in probe if not r["missed"] and r["seen"] == 0]
    random.Random(0).shuffle(clean)
    clean = sorted(clean[:n_control])

    print(f"{CKPT} (iter {ck.get('iter')}), self-play, seat 0 may use the solver")
    print(f"  TREATMENT: {len(missed)} games the probe found a thrown-away win in")
    print(f"  CONTROL  : {len(clean)} games it found no forced win in at all\n")

    out = {}
    for label, games in (("treatment", missed), ("control", clean)):
        base_w = solv_w = flips_for = flips_against = 0
        sol = Solver()
        for k, g in enumerate(games, 1):
            b = play(tr.net, table, cfg, deal, enc, g, None)
            a = play(tr.net, table, cfg, deal, enc, g, sol)
            base_w += int(b == 0)
            solv_w += int(a == 0)
            if a == 0 and b != 0:
                flips_for += 1
            elif b == 0 and a != 0:
                flips_against += 1
            if k % 5 == 0 or k == len(games):
                print(f"    {label} {k}/{len(games)}: baseline {base_w} wins, "
                      f"solver {solv_w} wins  (+{flips_for} / -{flips_against})",
                      flush=True)
        out[label] = dict(n=len(games), base=base_w, solver=solv_w,
                          gained=flips_for, lost=flips_against,
                          solver_stats=sol.summary())
        print(f"  {label}: {sol.summary()}\n")

    t, c = out["treatment"], out["control"]
    print("=" * 72)
    print(f"TREATMENT ({t['n']} games with a thrown-away win)")
    print(f"  seat 0 wins: baseline {t['base']} -> with solver {t['solver']}")
    print(f"  games flipped TO a win: {t['gained']}   flipped AWAY: {t['lost']}")
    print(f"CONTROL ({c['n']} games with no forced win at all)")
    print(f"  seat 0 wins: baseline {c['base']} -> with solver {c['solver']}")
    print(f"  games flipped TO a win: {c['gained']}   flipped AWAY: {c['lost']}")
    print(f"    (the solver should be INERT here; movement means it is wrong)")
    net_gain = (t["gained"] - t["lost"]) + (c["gained"] - c["lost"])
    print("-" * 72)
    print(f"net games gained across {t['n'] + c['n']} played: {net_gain:+d}")
    print(f"scaled to the probe's 400-game fleet, the treatment arm is "
          f"{(t['gained'] - t['lost']) / 400:+.2%} of all games")
    json.dump(out, open("rl/runs/solver_ab.json", "w"), indent=1)
    print("detail in rl/runs/solver_ab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
