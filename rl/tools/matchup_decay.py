"""How much record can a 940-cell PFSP matrix actually hold? (BACKLOG #21)

    PYTHONPATH="$PWD" python3 rl/tools/matchup_decay.py [games]

**The question.** `OpponentPool.record` decays EVERY entry on EVERY game at
0.98. Over 20 entries that is fine -- an entry is revisited every ~20 games and
0.98**20 = 0.67, so each entry keeps a few games of memory. Lifting that rule
unchanged onto a (20 opponents x 47 decks) matrix is the obvious thing to do and
it is also the thing that would make matchup-aware PFSP a no-op that LOOKS
implemented: a cell is revisited every ~940 games, and 0.98**940 = 1e-8, so a
cell's record is erased ~19x over before it can be seen twice. Every cell would
sit at the 0.5 prior forever and the sampler would be uniform.

This measures both variants over a simulated run rather than arguing about it.
`mean games/cell` is the number that matters: it is the evidence a weight is
computed from, and at <<1 there is no evidence anywhere in the matrix.

Printed for the real shape of a v11/v12-sized run: 20 entries, 47 decks, and a
few thousand frozen-opponent episodes (v12 saw ~30k games total, half of them
against the pool).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from rl.pool import Entry, OpponentPool


def _pool(n_entries: int) -> OpponentPool:
    p = OpponentPool(matchup=True)
    for i in range(n_entries):
        p.entries.append(Entry(name=f"iter{i}", state_dict={}, iteration=i))
    return p


def run(games: int, n_entries: int = 20, n_decks: int = 47, seed: int = 0):
    """Play `games` uniformly over the grid under each decay rule."""
    keys = [f"deck{i}" for i in range(n_decks)]
    rng = np.random.default_rng(seed)
    draws = [(int(rng.integers(n_entries)), keys[int(rng.integers(n_decks))],
              bool(rng.random() < 0.45)) for _ in range(games)]

    per_cell = _pool(n_entries)
    for i, k, won in draws:
        per_cell.record(per_cell.entries[i], learner_won=won, deck_key=k)

    # The rejected variant: the same global sweep the per-opponent record uses.
    glob = _pool(n_entries)
    for i, k, won in draws:
        for e in glob.entries:
            for kk in e.cells:
                e.cells[kk][0] *= glob.decay
                e.cells[kk][1] *= glob.decay
        w, g = glob.entries[i].cells.get(k, (0.0, 0.0))
        glob.entries[i].cells[k] = [w + float(won), g + 1.0]

    out = {}
    for name, p in (("per-cell (shipped)", per_cell), ("global (rejected)", glob)):
        g = np.array([v[1] for e in p.entries for v in e.cells.values()])
        w = p.cell_weights(keys)
        out[name] = {
            "cells touched": int((g > 0).sum()),
            "mean games/cell": float(g.sum() / (n_entries * n_decks)),
            "max games in a cell": float(g.max()) if g.size else 0.0,
            # How far from uniform the sampler ends up. 1.0 = uniform = no-op.
            "max weight / uniform": float(w.max() * w.size),
            "cells above 2x uniform": int((w * w.size > 2.0).sum()),
        }
    return out


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
    print(f"20 opponents x 47 decks = 940 cells, {games} frozen-opponent games\n")
    res = run(games)
    cols = list(next(iter(res.values())).keys())
    print(f"{'':20}" + "".join(f"{c:>24}" for c in cols))
    for name, row in res.items():
        print(f"{name:20}" + "".join(f"{row[c]:>24.4g}" for c in cols))
    print("\nA 'max weight / uniform' of ~1.0 is a sampler that does nothing.")
