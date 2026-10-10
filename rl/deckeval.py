"""Deck evaluation -- the §1.1/§1.2 deliverable.

    python3 rl/deckeval.py --ckpt rl/runs/fp/best.pt --a decks/... --b decks/...
    python3 rl/deckeval.py --ckpt ... --round-robin --top 6

Two numbers, and they answer different questions:

**Head-to-head win rate.** Play the matchup many times with the same policy on
both sides, seats swapped on paired seeds. This is the number that means "deck A
beats deck B", and pairing matters more here than anywhere else in the project:
going first is worth ~14 points at victory 3 (measured), so an unpaired result
is mostly a coin-flip about who was dealt seat 0.

**The critic's V(s0).** With `gamma = 1.0` and terminal-only reward the value
head is a literal win-probability predictor, so its opinion of the opening
position *is* a deck rating -- available without playing a single game. That is
the deliverable the whole architecture was pointed at (§1.2).

Reporting both is deliberate: V(s0) is fast and comparable across many decks,
the win rate is slow and true, and **the gap between them is the interesting
diagnostic**. A critic that rates a deck highly and then loses with it has
learned something about positions that does not survive contact with play.

**Every result carries its deck coverage.** At 50% coverage the engine is
playing a proxy built by substituting unimplemented cards, so the number
describes the proxy. `DeckLoad.coverage` travels with the result for exactly
that reason, and `--min-coverage` refuses decks below a threshold.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from rl.config import Config
from rl.decks import DeckLoad, load_all, load_deck, matchup
from rl.engine import actions as A
from rl.engine import game
from rl.engine.cardtable import full_table
from rl.env import RiftboundEnv
from rl.eval import net_choice
from rl.nets import RiftboundNet, to_torch, load_weights, shapes_growable
from rl.obs import Encoder
from rl.vec import batch


def ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


@torch.no_grad()
def opening_value(net, enc, table, cfg, a: DeckLoad, b: DeckLoad,
                  n: int = 40) -> float:
    """Mean V(s0) for deck A, averaged over deals and over both seats.

    Averaging over seats is not optional: V is "good for the side to move", and
    at victory 3 the side to move first is worth ~14 points of win rate. A
    single-seat average would mostly measure the seat.
    """
    vals = []
    for i in range(n):
        for swap in (0, 1):
            decks, runes, bfs, legs, champs = (matchup(b, a) if swap
                                              else matchup(a, b))
            env = RiftboundEnv(table, cfg, encoder=enc)
            obs = env.reset(10_000 + i, decks, runes, bfs, legs, champs)
            if obs is None:
                continue
            t = to_torch(batch([obs]), "cpu")
            _, v = net(t)
            v = float(v.item())
            a_seat = 1 if swap else 0
            vals.append(v if obs.to_move == a_seat else -v)
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def head_to_head(net, enc, table, cfg, a: DeckLoad, b: DeckLoad,
                 n: int = 60, deterministic: bool = False) -> dict:
    """Deck A's win rate, both decks driven by the same policy, seats swapped."""
    gen = torch.Generator().manual_seed(0)
    wins = games = 0
    for i in range(n):
        for swap in (0, 1):
            decks, runes, bfs, legs, champs = (matchup(b, a) if swap
                                              else matchup(a, b))
            env = RiftboundEnv(table, cfg, encoder=enc)
            obs = env.reset(20_000 + i, decks, runes, bfs, legs, champs)
            while obs is not None:
                obs = env.step(net_choice(net, obs, "cpu", deterministic,
                                          gen)).obs
            w = int(env.state.winner)
            a_seat = 1 if swap else 0
            if w >= 0:
                wins += int(w == a_seat)
            games += 1
    p = wins / max(1, games)
    return {"winrate": p, "ci95": ci95(p, games), "games": games}


def _swapped(d: DeckLoad) -> float:
    """Fraction of the deck replaced by a *different* card."""
    return sum(d.substituted.values()) / max(1, len(d.main))


def evaluate(net, enc, table, cfg, a: DeckLoad, b: DeckLoad, n: int) -> dict:
    h = head_to_head(net, enc, table, cfg, a, b, n)
    return {"a": a.name, "b": b.name,
            "coverage_a": a.coverage, "coverage_b": b.coverage,
            "bf_a": a.bf_coverage, "bf_b": b.bf_coverage,
            "swapped_a": _swapped(a), "swapped_b": _swapped(b),
            "v0": opening_value(net, enc, table, cfg, a, b),
            **h}


def _fmt(r: dict) -> str:
    """Two fidelity numbers, because they degrade the result differently.

    `cov` is how much of the deck is played as printed. `swap` is how much was
    replaced by a *different card* -- that one changes the deck's curve and
    domains, so it is the flag worth raising. A 80%-coverage threshold was
    used here once; no real deck reaches it, so the warning fired on every row
    and stopped carrying information.
    """
    warn = "  <-- SHAPE CHANGED" if max(r["swapped_a"], r["swapped_b"]) > 0.25 \
        else ("  <-- proxy" if min(r["coverage_a"], r["coverage_b"]) < 0.5
              else "")
    # `bf` is reported next to `cov` because a deck can read 100% covered while
    # all three of its battlefields do nothing -- and two decks differing only
    # in battlefields are then INDISTINGUISHABLE here, which is a property of
    # the engine rather than of the decks. Printing it stops the win rate being
    # quoted as if it settled a battlefield choice.
    if max(r["bf_a"], r["bf_b"]) < 1.0 and not warn:
        warn = "  <-- battlefields inert"
    return (f"  {r['a']:<34} vs {r['b']:<30} "
            f"{r['winrate']:>6.1%} +/-{r['ci95']:.1%}  "
            f"V(s0) {r['v0']:+.3f}   "
            f"cov {r['coverage_a']:.0%}/{r['coverage_b']:.0%} "
            f"bf {r['bf_a']:.0%}/{r['bf_b']:.0%} "
            f"swap {r['swapped_a']:.0%}/{r['swapped_b']:.0%}{warn}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="rl/runs/fp/best.pt")
    p.add_argument("--a"), p.add_argument("--b")
    p.add_argument("--round-robin", action="store_true")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--victory", type=int, default=3)
    p.add_argument("--min-coverage", type=float, default=0.0)
    args = p.parse_args(argv)

    from dataclasses import replace
    table = full_table()
    cfg = replace(Config().at_victory_score(args.victory), units_only=False,
                  choose_payment=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    enc = Encoder(table, cfg)
    if not shapes_growable(ck["shapes"], enc.shapes()):
        raise SystemExit("checkpoint predates the current encoder; retrain.")
    net = RiftboundNet(enc.shapes())
    load_weights(net, ck, enc.shapes())
    net.eval()

    if args.round_robin:
        decks = [d for d in load_all(table) if d.coverage >= args.min_coverage]
        decks.sort(key=lambda d: -d.coverage)
        decks = decks[:args.top]
        print(f"round robin over {len(decks)} decks "
              f"(coverage >= {args.min_coverage:.0%}), {args.games} deals x 2 seats\n")
        score: dict[str, list] = {d.name: [] for d in decks}
        for i, x in enumerate(decks):
            for y in decks[i + 1:]:
                r = evaluate(net, enc, table, cfg, x, y, args.games)
                print(_fmt(r))
                score[x.name].append(r["winrate"])
                score[y.name].append(1 - r["winrate"])
        print("\n  standings (mean win rate across opponents):")
        for name, ws in sorted(score.items(), key=lambda t: -np.mean(t[1] or [0])):
            print(f"    {name:<40}{np.mean(ws or [0]):.1%}")
        return 0

    if not (args.a and args.b):
        raise SystemExit("give --a and --b, or --round-robin")
    a = load_deck(Path(args.a), table)
    b = load_deck(Path(args.b), table)
    print(f"  {a.report()}\n  {b.report()}\n")
    print(_fmt(evaluate(net, enc, table, cfg, a, b, args.games)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
