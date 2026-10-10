"""An Elo ladder across checkpoints -- the yardstick `vs_greedy` cannot be.

    PYTHONPATH="$PWD" python3 rl/ladder.py [--deals N] [--only a,b] [--fresh]

**Why this exists.** v10-exploiter beat its own predecessor 56.3% head-to-head
while `vs_greedy` did not move -- 70.5% before, 68.3% after, inside noise. The
policy improved and the metric could not see it, so "71% is this architecture's
ceiling" was a statement about the yardstick, not about the agent.

BACKLOG 15c says why, and it is structural rather than a tuning problem: greedy
attacks only when it can wipe every defender. It can never swing one unit, never
make a deliberate losing trade, and never kill a specific blocker to unlock a
later play. A policy can learn all three and beat greedy by exactly as much as
before.

So strength is measured against OTHER POLICIES here, and greedy and random stay
in only as fixed rungs that give the scale an absolute floor.

**Anchored on purpose.** Elo computed inside a moving population can rise while
every member gets weaker -- everyone beating everyone slightly better proves
nothing, which is the warning already written at the top of `rl/pool.py`. Greedy
is pinned at 0 and never trains, so a rating here means the same thing across
runs and months.

**Incremental, because a round-robin is expensive.** Results are cached per pair
in `rl/runs/ladder.json`, and only missing pairs are played. Add a checkpoint,
rerun, and it plays just that checkpoint's pairings. `--fresh` discards the cache.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch
from rl.nets import load_weights, shapes_growable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.agents.greedy import greedy_agent
from rl.config import Config
from rl.engine import game
from rl.engine.cardtable import full_table
from rl.engine.state import BoardOverflow
from rl.env import RiftboundEnv
from rl.eval import duel, net_choice
from rl.obs import Encoder
from rl.pool import elo_from_winrate
from rl.ppo import HP, Trainer, deck_pool_deal, select_decks

CACHE = Path("rl/runs/ladder.json")
# Greedy is the anchor: pinned at 0 so a rating is comparable across runs.
ANCHORS = ("greedy", "random")


# ---------------------------------------------------------------------------
# players
# ---------------------------------------------------------------------------

def discover() -> list[str]:
    """Every `best.pt` under rl/runs, newest run last.

    `last.pt` is deliberately left out: it is whatever the clock happened to
    stop on, so laddering it measures where a run was interrupted rather than
    what it achieved.
    """
    out = []
    for p in sorted(Path("rl/runs").glob("*/best.pt")):
        out.append(str(p))
    return out


def load_net(path: str, tr: Trainer, cfg) -> tuple:
    """Weights plus the config they were trained in.

    The config check is the lesson of v9-exploiter: it trained on the
    units-only game against a champion trained with spells, and `ck["shapes"]`
    could not see it because the observation layout is identical either way.
    A checkpoint from a different game is still laddered -- it is a real policy
    and its rating is real -- but it is LABELLED, because its number means
    "this is how a net trained for another game does at this one".
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert shapes_growable(ck["shapes"], tr.vec.shapes()), f"{path}: different obs layout"
    net = copy.deepcopy(tr.net)
    load_weights(net, ck, tr.vec.shapes())
    net.eval()
    note = ""
    old = ck.get("cfg")
    if old is None:
        note = "cfg unknown"
    elif old.get("units_only") != cfg.units_only:
        note = f"TRAINED units_only={old.get('units_only')}"
    return net, ck, note


# ---------------------------------------------------------------------------
# one pairing
# ---------------------------------------------------------------------------

def net_vs_net(a, b, table, cfg, deal, enc, n: int, seed0: int) -> tuple[int, int, int]:
    """`a`'s wins, `b`'s wins, undecided -- each deal played from both seats.

    Seats are swapped on every deal because the first-player edge would
    otherwise decide it, and the control for this harness is a net against
    itself, which must come out at 50%.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed0)
    wa = wb = dr = 0
    for i in range(n):
        for a_seat in (0, 1):
            env = RiftboundEnv(table, cfg, encoder=enc)
            obs = env.reset(seed0 + i, *deal(seed0 + i))
            try:
                steps = 0
                while obs is not None:
                    steps += 1
                    if steps > 5000:
                        break
                    net = a if obs.to_move == a_seat else b
                    obs = env.step(net_choice(net, obs, "cpu", False, gen)).obs
                w = int(env.state.winner)
            except BoardOverflow:
                w = -1
            if w < 0:
                dr += 1
            elif w == a_seat:
                wa += 1
            else:
                wb += 1
    return wa, wb, dr


def agent_vs_agent(fa, fb, table, cfg, deal, n: int, seed0: int) -> tuple[int, int, int]:
    """Two heuristics, for the greedy-vs-random rung. No network involved."""
    wa = wb = dr = 0
    for i in range(n):
        for a_seat in (0, 1):
            # `deal` yields 5 items; new_game takes legends and champions by
            # keyword, so unpack rather than splatting -- splatting lands
            # `legends` on the `seed` parameter.
            decks, runes, bfs, legends, champs = deal(seed0 + i)
            st = game.new_game(table, cfg, decks, runes, bfs, seed=seed0 + i,
                               legends=legends, champions=champs)
            agents = [None, None]
            agents[a_seat] = fa(np.random.default_rng(seed0 + i))
            agents[1 - a_seat] = fb(np.random.default_rng(seed0 + i + 7))
            try:
                w = int(game.play_game(table, cfg, st, agents).get("winner", -1))
            except BoardOverflow:
                w = -1
            if w < 0:
                dr += 1
            elif w == a_seat:
                wa += 1
            else:
                wb += 1
    return wa, wb, dr


# ---------------------------------------------------------------------------
# rating fit
# ---------------------------------------------------------------------------

def fit_elo(results: dict, players: list[str], anchor: str = "greedy",
            iters: int = 2000) -> dict:
    """Bradley-Terry ratings by MM, with `anchor` pinned at 0.

    Minorization-maximization rather than gradient ascent: the BT likelihood's
    MM update is monotone and needs no step size. A gradient step here is a trap
    -- the residual is in units of GAMES, so any learning rate large enough to
    move a rating meaningfully also diverges.

    Fitting all pairings at once rather than applying `elo_update` game by game,
    because sequential Elo depends on the order results arrive, which for a set
    of already-completed matches is an artefact and nothing else.
    """
    pairs = []
    for key, (wa, wb, _) in results.items():
        a, b = key.split("|")
        if a in players and b in players and wa + wb > 0:
            pairs.append((a, b, float(wa), float(wb)))
    if not pairs:
        return {p: 0.0 for p in players}

    # Strengths on the multiplicative BT scale; Elo is 400*log10 of these.
    p_ = {p: 1.0 for p in players}
    wins = {p: 0.0 for p in players}
    games: dict[tuple[str, str], float] = {}
    for a, b, wa, wb in pairs:
        wins[a] += wa
        wins[b] += wb
        games[(a, b)] = games.get((a, b), 0.0) + wa + wb
    # A player who won everything or nothing has no finite rating. Add a tiny
    # symmetric prior against both anchors so the fit stays defined rather than
    # running off to infinity on a 100% result.
    for q in players:
        wins[q] += 0.5
        games[(q, anchor)] = games.get((q, anchor), 0.0) + (0.0 if q == anchor else 1.0)

    for _ in range(iters):
        new_p = {}
        for q in players:
            denom = 0.0
            for (a, b), n in games.items():
                if a == q:
                    denom += n / (p_[q] + p_[b])
                elif b == q:
                    denom += n / (p_[a] + p_[q])
            new_p[q] = wins[q] / denom if denom > 0 else p_[q]
        # Renormalize to stop the whole vector drifting; the anchor is applied
        # after converting to Elo, where it is a simple subtraction.
        scale = sum(new_p.values()) / len(new_p)
        p_ = {q: v / scale for q, v in new_p.items()}

    elo = {q: 400.0 * math.log10(max(v, 1e-12)) for q, v in p_.items()}
    base = elo[anchor]
    return {q: v - base for q, v in elo.items()}


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=40,
                    help="deals per pairing; each is played from BOTH seats, so "
                         "games per pairing is twice this (default 40 -> 80)")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; ladder just these players")
    ap.add_argument("--fresh", action="store_true", help="discard the cache")
    ap.add_argument("--seed0", type=int, default=900_000,
                    help="deal seeds. Deliberately far from any run's training "
                         "seeds, which start at 0")
    ap.add_argument("--victory", type=int, default=8)
    ap.add_argument("--snapshots", default=None, metavar="RUN",
                    help="also ladder this run's pool snapshots "
                         "(rl/runs/RUN/snapshots/*.pt), which is how you see "
                         "WITHIN a run whether it was still improving. Implies "
                         "--anchors-only: a round-robin over 50 snapshots is "
                         "1,225 pairings, which is not a measurement anyone will "
                         "wait for.")
    ap.add_argument("--every", type=int, default=1, metavar="N",
                    help="with --snapshots, ladder only every Nth snapshot. A "
                         "curve needs shape, not every point: 9 rungs across a "
                         "run show where it flattened just as well as 27 and cost "
                         "a third of the games. The newest is always kept.")
    ap.add_argument("--anchors-only", action="store_true",
                    help="play each player against the anchors (and --ref) "
                         "instead of a full round-robin: O(n) pairings, not "
                         "O(n^2). The ratings are then on a common scale without "
                         "every pair having met.")
    ap.add_argument("--ref", default=None, metavar="NAME",
                    help="an extra fixed reference every player is measured "
                         "against under --anchors-only, e.g. v8-postfix. Worth "
                         "setting: greedy is a weak anchor and a strong rung "
                         "pins the top of the scale much better.")
    a = ap.parse_args(argv)

    table = full_table()
    cfg = dc_replace(Config().at_victory_score(a.victory), units_only=False,
                     choose_payment=True)
    pool = select_decks(table, None, 0.6)
    deal = deck_pool_deal(table, 0.6, seed_decks=pool)
    enc = Encoder(table, cfg)
    tr = Trainer(table, cfg, deal, HP(n_envs=2, rollout=16), device="cpu", seed=0)

    ckpts = discover()
    if a.snapshots:
        snaps = sorted(Path("rl/runs").glob(f"{a.snapshots}/snapshots/*.pt"))
        if not snaps:
            print(f"  no snapshots under rl/runs/{a.snapshots}/snapshots/ -- "
                  f"runs before 2026-10-01 did not keep them")
        if a.every > 1 and snaps:
            # Keep the newest regardless of stride -- where a run ENDED is the
            # one rung you always want.
            keep = snaps[::a.every]
            if snaps[-1] not in keep:
                keep.append(snaps[-1])
            snaps = keep
        ckpts += [str(x) for x in snaps]
        a.anchors_only = True
    if a.only:
        wants = [w.strip().lower() for w in a.only.split(",")]
        ckpts = [c for c in ckpts if any(w in c.lower() for w in wants)]
    nets, notes = {}, {}
    for path in ckpts:
        pp = Path(path)
        name = (f"{pp.parent.parent.name}@{pp.stem.replace('iter', '')}"
                if pp.parent.name == "snapshots" else pp.parent.name)
        try:
            nets[name], ck, note = load_net(path, tr, cfg)
            notes[name] = note or f"iter {ck.get('iter')}"
        except AssertionError as e:
            print(f"  skipped {name}: {e}")
    players = list(nets) + list(ANCHORS)
    print(f"ladder: {len(nets)} checkpoints + {len(ANCHORS)} anchors, "
          f"{a.deals * 2} games per pairing, deal seeds from {a.seed0:,}")
    for n, note in notes.items():
        print(f"  {n:<20} {note}")

    results = {}
    if CACHE.exists() and not a.fresh:
        results = json.load(open(CACHE))
        print(f"  cache: {len(results)} pairings already played")

    factories = {"greedy": greedy_agent, "random": game.random_agent}
    if a.anchors_only:
        fixed = list(ANCHORS) + ([a.ref] if a.ref else [])
        cand = [(x, y) for x in players for y in fixed
                if x != y and not (x in ANCHORS and y in ANCHORS)]
        # Plus the anchor pair itself, so the scale has a known interior rung.
        cand += [(ANCHORS[0], ANCHORS[1])]
        seen, pairs = set(), []
        for x, y in cand:
            k = tuple(sorted((x, y)))
            if k not in seen:
                seen.add(k)
                pairs.append((x, y) if f"{y}|{x}" not in results else (y, x))
    else:
        pairs = list(itertools.combinations(players, 2))
    todo = [p for p in pairs
            if f"{p[0]}|{p[1]}" not in results and f"{p[1]}|{p[0]}" not in results]
    print(f"  {len(todo)} pairings to play\n")

    for k, (x, y) in enumerate(todo, 1):
        key = f"{x}|{y}"
        if x in factories and y in factories:
            res = agent_vs_agent(factories[x], factories[y], table, cfg, deal,
                                 a.deals, a.seed0)
        elif y in factories:
            wr = duel(nets[x], factories[y], table, cfg, deal, a.deals, "cpu",
                      seed0=a.seed0)
            # `duel` returns a rate over decided games; recover counts for the fit.
            g = a.deals * 2
            res = (int(round(wr * g)), g - int(round(wr * g)), 0)
        elif x in factories:
            wr = duel(nets[y], factories[x], table, cfg, deal, a.deals, "cpu",
                      seed0=a.seed0)
            g = a.deals * 2
            res = (g - int(round(wr * g)), int(round(wr * g)), 0)
        else:
            res = net_vs_net(nets[x], nets[y], table, cfg, deal, enc,
                             a.deals, a.seed0)
        results[key] = res
        dec = res[0] + res[1]
        pct = res[0] / max(1, dec)
        ci = 1.96 * math.sqrt(pct * (1 - pct) / max(1, dec))
        print(f"  [{k}/{len(todo)}] {x} vs {y}: {pct:.1%} +/-{ci:.1%}"
              f"{f'  ({res[2]} undecided)' if res[2] else ''}", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(CACHE, "w"), indent=1)

    r = fit_elo(results, players)
    print(f"\n{'player':<22}{'Elo':>7}  {'(greedy = 0)':<14}")
    for p, v in sorted(r.items(), key=lambda t: -t[1]):
        tag = "  <-- anchor" if p in ANCHORS else ""
        print(f"  {p:<20}{v:>+7.0f}{tag}")
    print(f"\ncached in {CACHE}; rerun after a new run to extend it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
