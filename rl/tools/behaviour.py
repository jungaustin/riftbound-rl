"""What does the policy DO? -- BACKLOG 15b, behavioural metrics.

    PYTHONPATH="$PWD" python3 rl/tools/behaviour.py [--games N] [--opp self|greedy|random|PATH] AGENT...

    # the baseline table
    PYTHONPATH="$PWD" python3 rl/tools/behaviour.py --games 60 \
        random greedy rl/runs/v11-pool/last.pt rl/runs/v12-entropy/last.pt

**Why this exists.** Win rate cannot say *what* was learned. The ladder can say a
policy got stronger; it cannot say what changed, and after three measured nulls
(more self-play, more entropy, search at inference) that is the expensive blind
spot: if the next run moves nothing we again will not know why. `MEASURING.md`
lists this as the top missing instrument.

Shape copied from `rl/tools/lethal_probe.py`: drive real games through
`RiftboundEnv`, interrogate every decision of the agent's own seats, cache
per-game rows to JSON every 10 games, print progress with an ETA.

**Every agent is driven on the same footing** -- same deck pool, same deals, same
seeds, same seat structure -- so the numbers are comparable. Comparing a policy
only against itself says nothing, which is why `random` and `greedy` are in the
table and why the default opponent is the agent itself (self-play), making the
measurement a property of the agent and not of whoever it was matched against.

===========================================================================
THE DEFINITIONS. Each one is exact, because a vague definition is how this
project has previously fooled itself.
===========================================================================

**ply / turn.** `state.ply` is one player's turn (see memory: "turn is a round,
ply is a turn"). Everything per-turn here is per *ply*, of the agent's own seat.

**committed combat.** The agent chose `A_COMMIT` -- it finished a Move
declaration -- into a battlefield where the *opponent had at least one live Unit
present at the moment of the commit*. By 461 that stages a Combat which the next
Cleanup initiates, so this is a combat the agent chose to start. A commit into
an empty or friendly-only battlefield is a Move, not a combat, and is counted
separately (`commits_no_combat`).

    Cross-checked against the engine: `A.apply`'s own log carries `combat_at`
    whenever a Cleanup initiates one. Both counters are reported. They can
    legitimately differ -- an [Ambush] unit played onto an enemy battlefield
    stages a combat with no commit at all -- so a non-commit-initiated count is
    reported rather than silently folded in.

**combat-turn.** A ply of the agent's in which it committed >= 1 combat.
**the headline:** the fraction of those plies in which it committed >= 2. The v5
baseline for exactly this number was random 12%, greedy 9%, v5 policy 10% -- i.e.
multi-combat sequencing was absent. Never re-measured since. `multi_bf` is the
stricter variant: 2+ committed combats at *distinct* battlefields.

**a combat it loses on raw Might.** Of its committed combats, the ones where the
summed `might_for_pool` of its attacking force (the declared units plus any of
its units already there) is strictly less than the summed `might_for_pool` of the
defenders present. Reported beside `no_wipe`: committed combats where the
attacking pool does not kill *every* defender by the engine's own `solve_kills`.
`no_wipe` is the rules-exact notion of a bad attack -- attackers are Recalled if
any defender survives (466.1.a.2), so a non-wiping attack achieves nothing --
while `might_deficit` is the cruder "I am outgunned" reading. Greedy attacks only
when it wipes, so **greedy's `no_wipe` must come back ~0%**; that is this
metric's validation against a case whose answer is known in advance.

**declined a scoring line.** At a decision with more than one legal action, each
legal action is applied on a clone and kept if it raises `state.points[seat]`
immediately. If any does, the decision is an *opportunity*; if the action the
agent actually chose does not, it is re-checked: if the same seat can still
score immediately from the position its own action reaches, it merely *deferred*
(the commonest case by far is ADD-then-COMMIT with a Move half-built) and the
decision leaves the denominator; only if the score is gone does it count as
*declined*. Deliberately one ply deep, so it is
a LOWER bound: points also arrive by Conquer, several plies after the Move that
earns them, and no such line is counted here. The opportunity count is printed
whether or not anything was declined -- a zero needs its denominator ("a zero is
a sample too").

**ended a turn with a Move available.** The agent chose `A_END_TURN` while an
`A_DECLARE` was legal: it had somewhere to move and went home instead. Not a
mistake in itself (holding back is real), which is why it is reported as a rate
and not as an error.

**Austin's complaints, measured** (`docs/REVIEW_HUMAN_GAMES.md`). Each is read
off the pending item's own script (`chain.item_spec`), never off card names:

  * **buff on an enemy** -- the agent fills a unit target slot whose ops give
    that target positive Might (`OP_MODIFY_MIGHT n>0`) with an ENEMY unit.
    Denominator: every such slot it fills.
  * **idle buff** -- the same, on its OWN unit, which then takes part in no
    combat for the rest of that ply (a showdown at its battlefield, or a commit
    that stages one). Pre-buffing a unit that then attacks is not idle.
    Counted only for a SPELL the agent chose to cast, and only when the buff
    serves nothing else in that spell: Rampage's +2 feeds its own damage
    exchange (`OP_FIGHT`), and Rengar - Pridestalker's +1 is an automatic
    trigger where only the target was chosen. Both were ~half the hits before
    this filter (sampled from v17, 2026-10-08).
  * **pointless replay** -- an `OP_BLINK` (banish, then play again) on its own
    unit that is not in a combat, has no "when you play me" trigger and no
    damage to reset: nothing changes but the card spent.
  * **left units home in a must-win attack** -- a commit that stages a combat
    while the opponent is within 2 of winning, with `A_ADD` options still
    unused at that moment.
  * **left a hold uncontested** -- an own ply in which the opponent controls a
    battlefield the agent could declare a move to, and it commits to none of
    them. Like `end_with_move`, a rate, not an error: sometimes the hold is
    too strong to attack (Austin's game 2).

Also reported, because they are free and they fingerprint a policy: the action-kind
histogram, decisions per turn (~16-20 with spells, ~5 without -- the cheap config
smell test from MEASURING.md), turns per game and the game outcomes.

**Uncertainty is cluster-bootstrapped over GAMES, not over plies.** Plies inside
one game are not independent, and this project has already been burned by a rare
event that clustered 61-to-339 across games. The interval printed resamples whole
games with replacement, which is the honest denominator.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from collections import Counter
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import torch
from rl.nets import load_weights, shapes_growable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.agents.greedy import greedy_agent
from rl.config import Config
from rl.engine import actions as A
from rl.engine import combat
from rl.engine.cardtable import full_table
from rl.engine import chain
from rl.engine.effects import (OP_BLINK, OP_MODIFY_MIGHT, TK_UNIT, TR_PLAY_ME,
                               CardSpec)
from rl.engine.state import (P_CTRL, P_DMG, P_LOC, BoardOverflow, bf_index,
                             bf_loc, is_battlefield)
from rl.env import RiftboundEnv
from rl.eval import net_choice
from rl.obs import Encoder
from rl.ppo import HP, Trainer, deck_pool_deal, select_decks
from rl.viewer import describe

SEED0 = 600_000
# Above this many legal actions the scoring scan is skipped rather than paying
# one clone+apply per option. Counted and reported, because a silently skipped
# decision is an unmeasured one.
SCAN_CAP = 80


# ---------------------------------------------------------------------------
# players, all behind one interface: chooser(env, obs) -> index
# ---------------------------------------------------------------------------

def make_chooser(spec: str, tr: Trainer, table, cfg, seed: int):
    """`random`, `greedy`, or a checkpoint path."""
    if spec == "random":
        rng = np.random.default_rng(seed)
        return lambda env, obs: int(rng.integers(len(env.legal)))
    if spec == "greedy":
        agent = greedy_agent(np.random.default_rng(seed))
        def choose(env, obs):
            act = agent(env.state, table, cfg, int(obs.to_move), env.legal)
            return env.legal.index(act)
        return choose
    ck = torch.load(spec, map_location="cpu", weights_only=False)
    assert shapes_growable(ck["shapes"], tr.vec.shapes()), f"{spec}: different obs layout"
    net = copy.deepcopy(tr.net)
    load_weights(net, ck, tr.vec.shapes())
    net.eval()
    old = ck.get("cfg")
    if old is not None and old.get("units_only") != cfg.units_only:
        print(f"  !! {spec} was trained with units_only={old.get('units_only')}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return lambda env, obs: net_choice(net, obs, "cpu", False, gen)


def label(spec: str) -> str:
    if spec in ("random", "greedy"):
        return spec
    p = Path(spec)
    return f"{p.parent.name}/{p.stem}"


# ---------------------------------------------------------------------------
# one agent, n games
# ---------------------------------------------------------------------------

def scoring_actions(env, seat: int) -> set[int] | None:
    """Indices of legal actions that immediately raise `points[seat]`.

    None when the scan was skipped (too many options). `GameState.clone` is safe
    for this -- it shares `rng`, which nothing during play consumes (#28a).
    """
    legal = env.legal
    if len(legal) > SCAN_CAP:
        return None
    st = env.state
    before = int(st.points[seat])
    out = set()
    for i, act in enumerate(legal):
        nxt = st.clone()
        try:
            A.apply(nxt, env.table, env.cfg, act)
        except Exception:
            continue
        if int(nxt.points[seat]) > before:
            out.add(i)
    return out


def defers_score(env, seat: int, act, before: int,
                 depth: int = 3, nodes: int = 500) -> bool:
    """After playing `act`, can this seat still reach the point anyway?

    **This exists because the naive one-ply decline count is mostly wrong**, and
    the error was found by printing the examples rather than trusting the
    number. Two false-positive classes, both measured:

      * a half-built Move -- with a declaration open `COMMIT` scores and `ADD`
        does not, so adding a second unit reads as a decline when the point is
        still one action away;
      * a Chain suspension -- committing two units queues their move triggers,
        so the Conquer point lands after the Chain resolves instead of inside
        the apply, and a point that is two applies away reads as gone.

    So a candidate decline is re-checked from the position the chosen action
    reaches: search the seat's OWN actions to `depth`, driving any opponent
    window with PASS (the `lethal_probe --assume-pass` device), and ask whether
    `points[seat]` ever exceeds `before`. If it does, the seat deferred rather
    than declined. Assuming the opponent passes is deliberately generous here --
    it can only *remove* declines, which keeps the reported number a floor.
    """
    table, cfg = env.table, env.cfg
    budget = [nodes]

    def drive(st) -> bool:
        """Hand the opponent's windows a PASS. False if the seat never acts."""
        hops = 0
        while (not A.is_terminal(st) and A.acting_seat(st) != seat
               and hops < 24):
            try:
                acts = A.legal_actions(st, table, cfg, A.acting_seat(st))
                pick = next((x for x in acts if x.kind == A.A_PASS),
                            acts[0] if acts else None)
                if pick is None:
                    return False
                A.apply(st, table, cfg, pick)
            except Exception:                        # noqa: BLE001
                return False
            hops += 1
        return not A.is_terminal(st) and A.acting_seat(st) == seat

    def rec(st, d: int) -> bool:
        if int(st.points[seat]) > before:
            return True
        if d <= 0 or budget[0] <= 0:
            return False
        if not drive(st):
            return int(st.points[seat]) > before
        try:
            legal = A.legal_actions(st, table, cfg, seat)
        except Exception:                            # noqa: BLE001
            return False
        if len(legal) > SCAN_CAP:
            return False
        for a in legal:
            if budget[0] <= 0:
                return False
            budget[0] -= 1
            nxt = st.clone()
            try:
                A.apply(nxt, table, cfg, a)
            except Exception:                        # noqa: BLE001
                continue
            if rec(nxt, d - 1):
                return True
        return False

    st = env.state.clone()
    try:
        A.apply(st, table, cfg, act)
    except Exception:                                # noqa: BLE001
        return False
    return rec(st, depth)


def _complaints(env, st, seat: int, ply: int, act, legal, row, buffed,
                hold_ply) -> None:
    """Tally Austin's complaint metrics for one of the agent's decisions."""
    table = env.table
    # Holds the agent could contest this ply, and where it actually went.
    if seat == int(st.active):
        h = hold_ply.setdefault((seat, ply), {"held": set(), "went": set()})
        for a in legal:
            if a.kind == A.A_DECLARE and is_battlefield(int(a.arg)):
                i = bf_index(int(a.arg))
                if int(st.bf_ctrl[i]) == 1 - seat:
                    h["held"].add(int(a.arg))
        if act.kind == A.A_COMMIT:
            h["went"].add(int(st.decl_dst))
    # A must-win attack that leaves units behind.
    if act.kind == A.A_COMMIT and is_battlefield(int(st.decl_dst)) \
            and st.units_at(int(st.decl_dst), 1 - seat).size:
        win_at = int(st.victory_score) + int(st.victory_bonus)
        if int(st.points[1 - seat]) >= win_at - 2:
            row["mustwin_attacks"] += 1
            if any(a.kind == A.A_ADD for a in legal):
                row["mustwin_left_home"] += 1
    # Target slots: buffs and replays, read off the pending item's script.
    if act.kind != A.A_TARGET or chain.open_slot_kind(st, table) != TK_UNIT:
        return
    tgt = int(act.arg)
    if not (0 <= tgt < int(st.n_perms)):
        return
    item = chain.oldest_pending(st)
    if item < 0 or int(st.pend_slot) < 0:
        return
    spec = chain.item_spec(st, table, item)
    slot = int(st.pend_slot)
    ops = [o for o in getattr(spec, "ops", ()) if getattr(o, "target", -1) == slot]
    mine = int(st.perms[tgt, P_CTRL]) == seat
    if any(o.op == OP_MODIFY_MIGHT and int(o.n) > 0 for o in ops):
        row["buffs"] += 1
        if not mine:
            row["buff_enemy"] += 1
        else:
            is_spell = isinstance(spec, CardSpec)
            serves = any(o.op != OP_MODIFY_MIGHT and (
                getattr(o, "target", -1) == slot
                or getattr(o, "target_b", -1) == slot)
                for o in getattr(spec, "ops", ()))
            if is_spell and not serves:
                row["buffs_own"] += 1
                buffed.append((tgt, ply))
    if any(o.op == OP_BLINK for o in ops) and mine:
        row["replays_own"] += 1
        sb = int(st.showdown_bf)
        in_fight = sb >= 0 and int(st.perms[tgt, P_LOC]) == bf_loc(sb)
        if not in_fight and int(st.perms[tgt, P_DMG]) == 0 \
                and not chain.row_has_trigger(st, table, tgt, TR_PLAY_ME):
            row["replay_pointless"] += 1


def run_agent(spec: str, opp: str, n_games: int, table, cfg, deal, enc, tr,
              out_path: Path, cache: dict, examples_wanted: int = 4) -> dict:
    """Drive `n_games` and return the tallies, per-game rows included."""
    name = label(spec)
    per_game: list[dict] = []
    kinds: Counter = Counter()
    tot = Counter()
    examples: list[str] = []
    t0 = time.time()
    for g in range(n_games):
        seed = SEED0 + g
        # Both seats get a fresh chooser per game, seeded off the game, so two
        # agents measured on the same game see the same deal and the same
        # stream position. A net's own sampling stream is per-chooser.
        me = make_chooser(spec, tr, table, cfg, seed)
        other = make_chooser(spec if opp == "self" else opp, tr, table, cfg,
                             seed + 1)
        if opp == "self":
            tracked, choosers = (0, 1), [me, other]
        else:
            # Seat alternates by game so the first-player edge cannot shape the
            # behaviour being reported.
            s0 = g % 2
            tracked, choosers = (s0,), [None, None]
            choosers[s0], choosers[1 - s0] = me, other
        env = RiftboundEnv(table, cfg, encoder=enc)
        obs = env.reset(seed, *deal(seed))
        row = Counter()
        plies: dict[tuple[int, int], dict] = {}
        # (active seat, ply, battlefield) of every combat the ENGINE initiated,
        # deduplicated: `run_combat` stamps `combat_at` on the log of *every*
        # action that resumes an ongoing combat, including the defender's
        # passes, so counting log hits counts priority windows rather than
        # combats. The key is what makes it a count of combats.
        eng_keys: set[tuple[int, int, int]] = set()
        # (seat, ply, battlefield) of every combat-staging commit, so an
        # engine-logged combat can be traced back to one. Matching on the KEY
        # rather than on "was the commit the action that logged it" matters: a
        # commit whose Cleanup suspends for a trigger-ordering decision has its
        # combat initiated by a later action, and comparing actions reported
        # most of these as combats no commit caused.
        commit_keys: set[tuple[int, int, int]] = set()
        eng_first: dict[tuple[int, int, int], str] = {}
        # Complaint metrics. `fought` holds (perm, ply) pairs that were in a
        # combat; `buffed` the own-unit buffs awaiting that verdict.
        fought: set[tuple[int, int]] = set()
        buffed: list[tuple[int, int]] = []
        hold_ply: dict[tuple[int, int], dict] = {}
        err = ""
        try:
            while obs is not None:
                st = env.state
                seat = int(obs.to_move)
                ply = int(st.ply)
                legal = env.legal
                watch = seat in tracked
                scoring = None
                if watch:
                    row["decisions"] += 1
                    # A ply counts as one of the agent's OWN turns only when it
                    # is the active player: it can commit a Move only then, and
                    # counting the plies where it merely held priority as a
                    # defender would inflate the denominator by ~25%.
                    if seat == int(st.active):
                        plies.setdefault((seat, ply), {"combats": 0,
                                                       "bfs": set()})
                    if len(legal) > 1:
                        scoring = scoring_actions(env, seat)
                        if scoring is None:
                            row["scan_skipped"] += 1
                if int(st.showdown_bf) >= 0:
                    sbl = bf_loc(int(st.showdown_bf))
                    for s_ in (0, 1):
                        for p_ in st.units_at(sbl, s_):
                            fought.add((int(p_), ply))
                idx = int(choosers[seat](env, obs))
                act = legal[idx]
                if act.kind == A.A_COMMIT and is_battlefield(int(st.decl_dst)):
                    dst_ = int(st.decl_dst)
                    if st.units_at(dst_, 1 - seat).size or st.units_at(dst_, seat).size:
                        here = [int(i) for i in st.declared()]
                        if st.units_at(dst_, 1 - seat).size:
                            for s_ in (0, 1):
                                here += [int(i) for i in st.units_at(dst_, s_)]
                            for p_ in here:
                                fought.add((p_, ply))
                if watch:
                    _complaints(env, st, seat, ply, act, legal, row, buffed,
                                hold_ply)
                    kinds[A.KIND_NAMES[act.kind]] += 1
                    if scoring:
                        if idx in scoring:
                            row["score_chances"] += 1
                            row["score_taken"] += 1
                        elif defers_score(env, seat, act,
                                          int(st.points[seat])):
                            # Still on offer after this action: deliberation,
                            # not a decline. Outside the denominator entirely.
                            row["score_deferred"] += 1
                        else:
                            row["score_chances"] += 1
                            row["score_declined"] += 1
                            if len(examples) < examples_wanted:
                                want = legal[sorted(scoring)[0]]
                                examples.append(
                                    f"game {g} turn {int(st.turn)} seat {seat}: "
                                    f"{describe(want, st, table, seat).strip()}"
                                    f" would have scored; played "
                                    f"{describe(act, st, table, seat).strip()}")
                    if act.kind == A.A_END_TURN and any(
                            a.kind == A.A_DECLARE for a in legal):
                        row["end_with_move"] += 1
                    if any(a.kind == A.A_DECLARE for a in legal):
                        row["move_chances"] += 1
                    if act.kind == A.A_COMMIT:
                        dst = int(st.decl_dst)
                        movers = [int(i) for i in st.declared()]
                        foes = ([int(i) for i in st.units_at(dst, 1 - seat)]
                                if is_battlefield(dst) else [])
                        if foes:
                            mine = sorted(set(movers) | set(
                                int(i) for i in st.units_at(dst, seat)))
                            ap = sum(combat.might_for_pool(st, table, i)
                                     for i in mine)
                            dp = sum(combat.might_for_pool(st, table, i)
                                     for i in foes)
                            row["combats"] += 1
                            commit_keys.add((seat, ply, bf_index(dst)))
                            p = plies.setdefault((seat, ply),
                                                 {"combats": 0, "bfs": set()})
                            p["combats"] += 1
                            p["bfs"].add(dst)
                            if ap < dp:
                                row["might_deficit"] += 1
                            kills = combat.solve_kills(st, table, ap, foes)
                            if len(kills) != len(foes):
                                row["no_wipe"] += 1
                        else:
                            row["commits_no_combat"] += 1
                active = int(st.active)
                res = env.step(idx)
                log = res.info.get("log") or {}
                if "combat_at" in log and active in tracked:
                    key = (active, ply, int(log["combat_at"]))
                    if key not in eng_keys:
                        eng_keys.add(key)
                        eng_first[key] = A.KIND_NAMES[act.kind]
                obs = res.obs
        except BoardOverflow:
            err = "board_overflow"
        except Exception as e:                       # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:160]
        for perm_, ply_ in buffed:
            if (perm_, ply_) not in fought:
                row["buff_idle"] += 1
        for h in hold_ply.values():
            if h["held"]:
                row["hold_chances"] += 1
                if not (h["held"] & h["went"]):
                    row["hold_uncontested"] += 1
        for (seat, ply), p in plies.items():
            row["plies"] += 1
            if p["combats"] >= 1:
                row["combat_turns"] += 1
                if p["combats"] >= 2:
                    row["multi_turns"] += 1
                if len(p["bfs"]) >= 2:
                    row["multi_bf_turns"] += 1
        # Same two numbers from the engine's own combat log, as a cross-check
        # on the commit-based definition above.
        # Not every combat starts with a commit: an [Ambush] unit played onto
        # an enemy battlefield stages one too (461 reads presence).
        for key in eng_keys - commit_keys:
            row["combats_not_from_commit"] += 1
            # WHICH action the engine first logged the combat under, so the
            # gap between the two counters is explained rather than assumed.
            row[f"nocommit_{eng_first.get(key, '?')}"] += 1
        eng_per_ply: Counter = Counter((s, p) for s, p, _ in eng_keys)
        row["engine_combats"] += len(eng_keys)
        row["engine_combat_turns"] += len(eng_per_ply)
        row["engine_multi_turns"] += sum(1 for v in eng_per_ply.values()
                                         if v >= 2)
        try:
            winner = int(env.state.winner)
            turns = int(env.state.turn)
        except Exception:                            # noqa: BLE001
            winner, turns = -1, 0
        rec = dict(row)
        rec.update({"game": g, "winner": winner, "turns": turns,
                    "tracked": list(tracked), "err": err})
        per_game.append(rec)
        for k, v in row.items():
            tot[k] += v
        if (g + 1) % 10 == 0 or g + 1 == n_games:
            cache[name] = {"spec": spec, "opp": opp, "games": g + 1,
                           "totals": dict(tot), "kinds": dict(kinds),
                           "per_game": per_game}
            json.dump(cache, open(out_path, "w"), indent=1)
            el = time.time() - t0
            print(f"  [{name}] {g + 1}/{n_games} games: "
                  f"{tot['decisions']:,} decisions, "
                  f"{tot['combat_turns']} combat-turns, "
                  f"{tot['multi_turns']} with 2+ "
                  f"| {el / (g + 1):.1f}s/game, "
                  f"{(n_games - g - 1) * el / (g + 1) / 60:.1f}min left",
                  flush=True)
    return {"spec": spec, "opp": opp, "games": n_games, "totals": dict(tot),
            "kinds": dict(kinds), "per_game": per_game, "examples": examples}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def boot_ratio(rows: list[dict], num: str, den: str, iters: int = 2000,
               seed: int = 7) -> tuple[float, float, float, int, int]:
    """Pooled ratio plus a 95% cluster bootstrap over GAMES.

    Resamples whole games because plies inside a game are correlated, and this
    project has already been burned by a rare event that clustered across games
    rather than spreading over them.
    """
    N = np.array([r.get(num, 0) for r in rows], float)
    D = np.array([r.get(den, 0) for r in rows], float)
    tot_n, tot_d = N.sum(), D.sum()
    if tot_d == 0:
        return float("nan"), float("nan"), float("nan"), int(tot_n), int(tot_d)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iters):
        pick = rng.integers(0, len(rows), len(rows))
        d = D[pick].sum()
        if d > 0:
            draws.append(N[pick].sum() / d)
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if draws
              else (float("nan"), float("nan")))
    return tot_n / tot_d, float(lo), float(hi), int(tot_n), int(tot_d)


def pct(x: float) -> str:
    return "  n/a " if x != x else f"{100 * x:5.1f}%"


def report(results: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("BEHAVIOUR  (ratio [95% cluster-bootstrap over games], numerator/denominator)")
    print("=" * 78)
    rows = [
        ("2+ committed combats / combat-turn", "multi_turns", "combat_turns"),
        ("  ... at 2+ distinct bfs", "multi_bf_turns", "combat_turns"),
        ("  ... same, from the engine's combat log",
         "engine_multi_turns", "engine_combat_turns"),
        ("combat-turns / ply", "combat_turns", "plies"),
        ("committed combat it does not wipe", "no_wipe", "combats"),
        ("committed combat, Might deficit", "might_deficit", "combats"),
        ("declined an immediate score", "score_declined", "score_chances"),
        ("ended turn with a Move available", "end_with_move", "move_chances"),
        ("-- Austin's complaints --", None, None),
        ("buff given to an ENEMY unit", "buff_enemy", "buffs"),
        ("own-unit buff SPELL, unit never fights that ply", "buff_idle", "buffs_own"),
        ("replay (banish+play) that changes nothing", "replay_pointless", "replays_own"),
        ("must-win attack leaving units home", "mustwin_left_home", "mustwin_attacks"),
        ("opponent's hold left uncontested", "hold_uncontested", "hold_chances"),
    ]
    for title, num, den in rows:
        if num is None:
            print(f"\n{title}")
            continue
        cells = []
        for r in results:
            p, lo, hi, n, d = boot_ratio(r["per_game"], num, den)
            cells.append(f"{pct(p)} [{pct(lo)},{pct(hi)}] {n}/{d}"
                         if d else f"  n/a  0/0")
        print(f"\n{title}")
        for r, c in zip(results, cells):
            print(f"    {label(r['spec']):<22} {c}")
    print("\ncounts")
    print(f"    {'agent':<22} {'games':>6} {'decisions':>10} {'plies':>7} "
          f"{'combats':>8} {'eng.cmb':>8} {'turns/g':>8} {'dec/turn':>9} "
          f"{'wins':>6} {'errs':>5}")
    for r in results:
        t, pg = r["totals"], r["per_game"]
        turns = sum(x["turns"] for x in pg) or 1
        # Meaningless in self-play -- a tracked seat always wins -- so it is
        # printed only when there is a distinct opponent to have beaten.
        wins = ("-" if r.get("opp") == "self"
                else str(sum(1 for x in pg if x["winner"] in x["tracked"])))
        print(f"    {label(r['spec']):<22} {r['games']:>6} "
              f"{t.get('decisions', 0):>10,} {t.get('plies', 0):>7} "
              f"{t.get('combats', 0):>8} {t.get('engine_combats', 0):>8} "
              f"{turns / len(pg):>8.1f} "
              f"{t.get('decisions', 0) / turns:>9.1f} {wins:>6} "
              f"{sum(1 for x in pg if x['err']):>5}")
    print("\nvalidation")
    for r in results:
        t = r["totals"]
        c, e = t.get("combats", 0), t.get("engine_combats", 0)
        nc = t.get("combats_not_from_commit", 0)
        print(f"    {label(r['spec']):<22} commit-detected {c}, engine-logged "
              f"{e} (of which {nc} started by something other than a commit), "
              f"[{', '.join(f'{k[9:]} {v}' for k, v in sorted(t.items()) if k.startswith('nocommit_'))}], "
              f"score chances {t.get('score_chances', 0)} "
              f"(taken {t.get('score_taken', 0)}, deferred "
              f"{t.get('score_deferred', 0)}), "
              f"skipped score scans {t.get('scan_skipped', 0)}")
    print("    greedy's 'does not wipe' must be ~0: it attacks only when it "
          "wipes every defender.")
    print("    a 0 with a denominator of 0 proves nothing; read the counts row.")
    print("\nwhat it spends its decisions on (share of the agent's own actions)")
    shares = [(r, sum(r["kinds"].values()) or 1) for r in results]
    for k in sorted({k for r in results for k in r["kinds"]},
                    key=lambda k: -sum(r["kinds"].get(k, 0) for r in results)):
        cells = " ".join(f"{label(r['spec'])} {100 * r['kinds'].get(k, 0) / n:4.1f}%"
                         for r, n in shares)
        print(f"    {k:<14} {cells}")
    for r in results:
        if r.get("examples"):
            print(f"\n  {label(r['spec'])} -- declined scores, first few:")
            for e in r["examples"]:
                print(f"    {e}")


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    n_games, opp, out = 30, "self", "rl/runs/behaviour.json"
    for flag, cast in (("--games", int), ("--opp", str), ("--out", str)):
        if flag in argv:
            i = argv.index(flag)
            v = cast(argv[i + 1])
            del argv[i:i + 2]
            if flag == "--games":
                n_games = v
            elif flag == "--opp":
                opp = v
            else:
                out = v
    if "--render" in argv:
        # Re-print the table from a cached JSON, without replaying anything.
        # The rows are written every 10 games, so a run killed at hour three
        # still reports.
        i = argv.index("--render")
        cache = json.load(open(argv[i + 1]))
        report(list(cache.values()))
        return 0
    specs = argv or ["random", "greedy", "rl/runs/v11-pool/last.pt"]

    table = full_table()
    # The current training game: victory 8, spells on (MEASURING.md's config
    # check -- a rules difference is invisible to a tensor shape).
    cfg = dc_replace(Config().at_victory_score(8), units_only=False,
                     choose_payment=True)
    pool = select_decks(table, None, 0.6)
    deal = deck_pool_deal(table, 0.6, seed_decks=pool)
    enc = Encoder(table, cfg)
    tr = Trainer(table, cfg, deal, HP(n_envs=2, rollout=16), device="cpu", seed=0)
    print(f"behaviour: {n_games} games/agent, opponent={opp}, "
          f"victory {cfg.victory_score}, units_only={cfg.units_only}, "
          f"seeds {SEED0}..{SEED0 + n_games - 1} (identical across agents)")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    results = []
    for spec in specs:
        print(f"\n== {label(spec)} ({spec})", flush=True)
        results.append(run_agent(spec, opp, n_games, table, cfg, deal, enc, tr,
                                 out_path, cache))
    report(results)
    json.dump(cache, open(out_path, "w"), indent=1)
    print(f"\n  per-game rows in {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
