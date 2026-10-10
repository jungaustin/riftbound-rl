"""A greedy heuristic baseline (PLAN.md Phase 2).

Deliberately shallow -- this exists to prove the engine rewards sane play, not
to play well. If it does not crush a random agent, either the engine is wrong or
the heuristic is, and finding that out here is far cheaper than finding it out
after a training run.

Priority order, and why:

  1. **Attack only when the maths says you wipe them.** Because attackers are
     Recalled if any defender survives (466.1.a.2), a failed attack achieves
     *nothing at all* -- no chip damage, no ground. So the bar is not "I win the
     fight", it is "my pool kills every defender".
  2. **Otherwise develop**, biggest body first, and never retreat.

The attack test is exact rather than heuristic, because with vanilla units the
outcome is a closed-form function of summed Might (see `combat.py`).

**This used to open with "play a unit onto an empty Battlefield -- free points,
an immediate Conquer with no Showdown to lose."** That line described an engine
bug, not the game: 806.3 restricts playing units to your base or a battlefield
you already **control**, so the free Conquer never existed. When the engine was
corrected the heuristic did not merely become useless, it became actively
harmful -- its filter looked for battlefields `bf_ctrl != seat`, which the
action list can no longer contain, so every unit fell through to the "stay
home" branch and greedy stopped garrisoning anything at all. A heuristic that
silently turns into a no-op is worse than one that is wrong out loud, which is
why the reasoning is written down here rather than left implicit in a filter.
Found by the project owner reading this file.

Deployment is now: **reinforce a battlefield we control whose garrison the
opponent could currently wipe**, otherwise stay at base where the unit keeps
its options. The test has to be about our garrison's survival, not about enemy
presence -- in a Neutral Open State we can never be sharing a battlefield,
because a unit arriving where the enemy stands stages a Combat that the next
Cleanup initiates. And greedy pays
[Accelerate] when it is playing to base, because a unit that enters ready can
join an attack this turn while an exhausted one cannot act until the next --
worth one rune of attrition to a shallow baseline. It does not accelerate a
unit it is parking on a battlefield, where readiness buys only a retreat.
"""

from __future__ import annotations

import numpy as np

from rl.engine import actions as A
from rl.engine import combat
from rl.engine.state import (P_ALIVE, P_CARD, P_CTRL, GameState, bf_index,
                             is_battlefield)


def _would_wipe(state: GameState, table, cfg, dst: int,
                extra: list[int]) -> bool:
    """Would sending `extra` to `dst` destroy every defender there?

    Uses the engine's own solver, so the baseline cannot disagree with the rules
    about Tank/Backline ordering or exact-lethal spending.
    """
    seat = state.active
    foe = 1 - seat
    defenders = list(state.units_at(dst, foe))
    if not defenders:
        return True
    attackers = list(state.units_at(dst, seat)) + extra
    pool = sum(combat.might_for_pool(state, table, i) for i in attackers)
    return len(combat.solve_kills(state, table, pool, defenders)) == len(defenders)


def _can_wipe(state: GameState, table, cfg, dst: int) -> bool:
    """Could every movable unit together wipe `dst`? The go/no-go for attacking."""
    movable = combat.movable_units(state, table, cfg, dst)
    return bool(movable) and _would_wipe(state, table, cfg, dst, movable)


def _garrison_at_risk(state: GameState, table, loc: int, seat: int) -> bool:
    """Could the opponent, committing everything, wipe our units at `loc`?

    The test that decides whether to reinforce. It has to be about *our*
    garrison rather than about sharing the battlefield, because in a Neutral
    Open State we can never be sharing one: a unit arriving where the enemy
    stands stages a Combat (461) and the next Cleanup initiates it, so by the
    time anyone is offered a Main-Phase action the battlefield is decided.
    Two successive versions of this heuristic tested for enemy presence here
    and were therefore unreachable code that quietly did nothing.
    """
    foe = 1 - seat
    ours = list(state.units_at(loc, seat))
    if not ours:
        return False
    theirs = [i for i in range(state.n_perms)
              if state.perms[i, P_ALIVE] == 1
              and int(state.perms[i, P_CTRL]) == foe
              and table.is_type(int(state.perms[i, P_CARD]), "Unit")]
    pool = sum(combat.might_for_pool(state, table, i) for i in theirs)
    return len(combat.solve_kills(state, table, pool, ours)) == len(ours)


def _biggest_enemy(state: GameState, table, seat: int, cands):
    """Prefer stunning the largest enemy unit; fall back to the smallest friend.

    Crude on purpose. The point of the baseline is to be a *fair yardstick*, not
    to play well -- but it has to use combat tricks at all, or "beats greedy"
    silently starts measuring an opponent that ignores half the game.

    Only meaningful for a UNIT slot: an `A_TARGET` arg is a permanent row only
    when the open slot says so. For a location, a chain uid or a card in the
    trash it is a different kind of number entirely, and `state.perms[arg]`
    would be reading a row chosen at random -- or, for a card id, off the end
    of the array. The caller checks the kind; this asserts it rather than
    trusting, because the failure is silent for three of the four kinds.

    **And the kind alone is not enough** -- see the comment on the partition
    below. `or_legend` widens a TK_UNIT slot to legends, whose args are
    sentinels, not rows.
    """
    from rl.engine import chain as chain_mod
    from rl.engine.effects import TK_UNIT
    assert chain_mod.open_slot_kind(state, table) == TK_UNIT
    # **The slot kind is NOT sufficient, which this used to assume.** A TK_UNIT
    # slot with `or_legend` (Profiteer: "a unit or legend") also offers LEGEND
    # sentinels, and `legend_src(0)` is -51 -- so `state.perms[-51]` raised
    # IndexError and killed a 5-hour training run inside its own eval.
    #
    # It crashes rather than corrupts only by luck of layout: every sentinel band
    # (legend -51, trash -61, delayed -71, battlefield -101) sits below
    # MAX_PERMS = 48, so none of them lands inside numpy's -48..-1 wrap window
    # where it would silently have read the wrong row instead.
    #
    # Partition rather than assert: "the biggest enemy" is only meaningful for a
    # real row, so rank the rows and fall back to a sentinel only when there is
    # no row to choose at all.
    rows = [a for a in cands if 0 <= int(a.arg) < int(state.n_perms)]
    if not rows:
        # Nothing rankable -- a legend or other sentinel is the only option.
        # Deterministic so the baseline stays reproducible.
        return min(cands, key=lambda a: int(a.arg))
    enemy = [a for a in rows if state.perms[a.arg, P_CTRL] != seat]
    pool = enemy or rows
    key = (max if enemy else min)
    return key(pool, key=lambda a: int(table.might[int(state.perms[a.arg, P_CARD])]))


def greedy_agent(rng: np.random.Generator):
    def choose(state: GameState, table, cfg, seat: int, legal):
        kinds = {}
        for a in legal:
            kinds.setdefault(a.kind, []).append(a)

        # Combat damage assignment (465.2.c.2). Take the engine's own solver's
        # answer, for the same reason `_would_wipe` does: the baseline must not
        # disagree with the rules about Tank/Backline ordering or exact-lethal
        # spending. It also keeps this baseline exactly as strong as it was
        # before the choice was handed to players -- falling through to
        # `legal[0]` would have made greedy worse and inflated every win rate
        # measured against it.
        if state.pend_dmg >= 0:
            want = combat.dmg_solver_choice(state, table, seat)
            if want >= 0:
                for a in kinds.get(A.A_PICK, []):
                    if int(a.arg) == want:
                        return a
            if A.A_PICK_NONE in kinds:
                return kinds[A.A_PICK_NONE][0]

        # Filling a target slot: stun the biggest thing that is not ours.
        # Only unit slots carry a permanent row -- for every other kind the
        # baseline has no opinion and takes the first offer, which is honest
        # about what it is rather than sorting a number it cannot read.
        if A.A_TARGET in kinds:
            from rl.engine import chain as chain_mod
            from rl.engine.effects import TK_UNIT
            if chain_mod.open_slot_kind(state, table) != TK_UNIT:
                return kinds[A.A_TARGET][0]
            return _biggest_enemy(state, table, seat, kinds[A.A_TARGET])

        if A.A_HIDE_AT in kinds:
            return kinds[A.A_HIDE_AT][0]

        # A priority window. Spend a trick only when defending a showdown --
        # that is where Stun actually converts into a kill, because the
        # attacker deals nothing and is still Recalled or destroyed.
        if A.A_PASS in kinds:
            defending = (state.showdown_bf >= 0 and int(state.attacker) != seat)
            if defending:
                for k in (A.A_PLAY_HIDDEN, A.A_PLAY):
                    if k in kinds:
                        return kinds[k][0]
            return kinds[A.A_PASS][0]

        # Mid-decision points: finish what was started.
        if A.A_PLAY_AT in kinds or A.A_PLAY_AT_FAST in kinds:
            # 806.3 -- every destination offered here is our base or a
            # battlefield we already control. Reinforce one whose garrison the
            # opponent could wipe; otherwise stay home, where the unit can move
            # next turn instead of being locked down exhausted.
            normal = kinds.get(A.A_PLAY_AT, [])
            at_risk = [a for a in normal
                       if is_battlefield(a.arg)
                       and _garrison_at_risk(state, table, a.arg, seat)]
            if at_risk:
                # Prefer one not yet Scored this turn: at one point from
                # victory the Final Point by Conquer needs every battlefield
                # Scored this turn (471.1.b), so holding an already-scored one
                # can be worth nothing.
                return min(at_risk,
                           key=lambda a: int(state.bf_scored[seat,
                                                             bf_index(a.arg)]))
            # Going to base. Pay [Accelerate] if offered -- a ready unit can
            # join this turn's attack, an exhausted one cannot act at all.
            for k in (A.A_PLAY_AT_FAST, A.A_PLAY_AT):
                home = [a for a in kinds.get(k, []) if not is_battlefield(a.arg)]
                if home:
                    return home[0]
            return (normal or kinds[A.A_PLAY_AT_FAST])[0]

        if A.A_ADD in kinds or A.A_COMMIT in kinds:
            # Recomputed each step rather than remembered: `GameState` has
            # __slots__, and a stateless agent cannot desync from the position.
            declared = state.declared()
            # Declared units have not moved yet, so they are `extra`, not
            # already-present. Getting that backwards silently commits attacks
            # that cannot win, and Recall makes those worth exactly nothing.
            if declared and _would_wipe(state, table, cfg, state.decl_dst,
                                        declared):
                return kinds[A.A_COMMIT][0]
            if A.A_ADD in kinds:
                # Add the biggest body first, so we commit the smallest group
                # that clears the bar -- everything sent is a unit not defending
                # somewhere else, and the garrison left behind is what scores.
                return max(kinds[A.A_ADD],
                           key=lambda a: combat.might_for_pool(state, table, a.arg))
            if A.A_COMMIT in kinds:
                return kinds[A.A_COMMIT][0]
            return kinds[A.A_CANCEL][0]

        if A.A_HIDE in kinds:
            return kinds[A.A_HIDE][0]

        # Develop: play a card if we have one. The destination choice above
        # then decides between reinforcing and staying home.
        if A.A_PLAY in kinds:
            # `A.played_card`, not `state.hand`: 108.3.d makes the Chosen
            # Champion playable from its own zone, and its arg is deliberately
            # out of range for the hand. Reading the hand raised IndexError on
            # every real-deck game the moment the champion became playable.
            return max(kinds[A.A_PLAY],
                       key=lambda a: int(table.might[
                           A.played_card(state, seat, int(a.arg))]))

        # 2. Attack only where we can wipe them, preferring a battlefield not
        # yet Scored this turn -- see the Final Point note above.
        winnable = [a for a in kinds.get(A.A_DECLARE, [])
                    if _can_wipe(state, table, cfg, a.arg)]
        if winnable:
            return min(winnable,
                       key=lambda a: int(state.bf_scored[seat, bf_index(a.arg)]))

        if A.A_END_TURN in kinds:
            return kinds[A.A_END_TURN][0]
        return legal[0]
    return choose
