"""A scripted HOLD-AND-SCORE opponent -- built to punish passivity.

`docs/REVIEW_HUMAN_GAMES.md`, step 4 of its plan. The policy leaves an
opponent's hold uncontested 27-33% of the time it could contest it
(`behaviour.py`, v14b-v16). Self-play cannot fix that: every opponent in the
pool is a relative of the learner and shares the habit, so nothing punishes
it. This agent does nothing BUT hold: it puts every unit it can on a
battlefield and keeps it there, so a learner that lets it sit loses to holds.

It differs from `greedy` in exactly three places, and falls back to greedy for
everything else (combat tricks, damage assignment, targets):

  1. **Deploy onto the board.** A unit goes to a battlefield it controls, the
     one with the weakest garrison first, never to base. Greedy parks units at
     base unless a garrison is at risk.
  2. **Take empty ground.** Any battlefield nobody holds is claimed with the
     smallest unit that can reach it, since an empty battlefield is a free
     Conquer and then a hold every turn.
  3. **Contest a hold it can break.** An enemy-held battlefield it can wipe is
     attacked first, ahead of greedy's other options.

It never retreats. It is not meant to be strong; it is meant to be the
opponent a passive policy cannot beat by doing nothing.

**Measured 2026-10-08, and NOT yet good enough to train against** (60 games,
real decks, seats alternating): beats random 92%, but loses to greedy (37%)
and to v16 (23%), and holds only 0.4-0.6 battlefields at the start of its own
turns -- it claims empty ground with units the opponent then wipes. A
punisher that loses 77% teaches the learner little. It is kept as a yardstick
and a starting point: the next version should claim only with a garrison the
opponent cannot wipe (`greedy._garrison_at_risk`), which is greedy plus
expansion, before it goes into any pool.
"""
from __future__ import annotations

import numpy as np

from rl.agents.greedy import _can_wipe, _would_wipe, greedy_agent
from rl.engine import actions as A
from rl.engine import combat
from rl.engine.state import GameState, bf_index, is_battlefield


def _garrison(state: GameState, table, loc: int, seat: int) -> int:
    return sum(combat.might_for_pool(state, table, int(i))
               for i in state.units_at(loc, seat))


def holder_agent(rng: np.random.Generator):
    fallback = greedy_agent(rng)

    def choose(state: GameState, table, cfg, seat: int, legal):
        kinds: dict[int, list] = {}
        for a in legal:
            kinds.setdefault(a.kind, []).append(a)
        foe = 1 - seat

        # 1. Deploy onto a battlefield we control, the thinnest garrison first.
        if A.A_PLAY_AT in kinds or A.A_PLAY_AT_FAST in kinds:
            spots = [a for k in (A.A_PLAY_AT_FAST, A.A_PLAY_AT)
                     for a in kinds.get(k, []) if is_battlefield(int(a.arg))]
            if spots:
                return min(spots, key=lambda a: (
                    _garrison(state, table, int(a.arg), seat),
                    a.kind != A.A_PLAY_AT))

        # Building a move: commit once it wins, or as soon as the target is
        # empty (one unit is enough to claim empty ground).
        if A.A_ADD in kinds or A.A_COMMIT in kinds:
            dst = int(state.decl_dst)
            declared = state.declared()
            empty = is_battlefield(dst) and not state.units_at(dst, foe).size
            if declared and (empty or _would_wipe(state, table, cfg, dst,
                                                  declared)):
                if A.A_COMMIT in kinds:
                    return kinds[A.A_COMMIT][0]
            return fallback(state, table, cfg, seat, legal)

        if A.A_DECLARE in kinds and A.A_PASS not in kinds:
            decl = [a for a in kinds[A.A_DECLARE] if is_battlefield(int(a.arg))]
            # 3. Break an enemy hold we can wipe.
            theirs = [a for a in decl
                      if int(state.bf_ctrl[bf_index(int(a.arg))]) == foe
                      and _can_wipe(state, table, cfg, int(a.arg))]
            if theirs:
                return theirs[0]
            # 2. Claim empty ground nobody holds.
            empty = [a for a in decl
                     if int(state.bf_ctrl[bf_index(int(a.arg))]) < 0
                     and not state.units_at(int(a.arg), foe).size]
            if empty:
                return empty[0]

        return fallback(state, table, cfg, seat, legal)

    return choose
