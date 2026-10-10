"""Search at inference, in the one subgame where it is sound -- BACKLOG #28.

**Why.** `rl/tools/lethal_probe.py` measured the policy over 57,152 decisions: a
forced win was available 184 times and it threw **39** of them away, one per 10.3
games, and in 15 of the 31 games containing a miss it went on to lose. So a
thrown-away forced win coincides with losing in **3.8% of games** -- comparable to
an entire four-hour training run's +28 Elo, for a fix that needs no retraining.

**What is sound here, and why it is only this.** A line in which the acting seat
**keeps priority the whole way** gives the opponent no window at all. Nothing they
hold can answer it, so no read and no belief about hidden information is required:
the win is forced in the strongest sense. That is the entire claim. Lines that pass
through an opponent window may also be forced, but proving it needs adversarial
search over their hidden options, and that is not this.

**So this never guesses.** It either proves a win and plays it, or it returns
nothing and the policy decides. There is no heuristic fallback, because a wrong
override is strictly worse than no override -- the policy at least knows what it
does not know.

**The gate is what makes it affordable.** The full search costs ~0.4s per
decision, which is fine for one move at a keyboard and hopeless for a 300-game
duel at ~125 decisions each. But a forced win must *reach* the victory score, so a
seat far from it cannot have one. Gating on `victory - points <= reach` skips the
large majority of decisions for nothing, and `reach` is the one knob: too small
silently misses wins, too large costs time. Default 3, because the measured ways
the 8th point falls (Hold, Conquer with the board Scored, a point-gain effect)
cannot produce more than about that in one uninterrupted run.

**Gate validated, and the first attempt at validating it was vacuous.** Checking
it over 12 games found "0 wins skipped" -- out of 0 wins in 1,582 decisions, so it
proved nothing. The probe's own data explains why: forced wins **cluster**, with
339 of 400 games holding none and 61 games holding all 184, and games 0-11 are
among the empty ones. Re-run against games the probe had already found wins in, the
gate let through 2 of 2 and skipped none. **Both occurred at `need == 1`**, which
suggests `reach` could tighten to 1 and cut the cost a long way -- but two events
is not enough to move a safety margin on, so the default stays 3.
"""
from __future__ import annotations

from rl.engine import actions as A
from rl.engine.state import BoardOverflow

# Per-decision search budget. The probe used 4000/8 and found 184 wins across
# 57,152 decisions; these are the same numbers so the solver inherits that
# measurement rather than a fresh guess.
NODE_CAP = 4000
DEPTH_CAP = 8
# How close to the victory score a seat must be before searching at all.
REACH = 3


def could_win_now(state, cfg, seat: int, reach: int = REACH) -> bool:
    """The cheap gate: is this seat near enough to the victory score to matter?

    Reads `state.victory_score`, which BACKLOG #10 made the single source of
    truth on 2026-10-02. The original version of this docstring said the opposite
    -- that the cfg was the authority because `check_winner` took it as an
    argument -- which was true when written and is now exactly backwards. `cfg` is
    still accepted so the signature does not churn.
    """
    need = int(state.victory_score) + int(state.victory_bonus) - int(state.points[seat])
    return need <= reach


def forced_win(state, table, cfg, seat: int, node_cap: int = NODE_CAP,
               depth_cap: int = DEPTH_CAP) -> list | None:
    """A line of `seat`'s own actions that wins without ever ceding priority.

    Returns the action sequence, or None. Depth-first, first win found rather
    than the shortest -- the question is existence, and any proven win is enough.
    """
    budget = [node_cap]

    def rec(st, depth: int) -> list | None:
        if budget[0] <= 0 or depth > depth_cap:
            return None
        try:
            acts = A.legal_actions(st, table, cfg, seat)
        except Exception:
            return None
        for act in acts:
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            nxt = st.clone()
            try:
                A.apply(nxt, table, cfg, act)
            except (BoardOverflow, AssertionError, ValueError):
                continue
            if A.is_terminal(nxt):
                if int(nxt.winner) == seat:
                    return [act]
                continue
            # The moment the opponent could act, the line stops being forced.
            if int(nxt.priority) != seat:
                continue
            deeper = rec(nxt, depth + 1)
            if deeper is not None:
                return [act] + deeper
        return None

    return rec(state, 0)


class Solver:
    """Wraps a decision: prove a win and take it, or defer to the policy.

    Keeps counters so a run can report how often it fired, which is the only
    honest way to tell a solver that is helping from one that never triggers.
    """

    def __init__(self, reach: int = REACH, node_cap: int = NODE_CAP,
                 depth_cap: int = DEPTH_CAP) -> None:
        self.reach, self.node_cap, self.depth_cap = reach, node_cap, depth_cap
        self.gated = 0          # decisions the cheap gate skipped
        self.searched = 0       # decisions actually searched
        self.found = 0          # searches that proved a win
        self.overrode = 0       # times the proof changed the move

    def choice(self, state, table, cfg, seat: int, legal: list,
               policy_idx: int) -> int:
        """The index to play. `policy_idx` is what the network wanted."""
        if not could_win_now(state, cfg, seat, self.reach):
            self.gated += 1
            return policy_idx
        self.searched += 1
        line = forced_win(state, table, cfg, seat, self.node_cap, self.depth_cap)
        if not line:
            return policy_idx
        self.found += 1
        try:
            idx = legal.index(line[0])
        except ValueError:
            # The proven move is not in the mask-ordered list the policy saw.
            # Should not happen -- both come from `legal_actions` -- so defer
            # rather than play an index that means something else.
            return policy_idx
        if idx != policy_idx:
            self.overrode += 1
        return idx

    def summary(self) -> str:
        tot = self.gated + self.searched
        return (f"solver: {tot:,} decisions, gate skipped {self.gated:,} "
                f"({self.gated / max(1, tot):.1%}), searched {self.searched:,}, "
                f"proved a win {self.found}, changed the move {self.overrode}")
