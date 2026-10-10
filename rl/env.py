"""The environment -- PLAN.md Phase 3, §5.1.

**Not Gymnasium.** Gymnasium is single-agent, and this game is two-player with
alternating priority and a variable-length action list that fits no standard
`Space`. This is a plain class with a PettingZoo-AEC *shape*: one seat is to
move at a time, `step` advances until somebody has a real decision, and the
observation always belongs to whoever that is.

Two API decisions are deliberate and both exist to prevent specific bugs.

**`step` takes an index, not an action.** The policy picks a slot in the list
the env just handed it. That is what makes a variable-length action space
trainable at all (§5.1), and it means an off-by-one in the mask is impossible
to express rather than merely unlikely.

**Rewards are per seat, always both.** `StepResult.rewards[s]` is seat `s`'s
reward -- never "the reward for whoever just moved". The classic self-play bug
is crediting the final actor for the win (§5.3 gotcha 1), and it is a bug you
cannot write against this signature. The trainer keeps one buffer per seat and
writes both entries at game end.

**Auto-pass** collapses priority windows where `pass` is the only legal action.
Those are not decisions, and in v0 (no `[Reaction]` cards yet) every Showdown
window is one, which is most of the step count. The guard is strict and stays
strict: the list must be *literally* `[pass]`. The moment a Reaction is
affordable the window becomes a real decision -- and those windows are exactly
where bluffing lives (§5.3 gotcha 3), so auto-passing one would delete the
phenomenon this project is being built to study.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import numpy as np

from rl.config import Config
from rl.engine import actions as A
from rl.engine import game, invariants
from rl.engine.cardtable import CardTable
from rl.engine.state import N_SEATS, BoardOverflow, GameState
from rl.obs import Encoder, Obs


def should_auto_pass(legal: list[A.Action]) -> bool:
    """Is this priority window a no-op rather than a decision?

    Only when the legal set is *literally* `[pass]`. Deliberately not "any
    single forced action" and deliberately not "pass is available": the instant
    a `[Reaction]` is affordable the window becomes the most interesting
    decision in the game, and collapsing it would delete the bluffing behaviour
    Phase 7 exists to measure (§5.3 gotcha 3).

    A predicate rather than an inline condition because v0 cannot reach a
    Showdown priority window at all -- `combat.run_combat` resolves Combat
    without yielding -- so this rule is untestable through gameplay until
    Reactions land, and an untested rule guarding something that valuable
    should at least be testable on its own.
    """
    return len(legal) == 1 and legal[0].kind == A.A_PASS


class StepResult(NamedTuple):
    obs: Obs | None                  # None at termination
    rewards: tuple[float, float]     # indexed by SEAT, not by actor
    done: bool
    truncated: bool
    info: dict


class RiftboundEnv:
    """One self-play game. Reusable across episodes via `reset`."""

    def __init__(self, table: CardTable, cfg: Config,
                 encoder: Encoder | None = None, auto_pass: bool = True,
                 check: bool = False, privileged: bool = True) -> None:
        self.table = table
        self.cfg = cfg
        self.enc = encoder or Encoder(table, cfg, privileged=privileged)
        self.auto_pass = auto_pass
        self.check = check

        self.state: GameState | None = None
        self._legal: list[A.Action] = []
        self._obs: Obs | None = None
        self.steps = 0            # decisions the policy was asked for
        self.auto_passes = 0      # windows collapsed without asking
        self.n_legal_hist: Counter = Counter()
        # Sterile-loop detection. Positions already offered to a policy this
        # episode, as (state_hash, seat to move) -- see `cfg.loop_watch_after`.
        self._seen: set[tuple[int, int]] = set()
        self.loop_loser = -1      # >= 0 once a seat is caught looping
        self.board_overflow = False   # this episode hit `MAX_PERMS`
        self._last_log: dict = {}     # the log `_apply` produced, for `info`

    # -- properties ------------------------------------------------------

    @property
    def to_move(self) -> int:
        return -1 if self._obs is None else self._obs.to_move

    @property
    def legal(self) -> list[A.Action]:
        """The engine actions backing the current observation, in mask order."""
        return self._legal

    @property
    def done(self) -> bool:
        return self.state is None or A.is_terminal(self.state)

    # -- lifecycle -------------------------------------------------------

    def reset(self, seed: int, decks, rune_decks, battlefields,
              legends=None, champions=None, match=None) -> Obs:
        # `champions` is 5th and optional so `reset(seed, *deal_fn(seed))` takes
        # a dealer that supplies one (`decks.matchup`) and one that cannot
        # (`v1_deal` has no decklist, so no Chosen Champion) without either
        # knowing about the other.
        #
        # `match` (a `pregame.MatchCtx`) plays the next game of a Bo3. Its
        # knowledge of the opponent's list is what earlier games SHOWED
        # (`seen_match`, the project owner's call), never the open-list flag.
        known = ([0] * 2 if match is not None
                 else game.deck_knowledge(self.cfg, seed))
        self.state = game.new_game(self.table, self.cfg, decks, rune_decks,
                                   battlefields, seed=seed, legends=legends,
                                   champions=champions, deck_known=known,
                                   match=match)
        self._new_episode()
        self._advance()
        assert self._obs is not None, "game was terminal before the first move"
        return self._obs

    def reset_from(self, state: GameState) -> Obs | None:
        """Attach to an existing position -- for tests, replays and search."""
        self.state = state
        self._new_episode()
        self._advance()
        return self._obs

    def _new_episode(self) -> None:
        self.steps = 0
        self.auto_passes = 0
        self._seen.clear()
        self.loop_loser = -1
        self.board_overflow = False

    def step(self, action_index: int) -> StepResult:
        assert self.state is not None, "step before reset"
        assert self._obs is not None, "step after termination"
        assert 0 <= action_index < len(self._legal), (
            f"action index {action_index} outside the {len(self._legal)} legal "
            f"actions; the policy must sample under the mask")

        act = self._legal[action_index]
        if not self._apply(act):
            return StepResult(
                obs=None, rewards=self.final_rewards(), done=True,
                truncated=True, info={"action": act, "board_overflow": True})
        self.steps += 1
        self._advance()

        s = self.state
        # A loop inside one turn never advances the turn, so `turn_cap` cannot
        # see it. Truncating here makes a livelock cost one episode instead of
        # an entire run, and marks it the same way a too-long game is marked.
        if self.steps > self.cfg.decision_cap and not A.is_terminal(s):
            s.truncated = True
        self._check_sterile_loop()
        done = A.is_terminal(s) or bool(s.truncated)
        return StepResult(
            obs=self._obs,
            rewards=self.final_rewards() if done else (0.0, 0.0),
            done=done,
            truncated=bool(s.truncated),
            info={"action": act, "log": self._last_log},
        )

    def _check_sterile_loop(self) -> None:
        """End the episode if the position the policy is being offered is one it
        has already been offered (`cfg.loop_watch_after`).

        `state_hash` digests every slot, so an exact repeat with the same seat
        to move means the game returned to a position it has already been in --
        nothing in between made progress. That is the livelock signature, and
        catching it here rather than at `decision_cap` both saves the wasted
        decisions and says WHO was looping.

        **Attribution is only claimed where it is honest.** A seat with more
        than one legal action chose the loop over an alternative, so it takes
        the loss. A seat with exactly one legal action was forced into it, and
        blaming it would train against a decision it never made -- that case is
        left as an ordinary truncation and decided on points by 408.2.b.

        `ply` and `turn` are part of the digest and only ever increase, so this
        can only ever catch a loop INSIDE one turn. That is deliberate and it is
        the whole gap: a stall that does advance turns is what `turn_cap` is for.
        """
        s = self.state
        if (self._obs is None or s.truncated or A.is_terminal(s)
                or self.steps <= self.cfg.loop_watch_after):
            return
        key = (s.state_hash(), int(self._obs.to_move))
        if key not in self._seen:
            self._seen.add(key)
            return
        s.truncated = True
        if len(self._legal) > 1:
            self.loop_loser = int(self._obs.to_move)
        self._legal, self._obs = [], None

    def final_rewards(self) -> tuple[float, float]:
        """Per-seat terminal reward, including a sterile-loop loss.

        `A.outcome` is the rules answer (a win, or 408.2.b on a truncation); the
        loop loss is a house rule about a position the rules have nothing to say
        about, so it lives here rather than in the engine.
        """
        assert self.state is not None
        if self.loop_loser >= 0:
            return (-1.0, 1.0) if self.loop_loser == 0 else (1.0, -1.0)
        return A.outcome(self.state, self.cfg)

    # -- the driver ------------------------------------------------------

    def _apply(self, act) -> bool:
        """Apply one action. False means the board ran out of permanent ROWS and
        this episode is over.

        **Every `A.apply` in this class goes through here**, and that is the
        point: the first version wrapped only `step`'s own call, and the
        victory-8 run died at iteration 8 a second time because the overflow came
        from the AUTO-PASS inside `_advance` -- an auto-passed A_PASS resolves the
        Chain top, which is exactly where a token-maker resolves.

        A board-row overflow ends the episode rather than the run, the same way
        `decision_cap` does. The cap is on rows CREATED in one turn (they are
        only reclaimed at end of turn), and measured at victory 8 random play
        peaks at 33 of 48 and greedy at 36 -- so what reaches it is a policy, not
        the rules. The position is mid-mutation once it raises, so the episode is
        discarded; nothing reads it afterwards but the point totals, which are
        already written.
        """
        try:
            self._last_log = A.apply(self.state, self.table, self.cfg, act)
        except BoardOverflow:
            self.state.truncated = True
            self.board_overflow = True
            self._legal, self._obs = [], None
            return False
        return True

    def _advance(self) -> None:
        """Run the engine forward to the next real decision.

        Mirrors `game.play_game`'s loop exactly, because Phase 3's exit test is
        that a random policy here reproduces that loop bit for bit. Any
        divergence in control flow -- an extra invariant call that consumed
        randomness, a different pass ordering -- would show up as a hash
        mismatch there, which is the point of writing the test that way.
        """
        s = self.state
        assert s is not None
        while True:
            if A.is_terminal(s):
                self._legal, self._obs = [], None
                return
            seat = A.acting_seat(s)
            if seat < 0:
                self._legal, self._obs = [], None
                return
            legal = A.legal_actions(s, self.table, self.cfg, seat)
            if self.check:
                invariants.check(s)
                invariants.check_actions(s, self.table, self.cfg, seat, legal)
            if not legal:
                raise invariants.InvariantError(
                    f"seat {seat} to act with no legal actions "
                    f"on turn {s.turn}")

            if self.auto_pass and should_auto_pass(legal):
                if not self._apply(legal[0]):
                    return          # the board overflowed; the episode is over
                self.auto_passes += 1
                continue

            self.n_legal_hist[len(legal)] += 1
            self._legal = legal
            self._obs = self.enc.encode(s, seat, legal)
            return

    # -- convenience -----------------------------------------------------

    def summary(self) -> dict:
        """Same shape as `game.play_game`'s return, so results are comparable."""
        s = self.state
        assert s is not None
        return {
            "winner": int(s.winner),
            "truncated": bool(s.truncated),
            "turns": int(s.turn),
            "steps": self.steps,
            "points": s.points.tolist(),
            "hash": s.state_hash(),
            "loop_loser": self.loop_loser,
            "board_overflow": self.board_overflow,
        }


def random_policy(rng: np.random.Generator):
    """Uniform over legal actions, by index.

    Draws exactly one integer per decision from `rng`, in the same order as
    `game.random_agent`, so the two are the same policy under the same seed.
    That equality is what the Phase 3 exit test checks.
    """
    def choose(obs: Obs) -> int:
        return int(rng.integers(obs.n_legal))
    return choose


def policy_from_agent(agent):
    """Adapt an engine-level agent `(state, table, cfg, seat, actions) -> Action`
    into an index policy, so the greedy baseline can drive the env unchanged."""
    def choose(env: "RiftboundEnv") -> int:
        s = env.state
        act = agent(s, env.table, env.cfg, env.to_move, env.legal)
        return env.legal.index(act)
    return choose


def play(env: RiftboundEnv, policies, seed: int, decks, rune_decks,
         battlefields, legends=None, champions=None) -> dict:
    """Run one episode with per-seat index policies `policies[seat](obs)`."""
    obs = env.reset(seed, decks, rune_decks, battlefields, legends, champions)
    while obs is not None:
        r = env.step(policies[obs.to_move](obs))
        # **`r.done`, not `r.obs is not None`.** A truncation leaves the next
        # observation in place (the position is legal, the episode is simply
        # over), so looping on the observation alone spun forever the moment
        # `decision_cap` or a sterile loop fired. Latent while `trunc` sat at
        # 0.0%; `vec.step` has always keyed off `done`.
        obs = None if r.done else r.obs
    out = env.summary()
    out["rewards"] = env.final_rewards()
    out["auto_passes"] = env.auto_passes
    return out
