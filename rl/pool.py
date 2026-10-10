"""Opponent pool with PFSP sampling, and an Elo ladder -- PLAN.md Phase 5.

**Why this is not optional.** The spell-enabled run peaked against greedy at
iteration 100 and was 8.8 points worse by 150, with non-overlapping intervals,
while its win rate against random stayed flat. Nothing had got worse in any
absolute sense -- the policy had co-adapted to its own current self and lost
ground against a *fixed, different* opponent. Training only against your latest
self is what causes that: the opponent distribution is one point, it moves with
you, and there is no pressure to stay good against anything you have left
behind.

**PFSP, not uniform and not latest-only** (AlphaStar's scheme, §4 Phase 5).
Sample past checkpoints in proportion to how *informative* they are, which means
prioritising opponents you beat around half the time:

    weight(o) = f(P[win against o])        f(x) = (1 - x)^p

An opponent you beat 99% of the time teaches almost nothing; one that beats you
99% of the time gives gradients you cannot act on. `p` controls the sharpness --
p=2 is a reasonable default and 0 recovers uniform sampling.

**Matchup-aware PFSP (BACKLOG #21), behind `matchup=True`.** The weight above
is a function of ONE number per opponent -- how often the learner beats it,
averaged over every deck the learner piloted. That average is exactly what hid
the only lever that has ever measurably worked on this project: v10's exploiter
gained ~45 Elo by pinning the pool to a single frozen opponent, i.e. by deleting
the averaging. Strength is not a scalar; a specific opponent is beatable in a
specific matchup in a way the pool mean cannot see.

So with `matchup=True` the record is a MATRIX, one cell per
(frozen opponent, deck the learner is piloting), and the pair is sampled
jointly:

    weight(o, d) = (1 - P[learner wins | o, d]) ** p

Three properties this has to have, and does:

  * **An unseen cell is sampled before it has a record.** Same optimistic
    (w+1)/(g+2) prior as `Entry.learner_winrate`, so an empty cell reads 0.5.
  * **An all-empty matrix degrades to the current behaviour, not to a single
    cell.** Every cell reads 0.5, so every weight is 0.5**p and the joint
    distribution is exactly uniform -- which is uniform over opponents times
    uniform over decks, which is what an empty pool does today.
  * **The decay is per-cell, applied on that cell's own visits.** This is the
    one place the single-opponent code could not be lifted unchanged:
    `record` decays EVERY entry on every game, which over 20 entries leaves
    each one a few games of memory. Over a 20x47 = 940-cell matrix the same
    global decay erases a cell's record ~19x faster than that cell can be
    revisited, so the matrix would sit permanently at the 0.5 prior and the
    feature would be a no-op that looked implemented. Measured, not assumed:
    see `rl/tools/matchup_decay.py`. The rate itself is unchanged at 0.98, now
    read as "per visit to this cell" rather than "per game anywhere".

**Elo is measured against a frozen held-out set, never against the pool.**
Elo computed inside a moving population can rise while the population gets
weaker -- everyone beating everyone slightly better proves nothing. A held-out
ladder that never trains is the only version of the number that can be compared
across a run.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Entry:
    """One frozen opponent.

    `cells` is the per-matchup half (BACKLOG #21) and is empty unless the pool
    was built with `matchup=True`. It maps a **deck key -> [wins, games]** for
    the learner, where the key is the name of the decklist THE LEARNER PILOTED
    in that episode.

    **Why the learner's own decklist and not something finer.** The key has to
    be something the record site can actually supply. `Rollout.collect` records
    a result when an episode ends, by which point `VecRiftbound` has already
    auto-reset the slot and dealt a new game -- `env.summary()` carries no deck
    or legend, and the finished game's state is gone. So the only honest key is
    one the *dealer* chose and handed to the Trainer when the episode started
    (`ppo.MatchupDealer`), and the dealer's own unit of choice is an index into
    the deck pool. The learner's deck name is therefore observable by
    construction. Two alternatives were rejected:

      * the learner's **legend** -- also observable, and coarser (29 legends vs
        47 lists), but it is a function of the deck name anyway, so it buys
        density only by throwing away the distinction between two lists on the
        same legend, which is precisely the sideboard-level difference a
        matchup record is supposed to notice;
      * the **deck PAIRING** (learner deck x opponent deck) -- 47x47 per entry,
        ~44k cells for a pool of 20. At the observed rate of one episode per
        cell per several thousand games it would never hold a record at all.
    """
    name: str
    state_dict: dict
    iteration: int
    wins: float = 0.0        # decayed win count for the *learner*
    games: float = 0.0
    # **A pinned entry is never evicted and never falls below `pin_floor` of the
    # sampling mass.** Both halves are needed. Eviction is by age, so a pinned
    # ancestor would be the FIRST thing dropped; and PFSP weights by
    # `(1 - winrate)^p`, so an opponent the learner beats reliably sinks toward
    # zero sample share -- which is correct prioritization and also exactly how a
    # population forgets how to beat something it used to beat.
    pinned: bool = False
    # A checkpoint from before the payment choice (D6): it plays the engine's
    # old default there, as `nets.untrained_choice` does when a model is played.
    old_payment: bool = False
    cells: dict = field(default_factory=dict)   # deck key -> [wins, games]
    # Identity for caches (BACKLOG B2). **Never key anything on `id(entry)`**:
    # `add` evicts by popping, CPython frees the Entry, and the next one is
    # often allocated at the same address -- measured 49 of 200 adds at
    # capacity 3 -- so an id-keyed cache served the NEW snapshot the EVICTED
    # opponent's frozen net. Monotonic for the process, so never reused.
    uid: int = field(default_factory=itertools.count().__next__,
                     compare=False, repr=False)

    @property
    def learner_winrate(self) -> float:
        # Optimistic prior at 0.5 so a brand-new opponent is sampled before it
        # has any record, rather than being ignored or over-weighted.
        return (self.wins + 1.0) / (self.games + 2.0)

    def cell_winrate(self, key) -> float:
        """The learner's rate in ONE matchup, same optimistic prior as above.

        An unrecorded cell reads exactly 0.5 -- not 0 (never sampled) and not 1
        (sampled forever) -- which is what makes an all-empty matrix uniform.
        """
        w, g = self.cells.get(key, (0.0, 0.0))
        return (w + 1.0) / (g + 2.0)

    def cell_games(self, key) -> float:
        return float(self.cells.get(key, (0.0, 0.0))[1])


@dataclass
class OpponentPool:
    """Frozen past selves, sampled by PFSP."""
    capacity: int = 20
    p: float = 2.0
    decay: float = 0.98        # forget old results; the learner has moved on
    entries: list[Entry] = field(default_factory=list)
    # Minimum share of the sampling mass each PINNED entry keeps, however badly
    # it is losing. 0 disables the floor and restores pure PFSP.
    pin_floor: float = 0.05
    # BACKLOG #21. OFF by default: an unflagged run must sample exactly as it
    # did before, because every conclusion this project has reached came from a
    # single-variable run.
    matchup: bool = False

    def add(self, net, iteration: int, name: str | None = None) -> Entry:
        e = Entry(name=name or f"iter{iteration}",
                  state_dict={k: v.detach().clone().cpu()
                              for k, v in net.state_dict().items()},
                  iteration=iteration)
        self.entries.append(e)
        if len(self.entries) > self.capacity:
            # Drop the oldest UNPINNED entry. Oldest rather than weakest because
            # keeping a spread of ages is the point; unpinned because eviction by
            # age would otherwise remove a pinned ancestor first, which is the
            # one thing a pin is for.
            for i, old in enumerate(self.entries):
                if not old.pinned:
                    self.entries.pop(i)
                    break
            # If every entry is pinned the pool simply holds them all: pins are
            # few and deliberate, and silently dropping one would make a run's
            # opponent set differ from the command that launched it.
        return e

    def pin(self, net, iteration: int, name: str) -> Entry:
        e = self.add(net, iteration, name=name)
        e.pinned = True
        return e

    def weights(self) -> np.ndarray:
        if not self.entries:
            return np.zeros(0)
        w = np.array([(1.0 - e.learner_winrate) ** self.p for e in self.entries],
                     dtype=np.float64)
        total = w.sum()
        if total <= 0:
            w = np.full(len(w), 1.0 / len(w))
        else:
            w = w / total
        return self._apply_floor(w)

    def _apply_floor(self, w: np.ndarray) -> np.ndarray:
        """Lift every pinned entry to at least `pin_floor`, rescaling the rest.

        Written as a floor rather than a mixture because the guarantee wanted is
        "this opponent is still played", which is a statement about the minimum
        and not about the average. If the floors cannot all fit -- many pins, or a
        large floor -- they are shared out equally and the unpinned entries get
        nothing, which is the honest reading of the request rather than a silent
        partial application.
        """
        pins = np.array([e.pinned for e in self.entries])
        if not pins.any() or self.pin_floor <= 0:
            return w
        need = self.pin_floor * int(pins.sum())
        if need >= 1.0:
            out = np.zeros_like(w)
            out[pins] = 1.0 / int(pins.sum())
            return out
        out = w.copy()
        out[pins] = np.maximum(out[pins], self.pin_floor)
        spare = 1.0 - out[pins].sum()
        rest = w[~pins].sum()
        out[~pins] = (w[~pins] / rest * spare) if rest > 0 else (
            spare / max(1, int((~pins).sum())))
        return out / out.sum()

    # -- matchup-aware sampling (BACKLOG #21) -------------------------------

    def cell_weights(self, keys: list) -> np.ndarray:
        """PFSP weights over the whole (opponent x deck) grid, summing to 1.

        Shape `[len(entries), len(keys)]`. With no record anywhere every cell is
        `0.5 ** p`, so this is the uniform distribution -- i.e. independent
        uniform-over-opponents x uniform-over-decks, which is what the
        unflagged code does when the pool has no record either.
        """
        if not self.entries or not keys:
            return np.zeros((len(self.entries), len(keys)))
        w = np.array([[(1.0 - e.cell_winrate(k)) ** self.p for k in keys]
                      for e in self.entries], dtype=np.float64)
        total = w.sum()
        # Only reachable if EVERY cell reads a winrate of exactly 1.0, which
        # needs a cell with no losses and enough wins to swamp the prior. Uniform
        # is the right answer there: nothing in the grid is informative.
        if total <= 0:
            return np.full(w.shape, 1.0 / w.size)
        return w / total

    def sample_cell(self, rng: np.random.Generator, keys: list):
        """Joint sample of (opponent, learner deck). `(None, key)` if no pool.

        The deck is still returned when the pool is empty, because the dealer
        needs one either way -- it just falls back to uniform, which is what
        `deck_pool_deal` does on its own.
        """
        if not keys:
            return (self.sample(rng), None)
        if not self.entries:
            return (None, keys[int(rng.integers(len(keys)))])
        w = self.cell_weights(keys).ravel()
        flat = int(rng.choice(w.size, p=w))
        return (self.entries[flat // len(keys)], keys[flat % len(keys)])

    def sample(self, rng: np.random.Generator, deck_key=None) -> Entry | None:
        """Pick an opponent. With `deck_key` and `matchup`, pick it CONDITIONALLY.

        The conditional form exists for the one path that cannot sample jointly:
        a resumed run re-picks opponents for slots whose decks were already
        dealt by `VecRiftbound.reset`, so the deck is a given and only the
        opponent is free. p(opponent | deck) over the same cell weights.
        """
        if not self.entries:
            return None
        if self.matchup and deck_key is not None:
            col = np.array([(1.0 - e.cell_winrate(deck_key)) ** self.p
                            for e in self.entries], dtype=np.float64)
            tot = col.sum()
            w = (np.full(len(col), 1.0 / len(col)) if tot <= 0 else col / tot)
            return self.entries[int(rng.choice(len(col), p=w))]
        return self.entries[int(rng.choice(len(self.entries), p=self.weights()))]

    def record(self, entry: Entry, learner_won: bool, deck_key=None) -> None:
        for e in self.entries:               # decay everything, not just this one
            e.wins *= self.decay
            e.games *= self.decay
        entry.games += 1.0
        entry.wins += float(learner_won)
        if self.matchup and deck_key is not None:
            # **Per-cell, on this cell's own visit** -- see the module docstring.
            # Decaying all 940 cells here instead would throw a cell's record
            # away ~19x faster than the cell can be revisited.
            w, g = entry.cells.get(deck_key, (0.0, 0.0))
            entry.cells[deck_key] = [w * self.decay + float(learner_won),
                                     g * self.decay + 1.0]

    # -- state, for resume.pt ----------------------------------------------

    def records(self) -> dict:
        """The win records alone, keyed by entry name. No tensors.

        `ppo.py` rebuilds the pool's *nets* from the snapshots on disk, which
        carry no record at all; this is what lets the records survive a resume
        rather than silently resetting the matrix to the prior.
        """
        return {e.name: {"wins": float(e.wins), "games": float(e.games),
                         "cells": {k: [float(v[0]), float(v[1])]
                                   for k, v in e.cells.items()}}
                for e in self.entries}

    def load_records(self, rec: dict) -> int:
        """Re-attach `records()` by name. Returns how many entries matched.

        Deliberately forgiving: an entry with no saved record keeps the
        optimistic prior, and a saved record with no entry is dropped. Losing a
        record costs some sample efficiency; crashing, or quietly falling back
        to pure self-play, costs a run.
        """
        hit = 0
        for e in self.entries:
            r = rec.get(e.name)
            if not r:
                continue
            e.wins = float(r.get("wins", 0.0))
            e.games = float(r.get("games", 0.0))
            e.cells = {k: [float(v[0]), float(v[1])]
                       for k, v in (r.get("cells") or {}).items()}
            hit += 1
        return hit

    def summary(self) -> str:
        if not self.entries:
            return "pool: empty"
        w = self.weights()
        top = sorted(zip(self.entries, w), key=lambda t: -t[1])[:3]
        s = "pool: " + "  ".join(
            f"{e.name}@{e.learner_winrate:.0%}(w{p:.2f})" for e, p in top)
        return s + self._matchup_summary()

    def _matchup_summary(self) -> str:
        """What the MATRIX looks like, so a flagged run is legible from its log.

        Reports how much of the grid holds a record at all (the thing most
        likely to make this feature a no-op), and the worst cell the learner has
        actually played -- the hard matchup, which is the whole point.
        """
        if not self.matchup:
            return ""
        seen = [(e, k, v) for e in self.entries for k, v in e.cells.items()
                if v[1] > 0]
        if not seen:
            return "\n       matchups: 0 cells recorded (uniform)"
        games = sum(v[1] for _, _, v in seen)
        worst = min(seen, key=lambda t: t[0].cell_winrate(t[1]))
        best = max(seen, key=lambda t: t[0].cell_winrate(t[1]))
        return ("\n       matchups: {n} cells recorded over {tot} opponent(s)"
                " ({g:.0f} eff games)"
                "  worst {wo}/{wk}@{wr:.0%}  best {bo}/{bk}@{br:.0%}").format(
            n=len(seen), tot=len(self.entries), g=games,
            wo=worst[0].name, wk=worst[1],
            wr=worst[0].cell_winrate(worst[1]),
            bo=best[0].name, bk=best[1], br=best[0].cell_winrate(best[1]))


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------

def elo_update(ra: float, rb: float, score_a: float, k: float = 24.0):
    """One head-to-head result. `score_a` is A's score in [0, 1]."""
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))


def elo_from_winrate(p: float, anchor: float = 0.0) -> float:
    """Elo difference implied by a win rate. The natural way to report a duel.

    Clamped away from 0 and 1: a 100% result implies infinite Elo, which is an
    artefact of a finite sample rather than a measurement.
    """
    p = min(max(p, 1e-3), 1 - 1e-3)
    return anchor + 400.0 * math.log10(p / (1.0 - p))


@dataclass
class Ladder:
    """A frozen set of reference opponents, rated once and never retrained.

    Elo measured *inside* a self-play population can climb while the whole
    population weakens. This is the version of the number that can be trusted
    across a run, because the reference points never move.
    """
    names: list[str] = field(default_factory=list)
    ratings: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, rating: float = 0.0) -> None:
        self.names.append(name)
        self.ratings[name] = rating

    def rate(self, results: dict[str, float]) -> dict:
        """`results` maps opponent name -> learner win rate. Returns a report.

        The learner's rating is the sample-size-weighted mean of the Elo implied
        by each duel, which is a cheap approximation to a proper maximum-
        likelihood fit and is stable enough to plot.
        """
        est = [elo_from_winrate(p, self.ratings.get(n, 0.0))
               for n, p in results.items() if n in self.ratings]
        rating = float(np.mean(est)) if est else float("nan")
        return {"elo": rating, "vs": dict(results),
                "per_opponent_elo": {n: elo_from_winrate(p, self.ratings.get(n, 0.0))
                                     for n, p in results.items()}}
