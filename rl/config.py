"""Every v1 scope knob, in one place.

The plan (PLAN.md §3) freezes v1 scope deliberately, and the main way a project
like this dies is re-opening that scope mid-build. Keeping the knobs here rather
than scattered through the engine means a scope change is a diff you can read.

Two of these are curriculum values rather than rules values -- `victory_score` and
`turn_cap`. The real game is 8 points (rule 194.3); we train at 3 first because
episodes get ~3x shorter and credit assignment over 40 decisions is far easier
than over 150. Anneal with `Config.at_victory_score(5)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Keywords
# ---------------------------------------------------------------------------
# Every bracketed keyword in the card pool, ordered. Index in this tuple IS the
# bit position in a card's keyword mask, so DO NOT reorder -- append only.
ALL_KEYWORDS = (
    "Reaction", "Action", "Empowered", "Equip", "Hidden", "Deflect", "Empower",
    "Ganking", "Add", "Accelerate", "Temporary", "Tank", "Deathknell", "Repeat",
    "Flow", "Assault", "Shield", "Stun", "Ambush", "Weaponmaster", "Legion",
    "Vision", "Mighty", "Buff", "Hunt", "Predict", "Backline", "Unique",
    # 819 -- appended rather than slotted in alphabetically, because the index
    # into this tuple IS the `kw_mask` bit and the feature-matrix column.
    "Quick-Draw",
    # The RAD set (spoiler season). Appended for the same reason, and note
    # that appending here WIDENS the card feature matrix, so it invalidates
    # older checkpoints. "Show Off" is the first two-word keyword; the
    # bracket parser already allows spaces, so nothing there needed changing.
    "Deploy", "Show Off", "Disarm",
)

# The v1 SCOPE TARGET -- keywords v1 intends to reach. This is an aspiration
# and a filter for the v0 vanilla pool. It is **not** a claim that the engine
# implements them, and reading it as one is what let 169 deck slots be counted
# as fully covered while carrying a keyword nothing ever executes.
#
# `Hidden` and `Deathknell` are in scope deliberately against the "keep v1
# simple" instinct -- see PLAN.md §3. Hidden appears in all ten of the repo's
# decks and is the Facedown Zone mechanic the whole belief experiment rests on
# (PLAN.md Phase 6); cutting it would cut the point of the project.
TIER1_KEYWORDS = (
    "Action", "Reaction", "Tank", "Backline", "Shield", "Assault", "Legion",
    "Mighty", "Deflect", "Temporary", "Hidden", "Deathknell",
)

# The keywords the engine ACTUALLY reads, each with the line that reads it.
# Anything here must be greppable; anything not here is ignored at runtime, so
# a card carrying it is not being played as printed no matter how short its
# text is. Keep this list honest -- it is the input to the coverage metric, and
# a coverage number that flatters itself is worse than no number.
ENGINE_KEYWORDS = {
    "Temporary": "phases.py -- expires in the Beginning Phase",
    "Tank":      "combat._tiers -- assigned combat damage first",
    "Backline":  "combat._tiers -- assigned combat damage last",
    "Ganking":   "combat.can_move -- may move battlefield to battlefield",
    "Hidden":    "chain.hideable -- may be hidden in the Facedown Zone",
    "Accelerate": "actions.legal_actions -- the A_PLAY_AT_FAST variant, and "
                  "cost.accelerate_cost for the additional cost (805)",
    "Vision":    "effects._vision_abilities -- 817.1.b is 'When this is "
                 "played, predict', synthesised from the keyword like [Hunt] "
                 "rather than transcribed per card (436.1 Predict)",
    "Repeat":    "actions.legal_actions -- the A_PLAY_REPEAT variant, "
                 "cardtable.repeat_cost for the additional cost, and "
                 "chain.resolve_top for the one extra execution (820)",
    "Shield":    "combat.combat_role_bonus -- +X Might while a defender (814)",
    "Buff":      "combat.might -- a Buff counter is +1 Might (702/703)",
    "Flow":      "chain.flow_playable -- play from the trash for the Flow "
                 "cost, then banish it (829)",
    "Deflect":   "resolve.deflect_cost -- opponents pay +X Power to choose it "
                 "(809), and an unaffordable surcharge makes it not a target",
    "Hunt":      "effects.abilities_for -- synthesised as 'when I conquer or "
                 "hold, gain N XP' on every card carrying it (no per-card entry)",
    "Ambush":    "actions.play_destinations adds battlefields where you "
                 "control units, and actions.ambush_playable gives the "
                 "conditional [Reaction] speed that goes with them (822.1.b)",
    "Assault":   "combat.combat_role_bonus -- +X Might while an attacker (807)",
    "Action":    "effects.SPEED_ACTION, via a card's DSL spec",
    "Reaction":  "effects.SPEED_REACTION, via a card's DSL spec",
    "Equip":     "effects._equip_abilities -- 818.1.c.2 is '[Cost]: Attach "
                 "this gear to a unit you control', synthesised from the "
                 "keyword like [Hunt]; cardtable.equip_cost reads the cost",
    "Quick-Draw": "effects._quick_draw_abilities for 819.1.d's 'when you play "
                  "this, attach it to a unit you control'; the [Reaction] half "
                  "(819.1.b) is a speed and is read where speeds are",
    "Unique":    "a DECK-CONSTRUCTION limit (one copy per deck); nothing about "
                 "it happens in a game, so there is nothing to execute",
    "Weaponmaster": "effects._weaponmaster_abilities for 821.1.c's optional "
                    "play trigger; OP_WEAPONMASTER pays the Equip cost less "
                    "[A] (off any Power, domain runes included -- 356.4.f.1) and attaches",
    "Deploy":    "actions.play_destinations for 'play this only to a "
                 "battlefield' (it REPLACES 149.2's base-only set for gear), "
                 "and effects._deploy_abilities + phases' Hold step for 'when "
                 "an opponent holds here, kill this' (TR_ENEMY_HOLDS_HERE)",
    "Disarm":    "effects._disarm_abilities -- 'When I attack, give an enemy "
                 "unit here -1 Might this turn', synthesised from the keyword "
                 "like [Hunt] because the effect is fixed by the keyword "
                 "rather than written on the card",
}

# **Reading a keyword is not the same as playing the card as printed.** These
# two are honest entries above -- the engine really does attach for [Equip] and
# [Quick-Draw] -- and Equipment is still withheld from the coverage metric,
# because a card's Might Bonus is a printed field of its own (137.3) that
# `data/cards.json` does not carry. See `decks.EQUIP_NEEDS_DATA`.
# Keywords that ARE a triggered ability rather than a passive property --
# 808.1 makes [Deathknell] short for "When I die, [Effect]", and the effect is
# the card's own text. There is no separate keyword implementation to check: a
# card carrying one is played as printed exactly when its ability is
# transcribed in `effects.ABILITIES`, which `decks.plays_as_printed` tests.
# **Only [Deathknell] qualifies, and the distinction is the whole point.**
# 808.1 makes it short for "When I die, <the card's own effect>", so the effect
# is different on every card and transcribing the ability IS implementing the
# keyword. [Hunt] and [Vision] are the opposite: their effect is FIXED by the
# keyword ("gain N XP on a Score", "look at the top card"), identical on every
# card, so one generic implementation covers all of them and a card's own
# transcription says nothing about whether it exists. Listing those two here
# forgave them for any card that merely had a spec -- no card had tripped it
# yet, but the first encoded [Vision] unit would have been counted as fully
# played while doing nothing. [Hunt] is now implemented generically and lives
# in ENGINE_KEYWORDS; [Vision] is simply not implemented.
ABILITY_KEYWORDS = ("Deathknell",)

# In scope, but nothing executes them yet. Named rather than merely absent so
# the gap is legible: Deathknell alone is 94 deck slots and is a trigger, so it
# lands with the triggered-ability machinery rather than on its own.
UNIMPLEMENTED_TIER1 = tuple(k for k in TIER1_KEYWORDS if k not in ENGINE_KEYWORDS)

# Keywords that are pure combat-damage assignment ordering (PLAN.md §1.3.b).
# "I must be assigned combat damage first" / "...last".
DAMAGE_ORDER_FIRST = "Tank"
DAMAGE_ORDER_LAST = "Backline"

DOMAINS = ("Body", "Calm", "Chaos", "Fury", "Mind", "Order")
CARD_TYPES = ("Unit", "Spell", "Gear", "Battlefield", "Legend", "Rune")


@dataclass(frozen=True)
class Config:
    # --- rules constants (real values; do not treat as tunable) -------------
    victory_score_full: int = 8       # rule 194.3
    battlefields_in_play: int = 2     # rule 485.4 -- one chosen by each player
    battlefields_per_player: int = 3  # each player brings 3, burns one per game
    rune_deck_size: int = 12          # rule 103.3.a
    channel_per_turn: int = 2         # rule 315.3.b
    second_player_bonus_runes: int = 1  # rule 485.7

    # --- curriculum --------------------------------------------------------
    # Anneal 5 -> 8. NOT 3 any more: the Final Point restriction (471.1.b)
    # applies once a player is within one point of victory, which at
    # victory_score=3 means from 2 points onward -- most of the game. That
    # turns an endgame rule into a permanent one and makes the curriculum a
    # materially different game. Measured: greedy beats random 97.2% at
    # victory 8 but only 77.3% at victory 3, because the restriction dominates
    # there rather than gating the finish.
    victory_score: int = 5            # anneal 5 -> 8
    turn_cap: int = 30
    # **408.2.b -- how a game that runs out of clock is actually decided.** It
    # is NOT a draw by default: "a player is declared the winner of the game if
    # they have a point lead of two or more. If no player has a point lead of
    # two or more, the game is a draw." Paying a flat 0 on truncation instead
    # rewarded a LOSING player for stalling, since 0 beats the -1 they were
    # heading for. Latent at victory 5 (`trunc=0.0%`), and the reason this had
    # to be fixed before the victory-8 curriculum, which gives a stall far more
    # room to pay off. `actions.outcome` reads this.
    truncation_lead_to_win: int = 2
    # **A turn cap cannot catch a loop INSIDE one turn.** A livelocked priority
    # window never advances the turn, so `turn_cap` never fires and the episode
    # runs forever -- which is what dragged a mirror-match run's mean episode
    # from ~36 decisions to over 200 and starved it of finished games.
    #
    # Sized from measurement, not taste: across 4,000 real-deck games at
    # victory 3 the longest legal game took 352 decisions, and 1,500 at victory
    # 8 stayed under it. 1500 is >4x that ceiling, so a game reaching it is not
    # a long game.
    decision_cap: int = 1500
    # **Sterile-loop detection.** An exact `state_hash` repeat with the same
    # seat to move is a position the game has already been in, so nothing in
    # between made progress -- provable, because the digest covers every slot.
    # Ending there instead of grinding to `decision_cap` saves ~1100 wasted
    # decisions AND lets the loss be attributed: a seat that had another legal
    # action and chose the loop anyway loses it. A seat with exactly one legal
    # action was forced, so that case falls back to 408.2.b on points -- a
    # forced loop is not the looper's fault and scoring it as one would teach a
    # lie.
    #
    # Hashing costs 67us, which is ~13% of a decision, so it does not run on
    # every one. It starts only once an episode is longer than any legal game
    # measured -- and the number to measure against is **victory 8**, not the
    # victory-3 curriculum: 1,000 real-deck games at victory 8 ran to a mean of
    # 235 decisions and a max of 529, against 352 at victory 3. 700 is past that
    # ceiling, so no healthy game pays for the hash, and it still ends a
    # livelock 800 decisions before `decision_cap` would.
    loop_watch_after: int = 700

    # --- action space ------------------------------------------------------
    max_actions: int = 64             # assert on overflow, log the distribution

    # How often each seat's registered decklist is KNOWN to its opponent, drawn
    # independently per seat at reset. This is not a rules value -- it is the
    # match context the episode is standing in. A Legend is public from setup
    # (103.1.a, 355.10.a.1) and fixes the Domain Identity (103.1.b.2), so the
    # archetype is never hidden; what this knob varies is the stronger claim of
    # the full 40, which is the Bo3 game-2 situation.
    #
    # **Randomized rather than fixed on purpose.** Trained always-known, the
    # policy never learns to read an archetype from how it is being played, and
    # is helpless in game 1 -- which is most games against a stranger. Trained
    # always-unknown, the explicit conditioning is dead weight. 0.5 exercises
    # both paths; the per-seat independence also produces the one-sided case,
    # where the opponent knows your list and you do not know theirs.
    deck_known_prob: float = 0.5
    # Ordinary movement is base <-> battlefield ONLY; lateral battlefield-to-
    # battlefield movement requires [Ganking]. Confirmed with the project owner.
    lateral_movement_needs_ganking: bool = True
    # v0 plays units only. Spells need the effect DSL, which is deliberately
    # built after combat so its hooks are derived rather than guessed.
    units_only: bool = True
    # Which runes pay a Power cost is the payer's decision whenever the domains
    # that may pay it give two or more different splits (Lilting Lullaby: Calm
    # OR Mind). Off by default so every fixture, golden and transcript recorded
    # before it plays identically; training, the UI, play and duels turn it on
    # (`real_play`). See `cost.choose_split`.
    choose_payment: bool = False
    # In v1 the engine solves damage assignment instead of exposing it to the
    # policy: the subgame is separable and reduces to an ordered exact-lethal
    # knapsack, so there is nothing to learn. PLAN.md §5.3 gotcha 8.
    # 465.2.c.2 gives the choice of which units combat damage kills to the
    # player dealing it, and `[Tank]` / `[Backline]` exist to constrain exactly
    # that choice -- so deciding it in the engine made both keywords
    # strategically inert and left the policy with no gradient toward
    # sequencing its attacks: batching was never punished, because the
    # engine's defender never sniped the unit that mattered.
    #
    # False (the default since D1) hands the choice to the player whenever
    # there IS one -- `combat.assignment_is_a_choice`, i.e. the pool cannot
    # cover every target. True pins the old behaviour, which the judge cases
    # and the combat tests rely on to stay deterministic.
    engine_solves_damage_assignment: bool = False

    # --- reward ------------------------------------------------------------
    # Terminal-only, +/-1. gamma=1.0 makes the critic a literal win-probability
    # predictor, which is the deck-evaluator deliverable (PLAN.md §1.2).
    gamma: float = 1.0
    potential_shaping: bool = False   # ablatable; off by default

    # --- match format (BACKLOG D2 / D2b) ------------------------------------
    # **The correct behaviour of two separate mechanics depends on this**, which
    # is why it is one setting rather than two flags bolted onto each.
    #
    #   battlefield presentation -- 485.5 (Duel/Bo1): each player "randomly
    #     selects" one of their three, so `deck_pool_deal`'s `rng.integers(3)` is
    #     exactly right and nothing there is broken. 486.5 (Match/Bo3): each
    #     player "selects" one, a real decision, and a battlefield used in a game
    #     somebody WON is removed for the rest of the match (486.5), while
    #     486.5.a lets it be reused if nobody won. That is what makes the three
    #     want go-first / go-second / neutral roles.
    #   sideboarding -- 403.5 forbids it before game 1, so it does not exist in
    #     Bo1 by default; 406.1.g allows it in Bo1 at the head judge's option.
    #
    # **Defaults to 3**, on the project owner's call: "Bo1 are not rare, but most
    # people play Bo3." It also makes `deck_known_prob` principled rather than a
    # coin flip, because knowing the opponent's list IS the games-2-3 condition.
    #
    # Nothing in a single episode reads this yet, and that is the design, not an
    # omission: both mechanics happen BETWEEN games, so they belong to a match
    # layer above the episode rather than to new action kinds inside one. Adding
    # action kinds would change the action-row dimension of every observation and
    # kill every trained checkpoint; a match layer cannot.
    best_of: int = 3

    @property
    def match_mode(self) -> bool:
        """True for a Match (486), False for a Duel (485). See `best_of`."""
        return self.best_of > 1

    # 406.1.g -- Bo1 sideboarding is allowed at the head judge's option, so this
    # is a real setting and not a contradiction of `best_of`.
    sideboarding: bool = True

    def at_victory_score(self, score: int) -> "Config":
        return replace(self, victory_score=score)

    def with_solved_damage(self) -> "Config":
        """Pin the pre-D1 behaviour: the ENGINE assigns combat damage.

        For scenario tests. They assert which units a given pool kills, and
        that is only a fixed answer while `solve_kills` owns the choice -- with
        the player owning it the damage step suspends and waits to be asked,
        which is the point of D1 but not what those tests are measuring.
        """
        return replace(self, engine_solves_damage_assignment=True)


DEFAULT = Config()
