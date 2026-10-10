"""Real decklists -> playable decks, with every approximation made visible.

The engine implements 17 spells and no unit text, so most real decks contain
cards it cannot fully play. There are three honest options and only one of them
is usable:

  drop them      the deck shrinks below 39 and stops being the same deck
  refuse         nothing is evaluable until the pool is complete
  **approximate** keep the size and shape, and report exactly what was faked

A deck is faked in **two different ways**, and conflating them is what made the
old coverage number wrong:

  substituted    the card is gone, replaced by a different card. Spells with no
                 DSL spec and all Gear -- the engine has nothing to resolve.
  approximated   the card is present with the right cost, domain and body, but
                 printed text or keywords the engine never reads. Every unit
                 with rules text is in here.

`coverage` counts only cards played **as printed**, so it now excludes the
second category. It previously did not: `v1_legal` accepted any card whose
keywords were in the v1 scope target and whose text ran under 90 characters,
which counted 317 of 327 unit slots as covered while executing none of their
text. Reported coverage fell from ~54% to the low 30s when this was fixed --
the decks did not get worse, the number got honest.

**A deck at 50% coverage is a proxy, not that deck**, and any evaluation of it
is a statement about the proxy. That caveat has to travel with the number,
which is why `DeckLoad` carries all three counts rather than a bare list of
card ids.
"""

from __future__ import annotations

import re

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sim"))
from engine.cards import find  # noqa: E402

from rl.config import ABILITY_KEYWORDS, DOMAINS, ENGINE_KEYWORDS  # noqa: E402
from rl.engine.cardtable import CardTable, read_decklist  # noqa: E402
from rl.engine.effects import (BF_ABILITIES, BF_STATICS,
                               BF_COST_RULES, BF_DEATH_REPLACEMENT,
                               BF_EXTRA_HIDE, BF_GRANTS_LEGEND,
                               BF_GRANTS_UNITS, BF_IGNORE_DEFLECT,
                               BF_MOVE_FROM_ANYWHERE, BF_PLAY_COSTS,
                               BF_SCORE_RULES, GRANTED_ABILITIES,
                               ABILITIES, ABILITY_BORROWERS,  # noqa: E402
                               COND_EMPOWERED, abilities_for,
                               COND_LEGION, LEGEND_ABILITIES, LEGEND_STATICS,
                               OP_EMPOWER, OP_LOOK_TOP,
                               ENTERS_READY_IF, EQUIP_ABILITIES,
                               EQUIP_STATICS, EQUIP_DEATH_REPLACEMENT,
                               PLAY_FROM_LOOK,
                               PLAY_PERMISSIONS, SPECS,
                               STATICS, TEMPORARY_SUPPRESSORS, TOKEN_DOUBLERS,
                               DEATHKNELL_DOUBLERS, PAID_COST_ENTERS_READY,
                               EQUIP_BONUS_DOUBLERS, EARLY_SCORE_TO_DRAW,
                               TIE_RECALLS_ALL, MOVED_TWICE_NO_DAMAGE,
                               PLAY_AFTER_TURN, PLAY_ONLY_CONQUERED,
                               BLOCKS_OPP_POINTS, ONE_RUNE_CHANNEL, WARDEN_LOCKS,
                               EFFECT_BONUS_DAMAGE, RUNE_WARD, IGNORE_TANK_HERE,
                               EQUIP_EFFECT_BONUS_DAMAGE, FLOW_DISCOUNTERS,
                               PLAY_COSTS_XP, PLAY_COSTS_DISCARD, ADD_COST_REDUCERS,
                               NEVER_READIED, CHOSEN_DISCOUNT, SHOWDOWN_REPEAT,
                               NONHAND_DISCOUNT, HOLD_CONQUER_SWAP, HIDDEN_LOCKS,
                               EMPOWERED_SPELL_WARD, EMPOWERED_CHOSEN_TO_MIGHT,
                               TRASH_UNIT_PLAY, MOVE_TAXERS, MULTI_BUFF,
                               EQUIP_GRANTS_TAG, NAMED_SPELL_LOCKS,
                               TEXT_COPIERS, COPY_ON_ATTACH, NONHAND_ACCELERATE,
                               REVEAL_PEEKERS, PARTIAL_TRANSCRIPTIONS)

_DOMAIN_ID = {d.lower(): i for i, d in enumerate(DOMAINS)}


@dataclass
class DeckLoad:
    name: str
    main: list[int]                 # card ids, length = printed deck size
    runes: list[int]                # domain ids
    battlefields: list[int]         # card ids, in printed order
    coverage: float                 # fraction of the MAIN DECK played as printed
    # 103.1 -- the Champion Legend, which every decklist declares and which is
    # in the Legend Zone from turn 1 (111). -1 when the list did not name one.
    legend: int = -1
    # 103.2.a's Chosen Champion. **The 40th card**: 103.2 counts it in the
    # 40-card Main Deck, 133.4 starts it in the Champion Zone instead, and that
    # is exactly why `main` is 39 -- every list in the corpus is 39 + this, and
    # the 3-copy limit holds across both zones (11 of the 30 run further copies
    # in `main`, never a 4th).
    #
    # Read by the engine since 2026-09-24: `game.new_game` places it (112) and
    # the observation carries it as public information (108.3.e). What is still
    # missing is 108.3.d, playing it from that zone -- see `new_game`.
    champion: int = -1
    # **`coverage` deliberately excludes battlefields**, and that is a
    # reporting hazard, not a convenience: every deck in the corpus prints 3,
    # and a `coverage` of 1.0 says nothing at all about them. It was written
    # when none of the 66 was scripted -- which is why `lillia-fae-fawn/v6` and
    # `v7`, two decks differing ONLY in battlefields, scored identically
    # against all six opponents. All 66 are scripted now, so `bf_coverage` is
    # 1.0 for every deck in the corpus; it stays reported because it is the
    # number that would move first if a new battlefield arrived unscripted.
    #
    # The split is kept because `coverage` is what `--min-coverage` filters
    # training decks on, and folding battlefields in would drop every deck
    # below any useful threshold at once. `bf_coverage` carries the other half
    # so that anything REPORTING a number can report an honest one.
    bf_coverage: float = 0.0
    substituted: dict = field(default_factory=dict)   # printed name -> count
    # Kept in the deck, but with printed behaviour the engine ignores. This is
    # the category the old coverage number hid: the card is *there*, its cost
    # and body are right, and the text that makes it worth playing does
    # nothing. Tracked separately from `substituted` because the two fail
    # differently -- a substitution changes the deck, an approximation changes
    # the card.
    approximated: dict = field(default_factory=dict)   # printed name -> count
    missing: list[str] = field(default_factory=list)  # names not in the table
    # 601.1.c -- the Sideboard: 0 cards, or up to 10. **Parsed but not yet used
    # by any game**, which is deliberate: sideboarding happens BETWEEN games of a
    # match (403.5 forbids it before game 1), so it belongs to a match layer
    # above the episode, not to a new action kind inside one. Reading the data
    # first is what makes that layer testable against the real corpus instead of
    # against a fixture -- 53 of the lists carry a section, and until now every
    # one of them was silently dropped on the floor, the same shape as the
    # Chosen Champion gap: parsed by the corpus, ignored by the engine.
    #
    # Verified against `data/tournament_rules.txt`, which is the authority here
    # and the ONLY place these rules appear -- Core never mentions sideboards:
    #   601.1.c.1  10 or fewer cards
    #   601.1.c.2  only valid MAIN DECK cards, so NOT battlefields, and 403.4.b
    #              forbids changing Runes, Legend or Battlefields at all
    #   403.4      exchanged 1-for-1 with main-deck cards
    #   403.5      never before game 1
    #   403.10     no sideboarding after a draw
    #   403.4.a / 601.1.c.4  the Chosen Champion MAY be re-designated to one from
    #              the sideboard or the main deck -- a re-designation inside the
    #              registered 40 (103.2), not a 41st card
    # 103.2.d copy limits span deck AND sideboard combined, which is why
    # `sideboard_legal` counts them together rather than per zone.
    # **An ABSENT `Sideboard:` section is a legal empty sideboard, not a parse
    # failure.** 601.1.c.1 is "10 or fewer", and zero is a real, legal choice --
    # the player registered no sideboard and therefore did not sideboard. Nothing
    # here should be "fixed" for the 19 lists in the corpus that have none, and
    # `sideboard_legal([])` returns no notes for exactly that reason.
    sideboard: list[int] = field(default_factory=list)   # card ids; [] is legal
    sideboard_notes: list[str] = field(default_factory=list)  # why it is illegal

    def report(self) -> str:
        def top(d):
            return ", ".join(f"{n}x{c}" for n, c in
                             sorted(d.items(), key=lambda t: -t[1])[:4])
        # `bf` travels next to `coverage` because the two are different
        # questions and only one of them is in `coverage`: a deck can read 100%
        # covered with all three of its battlefields doing nothing. Anything
        # quoting one number has to quote the other, which is why it is here in
        # `report` and not only in `deckeval`'s table.
        return (f"{self.name}: {len(self.main)} cards, coverage "
                f"{self.coverage:.0%}, bf {self.bf_coverage:.0%}"
                + (f", substituted {sum(self.substituted.values())} "
                   f"({top(self.substituted)})" if self.substituted else "")
                + (f", approximated {sum(self.approximated.values())} "
                   f"({top(self.approximated)})" if self.approximated else "")
                + (f", sideboard {len(self.sideboard)}" if self.sideboard
                   else ", no sideboard (legal)")
                + (f", SIDEBOARD ILLEGAL {self.sideboard_notes}"
                   if self.sideboard_notes else "")
                + (f", MISSING {self.missing}" if self.missing else ""))


def sideboard_legal(table: CardTable, main: list[int], side: list[int],
                    champion: int = -1) -> list[str]:
    """Why this sideboard is illegal, or `[]`.

    **`data/tournament_rules.txt` is the authority and the only source** -- the
    Core rules never mention sideboards, and
    `reference/riftbound-deckbuilding-tips.txt` is wrong on two counts (it says
    battlefields are sideboardable and that Bo1 sideboarding is not allowed).

    Checks, each with the rule it comes from:

    * **601.1.c.1** -- 10 or fewer cards. **Zero is legal**: a list with no
      `Sideboard:` section registered no sideboard and simply does not
      sideboard, which is a legitimate choice and not a missing field. `[]`
      therefore returns no notes.
    * **601.1.c.2** -- only valid Main Deck cards. Battlefields, Runes and
      Legends are not, and 403.4.b forbids changing them at all, so one in a
      sideboard is a parse or list error rather than a strategic choice.
    * **103.2.d** -- the 3-copy limit spans deck **and** sideboard *combined*.
      Counting per zone would pass a list running three in the main and three
      more in the side, which is six.

    The Chosen Champion is counted in, because 103.2 puts it inside the 40 and
    403.4.a lets it be re-designated to a card from either zone -- so a list with
    three copies in `main` plus the champion is already at the limit.
    """
    notes: list[str] = []
    if len(side) > 10:
        notes.append(f"601.1.c.1 -- {len(side)} cards, the limit is 10")
    for cid in set(side):
        for kind in ("Battlefield", "Rune", "Legend"):
            if table.is_type(cid, kind):
                notes.append(f"601.1.c.2 -- {table.names[cid]} is a {kind}, "
                             f"not a valid Main Deck card")
    counted: dict[int, int] = {}
    for cid in list(main) + list(side) + ([champion] if champion >= 0 else []):
        counted[cid] = counted.get(cid, 0) + 1
    for cid, n in counted.items():
        # 103.2.d.1 exempts Runes from the copy limit; nothing else is exempt.
        if n > 3 and not table.is_type(cid, "Rune"):
            notes.append(f"103.2.d -- {n} copies of {table.names[cid]} across "
                         f"deck and sideboard, the limit is 3")
    return notes


def includable(table: CardTable, cid: int) -> bool:
    """Can this card sit in a deck without the engine choking on it?

    A **unit** always can. Every unit is a body with a cost, a domain and a
    Might, and the engine plays those correctly whether or not it executes the
    card's text -- so keeping Lillia as a vanilla 3-Might Calm body is strictly
    closer to the real deck than swapping her for a different 3-drop. Tokens
    are the exception: 185.3 makes them non-cards that only effects create.

    A **spell** cannot. Without a DSL spec there is nothing to resolve, so it
    has to be substituted. Gear likewise, until Gear exists at all.
    """
    if table.is_type(cid, "Unit"):
        return not table.is_token(cid)
    if table.is_type(cid, "Spell"):
        return table.names[cid] in SPECS
    if table.is_type(cid, "Gear"):
        # Gear is a permanent the engine can now play (149.2, base only), so a
        # gear whose text is transcribed goes in as itself. One without a spec
        # would sit on the board doing nothing, which is a worse lie than a
        # substitution, so it still has to be swapped out.
        #
        # An EQUIPMENT is the third case and needs neither registry: 818.1.c.2
        # makes [Equip] an ability in its own right and `_equip_abilities`
        # synthesises it from the keyword, exactly as [Hunt] is synthesised. So
        # the honest test is whether the engine has anything at all to do with
        # the card, which `abilities_for` answers for all three sources at once.
        # This stays a strictly lower bar than `plays_as_printed`: a gear whose
        # extra text is untranscribed is playable without being correct.
        return (table.names[cid] in ABILITIES or table.names[cid] in STATICS
                or table.names[cid] in TIE_RECALLS_ALL
                or bool(abilities_for(table, cid)))
    return False


def _encodes_legion(name: str) -> bool:
    """Does this card's transcription actually carry the [Legion] gate?"""
    for ab in ABILITIES.get(name, ()):
        if ab.cond == COND_LEGION or any(op.cond == COND_LEGION for op in ab.ops):
            return True
    return any(st.cond == COND_LEGION for st in STATICS.get(name, ()))


def _encodes_predict(name: str) -> bool:
    """Does this card's transcription actually perform the [Predict]?

    Credited the same way [Legion] is, and NOT through `ABILITY_KEYWORDS`.
    Blanket-crediting a keyword to any card that has a spec is how Hunt and
    Vision were once forgiven on cards that never implemented them: the spec
    covers the rest of the text and the keyword silently rides along. 436.1
    makes Predicting a real action -- look at the top card, choose whether to
    Recycle it -- so the only honest test is whether an OP_LOOK_TOP is there.
    """
    for entry in list(ABILITIES.get(name, ())) + (
            [SPECS[name]] if name in SPECS else []):
        if any(op.op == OP_LOOK_TOP for op in entry.ops):
            return True
    return False


# The sentence a card spends on its own 806.3 exception, which
# `actions.play_destinations` executes -- stripped from the residual so the
# clause is not counted as outstanding behaviour.
#
# "I can AMBUSH to ..." is the errata's wording for Rengar, Trophy Hunter and
# means the same thing: 822.1.d says Ambush also appears as a verb, and "in
# such a case the verb is taken to mean 'play with the permissions of the
# Ambush keyword'" -- with Rengar as the rulebook's own worked example. That is
# exactly what `play_destinations` does with PERM_ENEMY plus the `ambush_only`
# timing, so the alternative verb belongs here and not in a second mechanism.
_PLAY_PERM_CLAUSE = re.compile(
    r"(?:You may play me|I can be played|I can Ambush) to an? [^.]*\.?", re.I)


# **Every registry a BATTLEFIELD's text can legitimately live in.**
#
# `BF_STATICS`/`BF_ABILITIES` are only two of eleven. A clause about a PAYMENT,
# about the Victory Score, about where a unit may move from, or about an
# ability granted to somebody else's row is not expressible as a static or a
# triggered ability, so it is expressed in its own table and read from its own
# call site -- which is why 15 fully-scripted battlefields read as uncovered
# while the guard consulted two tables and the engine consulted eleven.
#
# Each entry here is a table the ENGINE itself reads, verified by its call
# site, not a list of forgiven names:
#
#   BF_COST_RULES         cost.bf_discount / cost.bf_tax
#   BF_PLAY_COSTS         actions (a paid destination -- Dragon Roost)
#   BF_SCORE_RULES        game.victory_score / combat (no_score_before_turn)
#   BF_IGNORE_DEFLECT     resolve.deflect_cost
#   BF_DEATH_REPLACEMENT  combat.altar_offer
#   BF_EXTRA_HIDE         chain (how many facedown slots the ground holds)
#   BF_MOVE_FROM_ANYWHERE combat.moves_from_anywhere (the Baron Pit, 187.9)
#   BF_GRANTS_UNITS       effects, with the text in `GRANTED_ABILITIES`
#   BF_GRANTS_LEGEND      effects, likewise
#
# `GRANTED_ABILITIES` is listed as well because that is where the granted text
# actually is: the two `BF_GRANTS_*` sets only say WHO receives it.
_BF_REGISTRIES = (BF_STATICS, BF_ABILITIES, BF_COST_RULES, BF_PLAY_COSTS,
                  BF_SCORE_RULES, BF_IGNORE_DEFLECT, BF_DEATH_REPLACEMENT,
                  BF_EXTRA_HIDE, BF_MOVE_FROM_ANYWHERE, BF_GRANTS_UNITS,
                  BF_GRANTS_LEGEND, GRANTED_ABILITIES)

# A `BF_COST_RULES` entry names the KEYWORD whose cost it modifies, and a
# battlefield that discounts "[Empower] costs of your units here" prints that
# keyword without having it. `unread_keywords` cannot tell the two apart --
# the clause leads the card's text, so it parses as the card's own keyword --
# and the discount is exactly what reads it. So the keyword is credited from
# the rule's `kind`, which is the engine's own name for the same thing, rather
# than from a hand-kept exception for Risen Altar.
_COST_RULE_KEYWORD = {"repeat": "Repeat", "empower": "Empower",
                      "reaction": "Reaction"}


def _bf_unread(table: CardTable, cid: int) -> list[str]:
    """`unread_keywords` for a battlefield, minus the ones its rule reads."""
    unread = set(table.unread_keywords(cid))
    name = table.names[cid]
    kw = _COST_RULE_KEYWORD.get(BF_COST_RULES.get(name, {}).get("kind"))
    if kw:
        unread.discard(kw)
    return sorted(unread)


def _encodes_empower(name: str, abilities=ABILITIES) -> bool:
    """Does this card's transcription carry the [Empower] activated ability?

    827.1.c.1 makes [Empower Cost] short for "[Cost]: Empower this", and the
    cost is different on every card -- {2 energy}, "{1 energy} or {Body rune}",
    "Kill a friendly unit". So the ENGINE implements the mechanism (the status,
    the once-only gate, the dependent-ability condition) while each card still
    has to state its own cost. That makes it a Deathknell-shaped keyword, not a
    Deflect-shaped one, and blanket-crediting it in `ENGINE_KEYWORDS` would
    pass ~35 cards that never transcribed a cost.

    `abilities` is the table to read, because a LEGEND's abilities are in
    `LEGEND_ABILITIES` and not in `ABILITIES`. Akali - Rogue Assassin and
    Jayce - Defender of Tomorrow both transcribe their [Empower] cost there,
    and read as uncovered for as long as this function looked in one table.
    """
    return any(op.op == OP_EMPOWER
               for ab in abilities.get(name, ()) for op in ab.ops)


def _encodes_empowered(name: str, abilities=ABILITIES, statics=STATICS) -> bool:
    """Does it carry the [Empowered] dependent ability (828.1.b.1)?"""
    return any(st.cond == COND_EMPOWERED for st in statics.get(name, ())) or \
        any(op.cond == COND_EMPOWERED
            for ab in abilities.get(name, ()) for op in ab.ops) or \
        any(ab.while_empowered for ab in abilities.get(name, ()))


_ATTACHED_BAND = re.compile(r"Attached:\s*[+-]?\d+\s*Might\.?\s*(.*)$", re.S)


def _grants_attached_ability(table: CardTable, cid: int) -> bool:
    """Does this Equipment's "Attached:" band carry more than a Might Bonus?

    718.3 appends an Attached card's Effect Text to its Top-Most Card, which is
    not implemented -- so a card whose band is nothing but "+N Might" IS played
    as printed, and one with a clause after it is not. Reading the band rather
    than keeping a list means the answer follows the data.
    """
    m = _ATTACHED_BAND.search(table.raw_text[cid] or "")
    rest = m.group(1).strip(" .") if m else ""
    if not rest:
        return False
    # 718.3's KEYWORD half is implemented: `cardtable.attached_keywords` parses
    # the band and `combat.perm_kw` hands the keyword to the Top-Most card. A
    # band that is nothing but keywords is therefore fully executed -- provided
    # the engine actually reads every one of them, which is the same honesty
    # test `unread_keywords` applies to a card's own keywords.
    granted = table.attached_kw[cid]
    if granted:
        return any(kw not in ENGINE_KEYWORDS for kw, _ in granted)
    # ...and 718.3's ABILITY half is implemented for anything transcribed in
    # `EQUIP_ABILITIES`, which fires from the unit's row with "I" reading as
    # the unit (136.2.c). Membership asserts the whole band is transcribed,
    # exactly as membership in `ABILITIES` asserts it for a card's own text.
    return (table.names[cid] not in EQUIP_ABILITIES
            and table.names[cid] not in EQUIP_STATICS
            and table.names[cid] not in EQUIP_DEATH_REPLACEMENT
            and table.names[cid] not in EQUIP_EFFECT_BONUS_DAMAGE
            and table.names[cid] not in HOLD_CONQUER_SWAP
            and table.names[cid] not in EQUIP_GRANTS_TAG
            and table.names[cid] not in TEXT_COPIERS
            and table.names[cid] not in COPY_ON_ATTACH)


def plays_as_printed(table: CardTable, cid: int) -> bool:
    """Does the engine execute **everything** this card says?

    The honest coverage question, and it is stricter than `includable` by a
    long way. This used to be `table.v1_legal(cid)` -- keywords in the v1 scope
    target plus text under 90 characters -- which counted 317 of 327 unit slots
    as covered while ignoring their rules text entirely. "When you play me,
    draw 1." is 25 characters. So is most of what makes a unit worth playing.

    Two exact checks replace the length heuristic:

      residual_text    anything printed beyond keywords needs a DSL spec
      unread_keywords  a keyword nothing consults is not being played
    """
    # A BATTLEFIELD is never `includable` -- it does not go in the main deck --
    # so it is judged on its own terms before that check. It counts as played
    # only when its text is transcribed, in ANY of the registries the engine
    # reads a ground rule from (`_BF_REGISTRIES`) -- not just the two ability
    # tables. Every one of the 66 battlefields prints rules text, so none is
    # covered by default.
    if table.is_type(cid, "Battlefield"):
        name = table.names[cid]
        if not any(name in reg for reg in _BF_REGISTRIES):
            return False
        return not _bf_unread(table, cid)
    # A LEGEND is not includable either, and for the same reason: 107.4.b puts
    # it in its own zone, never the Main Deck, so `includable` rejects it and
    # every legend read as uncovered -- including the three that are fully
    # transcribed. Lillia - Bashful Bloom is the most-played legend in the
    # corpus and scored zero.
    #
    # Judged like a battlefield rather than like a unit, because a legend has
    # no body to fall back on: it is never played, never attacks and has no
    # Might, so unlike a unit there is no sense in which an untranscribed one
    # is still "mostly right". Its abilities are the whole card.
    if table.is_type(cid, "Legend"):
        name = table.names[cid]
        if name not in LEGEND_STATICS and name not in LEGEND_ABILITIES:
            return False
        unread = set(table.unread_keywords(cid))
        # The same credit the UNIT branch below gives, against the legend's own
        # tables: a keyword that IS an ability is implemented by transcribing
        # that ability. Without this the branch held legends to a stricter
        # standard than units for no stated reason, and two fully-transcribed
        # legends failed on an [Empower] cost they both carry.
        if name in LEGEND_ABILITIES:
            unread -= set(ABILITY_KEYWORDS)
        if "Empower" in unread and _encodes_empower(name, LEGEND_ABILITIES):
            unread.discard("Empower")
        if "Empowered" in unread and _encodes_empowered(
                name, LEGEND_ABILITIES, LEGEND_STATICS):
            unread.discard("Empowered")
        return not unread
    if not includable(table, cid):
        return False
    if table.names[cid] in PARTIAL_TRANSCRIPTIONS:
        return False
    # **EQUIPMENT IS WITHHELD while it grants an ABILITY, and only then.**
    # The Might Bonus is read: 137.3's number lives in the "Attached:" band
    # that `data/errata.json` supplies and `cardtable.might_bonus` parses, so
    # Long Sword really does give its unit +2 Might, and [Equip]/[Quick-Draw]
    # are honest entries in `ENGINE_KEYWORDS`.
    #
    # What is not read is the other half of that band. 136.2/718.3 append an
    # Attached card's Effect Text to its Top-Most Card's Rules Text -- "When I
    # hold, score 1 point" (Trinity Force), "+2 Might while I'm an attacker"
    # (Serrated Dirk) -- and that is a mechanism, not a lookup: the ability has
    # to fire from a row that is not the one it is printed on. Until it does, an
    # Equipment that prints one attaches, grants its Might, and silently drops
    # the clause that made it worth playing.
    #
    # So the guard asks the band card by card and clears itself as 718.3 lands,
    # rather than needing a list maintained by hand.
    if "Equipment" in table.tags[cid] and _grants_attached_ability(table, cid):
        return False
    if table.is_type(cid, "Spell"):
        return True                       # a spec transcribes the whole text
    name = table.names[cid]
    unread = set(table.unread_keywords(cid))
    if name in ABILITIES:
        # 808.1 -- [Deathknell] IS the ability, not a property alongside it, so
        # transcribing the ability is what implements the keyword.
        unread -= set(ABILITY_KEYWORDS)
    # [Legion] is a GATE on an ability or static rather than an ability of its
    # own, so it is credited only when the transcription actually carries the
    # gate. Blanket-crediting it the way Deathknell is credited would pass any
    # card that merely has a spec, including one that transcribed the effect
    # and quietly dropped the "if you've played another card this turn".
    if "Legion" in unread and (_encodes_legion(name) or name in TRASH_UNIT_PLAY):
        unread.discard("Legion")
    if "Predict" in unread and _encodes_predict(name):
        unread.discard("Predict")
    if "Empower" in unread and _encodes_empower(name):
        unread.discard("Empower")
    if "Empowered" in unread and (_encodes_empowered(name)
                                  or name in EMPOWERED_SPELL_WARD
                                  or name in EMPOWERED_CHOSEN_TO_MIGHT):
        unread.discard("Empowered")
    if unread:
        return False
    # Presence in ABILITIES or STATICS means the same thing presence in SPECS
    # does: the card's whole text is transcribed. Cards with one implemented
    # ability and one unimplemented one (Scuttle Crab: an ETB draw and a
    # Deathknell) are deliberately absent, so this stays an allowlist rather
    # than a guess.
    # A printed play-destination permission (806.3's exceptions) is executed
    # by `actions.play_destinations`, so it counts as implemented -- but it
    # only ever covers its OWN sentence. Ocean Drake says "You may play me
    # to an open battlefield" AND "you may return a non-Dragon unit to its
    # owner's hand"; forgiving the whole residual because the first half is
    # handled would credit a card whose second half does nothing. So strip
    # just that clause and let whatever is left be judged normally.
    residual = table.residual_text(cid)
    if name in PLAY_PERMISSIONS:
        residual = _PLAY_PERM_CLAUSE.sub("", residual).strip()
    # `TOKEN_DOUBLERS` sits alongside ABILITIES/STATICS for the same reason:
    # membership asserts that the card's whole text is the thing the engine
    # implements. That is true of Zilean, whose text is nothing but the
    # replacement clause. A future doubler with extra text would need an
    # ABILITIES entry for that text, exactly as any other card does.
    if residual and not (name in ABILITIES or name in STATICS
                         or name in SPECS
                         # An Equipment's "Attached:" band is its whole
                         # residual, and `EQUIP_ABILITIES` transcribing it means
                         # the same thing an ABILITIES entry means for a card's
                         # own text. The guard above has already refused any
                         # band that is NOT transcribed.
                         or name in EQUIP_ABILITIES
                         or name in EQUIP_STATICS
                         or name in EQUIP_DEATH_REPLACEMENT
                         # Nocturne's whole residual IS the permission -- an
                         # extra legal action while a look is suspended, which
                         # `actions.legal_actions` offers and
                         # `_resolve_play_from_look` carries out. A plain
                         # membership test rather than a clause strip like
                         # PLAY_PERMISSIONS above, because there is no second
                         # sentence left over to judge.
                         or name in PLAY_FROM_LOOK
                         or name in TOKEN_DOUBLERS
                         or name in DEATHKNELL_DOUBLERS
                         or name in NONHAND_DISCOUNT
                         or name in HIDDEN_LOCKS
                         or name in TRASH_UNIT_PLAY
                         or name in MOVE_TAXERS
                         or name in EQUIP_GRANTS_TAG
                         or name in TEXT_COPIERS
                         or name in NONHAND_ACCELERATE
                         or name in REVEAL_PEEKERS
                         or name in COPY_ON_ATTACH
                         or name in HOLD_CONQUER_SWAP
                         or name in PAID_COST_ENTERS_READY
                         or name in EQUIP_BONUS_DOUBLERS
                         or name in EARLY_SCORE_TO_DRAW
                         or name in TIE_RECALLS_ALL
                         or name in MOVED_TWICE_NO_DAMAGE
                         or name in PLAY_AFTER_TURN
                         or name in PLAY_ONLY_CONQUERED
                         or name in BLOCKS_OPP_POINTS
                         or name in ONE_RUNE_CHANNEL
                         or name in WARDEN_LOCKS
                         or name in EFFECT_BONUS_DAMAGE
                         or name in RUNE_WARD
                         or name in IGNORE_TANK_HERE
                         or name in EQUIP_EFFECT_BONUS_DAMAGE
                         or name in FLOW_DISCOUNTERS
                         or name in PLAY_COSTS_XP
                         or name in PLAY_COSTS_DISCARD
                         or name in ADD_COST_REDUCERS
                         or name in NEVER_READIED
                         or name in CHOSEN_DISCOUNT
                         or name in SHOWDOWN_REPEAT
                         or name in TEMPORARY_SUPPRESSORS
                         or name in ENTERS_READY_IF
                         or name in ABILITY_BORROWERS):
        return False
    # A card that creates a token it cannot play correctly is approximating,
    # exactly as a card whose own text is ignored would be. The Bird token
    # carries [Deflect] and the Gold token an activated ability with a "Kill
    # this" cost, neither of which the engine reads -- so Carrion Dredger
    # "playing a Bird" really plays a vanilla 1-Might body.
    return all(_token_ok(table, op.token)
               for spec in list(ABILITIES.get(name, ())) + ([SPECS[name]]
                                                            if name in SPECS else [])
               for op in spec.ops if op.token)


def _token_ok(table: CardTable, token_name: str) -> bool:
    """Is the token this op creates itself played as printed?"""
    try:
        tid = table.id_of(token_name)
    except KeyError:
        return False
    if table.unread_keywords(tid):
        return False
    return (not table.residual_text(tid)
            or token_name in ABILITIES or token_name in STATICS)


def _pool(table: CardTable) -> tuple[list[int], list[int]]:
    """Substitution pools -- what a missing card may be replaced *with*.

    Deliberately `plays_as_printed`, not `includable`: a substitute is already
    an approximation, and picking one whose own text is ignored would stack a
    second silent approximation on top of the first.
    """
    units = [c for c in range(table.n)
             if table.is_type(c, "Unit") and plays_as_printed(table, c)]
    spells = [c for c in range(table.n)
              if table.is_type(c, "Spell") and plays_as_printed(table, c)]
    return units, spells


def _closest(table: CardTable, pool: list[int], want: int) -> int:
    """Nearest implemented card by domain first, then cost.

    Domain leads because the rune deck is what gives a deck its identity: an
    off-colour substitute is often uncastable and dilutes exactly the archetype
    signal the substitution is supposed to preserve. Cost breaks ties so the
    curve still survives.
    """
    e, p = int(table.energy[want]), int(table.power[want])
    mask = int(table.domain_mask[want])
    return min(pool, key=lambda c: (
        0 if (int(table.domain_mask[c]) & mask) else 1,      # shares a domain
        abs(int(table.energy[c]) - e) + abs(int(table.power[c]) - p),
        int(table.energy[c]), c))


# A main deck is 40 cards (39 + the Champion in every list in the corpus).
# Anything far outside that is a parse failure rather than a deck -- a flat
# export with no section headers puts battlefields and runes in the main list
# too, which produced a "131-card deck" that crashed a fuzz 1,500 games in
# rather than failing at load.
LEGAL_MAIN = (30, 45)


def _split_flat(parsed: dict, table: CardTable) -> dict:
    """Route a headerless export into sections by CARD TYPE.

    Some exports are a bare list -- "3 Petal Pixie [UNL] 76" -- with no
    `MainDeck:` header, so everything lands in one bucket including the
    battlefields and the rune pool. The card table already knows what each one
    is, so classify rather than guess.
    """
    if any(k in parsed for k in ("Battlefields", "Runes")):
        return parsed
    main, bfs, runes = [], [], []
    for count, name in parsed.get("MainDeck", []):
        if name.lower().endswith("rune"):
            runes.append((count, name))
            continue
        card = find(name)
        cid = table._index.get(card.name) if card else None
        if cid is not None and table.is_type(cid, "Battlefield"):
            bfs.append((count, name))
        else:
            main.append((count, name))
    return {**parsed, "MainDeck": main, "Battlefields": bfs, "Runes": runes}


def load_deck(path: Path, table: CardTable) -> DeckLoad:
    """Read a decklist and make it playable, recording every substitution."""
    parsed = _split_flat(read_decklist(Path(path)), table)
    units, spells = _pool(table)

    main: list[int] = []
    subs: dict[str, int] = {}
    approx: dict[str, int] = {}
    missing: list[str] = []
    as_printed = 0

    for count, name in parsed.get("MainDeck", []):
        card = find(name)
        cid = table._index.get(card.name) if card else None
        if cid is None:
            missing.append(name)
            main.extend([_closest(table, units, units[0])] * count)
            subs[name] = subs.get(name, 0) + count
            continue
        if includable(table, cid):
            # The card itself goes in either way; the only question is whether
            # it counts as covered.
            main.extend([cid] * count)
            if plays_as_printed(table, cid):
                as_printed += count
            else:
                approx[name] = approx.get(name, 0) + count
            continue
        pool = spells if table.is_type(cid, "Spell") else units
        # Gear has no implementation at all yet, so it becomes a unit -- the
        # least wrong option, and counted as a substitution either way.
        main.extend([_closest(table, pool or units, cid)] * count)
        subs[name] = subs.get(name, 0) + count

    runes: list[int] = []
    for count, name in parsed.get("Runes", []):
        dom = name.lower().replace(" rune", "").strip()
        if dom in _DOMAIN_ID:
            runes.extend([_DOMAIN_ID[dom]] * count)
    if not runes:
        runes = [0] * 12

    bfs: list[int] = []
    for count, name in parsed.get("Battlefields", []):
        card = find(name)
        cid = table._index.get(card.name) if card else None
        if cid is not None:
            bfs.extend([cid] * count)

    def _one(section: str) -> int:
        """The single card named in a one-line section, or -1."""
        for _count, nm in parsed.get(section, []):
            card = find(nm)
            cid = table._index.get(card.name) if card else None
            if cid is not None:
                return int(cid)
        return -1

    # 601.1.c -- the Sideboard. Unknown names are reported rather than
    # substituted: a sideboard card the engine cannot represent must not quietly
    # become a different card, because the whole point of the zone is choosing
    # exactly which card comes in.
    side: list[int] = []
    side_notes: list[str] = []
    for count, name in parsed.get("Sideboard", []):
        card = find(name)
        cid = table._index.get(card.name) if card else None
        if cid is None:
            side_notes.append(f"unknown card {name!r}")
            continue
        side.extend([int(cid)] * count)
    side_notes += sideboard_legal(table, main, side, champion=_one("Champion"))

    total = len(main) or 1
    bf_ok = sum(1 for c in bfs if plays_as_printed(table, c))
    return DeckLoad(name=Path(path).parent.name + "/" + Path(path).stem,
                    main=main, runes=runes, battlefields=bfs,
                    coverage=as_printed / total,
                    bf_coverage=bf_ok / (len(bfs) or 1),
                    legend=_one("Legend"), champion=_one("Champion"),
                    substituted=subs,
                    approximated=approx, missing=missing,
                    sideboard=side, sideboard_notes=side_notes)


_VERSION = re.compile(r"v(\d+)")


def decklist_files(root: Path | None = None,
                   latest_only: bool = False,
                   meta_only: bool = False) -> list[Path]:
    """The decklist files under `root`, optionally one per DECK.

    `decks/` holds two different things under one glob, and they must not be
    weighted alike. `decks/meta/` is a flat folder of distinct netdecks, one
    file each. Every other folder is ONE deck plus its iteration history:
    `lillia-fae-fawn-blind/` alone holds v1 through v8.

    So a plain `rglob("*.txt")` counts that single deck eight times. It is the
    right reading for `deckeval`, whose entire job is comparing v7 against v8 --
    and the wrong one everywhere else, where it silently multiplies whatever
    the repo owner happened to iterate on most. `latest_only` keeps just the
    newest version of each personal deck; the meta folder is always kept whole.
    """
    root = Path(root or (ROOT / "decks"))
    # `decks/banned/` holds real tournament lists that are ILLEGAL under the
    # current banlist -- kept for reference and meta analysis, never trained on.
    # `decks/upcoming/` holds lists for a set that is not out yet.
    # Excluded here rather than at the call sites: `rglob` would otherwise sweep
    # them in silently, and a policy trained against a banned card is learning a
    # game nobody is allowed to play. `cli.py check` is the other half of this.
    # `decks/upcoming/` is the same idea one set earlier: lists built from cards
    # that are not legal yet. A deck there must never be trained on even while
    # `RIFTBOUND_UPCOMING` is set, because the flag exists to let the ENGINE see
    # the cards for scripting, not to let the agent practise an illegal format.
    EXCLUDE = {"banned", "upcoming"}
    def _keep(f: Path) -> bool:
        return not (set(f.relative_to(root).parts[:-1]) & EXCLUDE)
    meta = sorted(f for f in (root / "meta").glob("*.txt") if _keep(f))
    if meta_only:
        return meta
    if not latest_only:
        return sorted(f for f in root.rglob("*.txt") if _keep(f))
    by_folder: dict[Path, list[Path]] = {}
    for f in sorted(root.rglob("*.txt")):
        # `_keep` is needed here too, not just above: a banned PERSONAL deck sits
        # at decks/banned/<deck>/vN.txt, whose parent is the deck name rather
        # than "meta", so the skip below would let it straight through.
        if f.parent.name == "meta" or not _keep(f):
            continue
        by_folder.setdefault(f.parent, []).append(f)
    # An unversioned filename sorts to -1, so a folder holding exactly one
    # unversioned list still yields that list rather than nothing.
    latest = [max(fs, key=lambda f: (int(m.group(1))
                                     if (m := _VERSION.search(f.stem)) else -1))
              for fs in by_folder.values()]
    return meta + sorted(latest)


def load_all(table: CardTable, root: Path | None = None,
             warn: bool = True, latest_only: bool = False) -> list[DeckLoad]:
    """Every decklist under `root` that parses to a plausible deck.

    A file that parses to a main deck far off 40 cards is rejected here, and
    loudly. It used to be admitted on `len(main) >= 10` and would then blow up
    deep inside a training run on `MAX_DECK` -- a parse failure surfacing as an
    engine crash thousands of games later, with nothing pointing at the file.

    `latest_only` collapses each personal folder to its newest version -- see
    `decklist_files`. Off by default because `deckeval` compares versions and
    must see them all; the TRAINING pool wants it on, or one deck's iteration
    history becomes an archetype prior.
    """
    root = root or (ROOT / "decks")
    out, bad, short = [], [], []
    for f in decklist_files(root, latest_only=latest_only):
        d = load_deck(f, table)
        if not LEGAL_MAIN[0] <= len(d.main) <= LEGAL_MAIN[1]:
            if len(d.main) >= 10:
                bad.append((d.name, len(d.main)))
            continue
        # **103.2 counts the Chosen Champion inside the 40**, and 133.4 starts it
        # in the Champion Zone, so `main` is 39 exactly when a champion was
        # registered and 40 when none was. Anything else means cards went
        # MISSING as the file was parsed, and the usual reason is the release
        # gate: a list built on an unreleased set resolves none of those names,
        # so it loads as a short deck with no champion and no legend rather than
        # failing. Admitting it puts a deck a card light into training with a
        # blanked archetype signal -- found when `decks/seraphine-not-alone/`
        # (all RAD cards) turned up in the pool at 39 + no champion.
        #
        # `decks/upcoming/` is the place for such a list; this is the backstop
        # for one that is not in it. Rejected here rather than at the call sites
        # for the same reason `decks/banned/` is: `rglob` sweeps silently.
        if len(d.main) + (1 if d.champion >= 0 else 0) != 40:
            short.append((d.name, len(d.main), d.champion >= 0))
            continue
        out.append(d)
    if warn and (bad or short):
        import sys as _sys
        if bad:
            print(f"  skipped {len(bad)} unparseable decklist(s): "
                  + ", ".join(f"{n} ({k} cards)" for n, k in bad),
                  file=_sys.stderr)
        if short:
            print(f"  skipped {len(short)} decklist(s) that did not resolve to "
                  f"40 registered cards -- cards the release gate hides, or a "
                  f"typo'd name: "
                  + ", ".join(f"{n} ({k} main{'' if c else ', no champion'})"
                              for n, k, c in short),
                  file=_sys.stderr)
    return out


def present(deck: DeckLoad, pick: int = 0,
            fallback: DeckLoad | None = None) -> int:
    """The battlefield `deck` presents, by 486.5: ONE of its own three.

    `pick` indexes the deck's printed battlefield list and wraps, so a caller
    sweeping 0..2 works on a deck that printed fewer than three. A deck that
    printed none borrows from `fallback` -- an incomplete decklist should not
    block a matchup -- and returns -1 if there is nothing anywhere.
    """
    pool = deck.battlefields or (fallback.battlefields if fallback else [])
    return pool[pick % len(pool)] if pool else -1


def matchup(a: DeckLoad, b: DeckLoad, rune_size: int = 12,
            n_bf: int = 2, picks: tuple[int, int] = (0, 0)
            ) -> tuple[list, list, list, list, list]:
    """Two DeckLoads as `game.new_game` arguments.

    Five elements: decks, rune decks, battlefields, legends, champions.

    486.5: **each player selects one of their OWN three battlefields**, and the
    two are placed simultaneously. With N_BF=2 that is exactly one from each
    deck, in seat order -- `bfs[0]` is what seat 0 brought.

    `picks` is which of its three each deck presents. It is a parameter rather
    than a fixed index because choosing is the decision the format is built
    around ([[riftbound-builds-for-bo3]]): a caller that wants to know which
    battlefield a deck should bring sweeps the 3x3 grid, and one that does not
    care gets each deck's first.

    This used to read deck A's FIRST battlefield and deck B's SECOND from a
    pool that fell back to whichever deck had any -- two bugs in one line. A
    deck with no battlefields listed supplied BOTH, and, worse, the pair
    depended on argument order: `matchup(a, b)` gave `[a[0], b[1]]` while
    `matchup(b, a)` gave `[b[0], a[1]]`. Every evaluator here averages over a
    seat swap to cancel the first-player advantage, so each half of that
    average was played on a DIFFERENT pair of battlefields. Presenting per deck
    makes the swap change only the seats, which is what it is for.
    """
    decks = [list(a.main), list(b.main)]
    runes = [(a.runes * 3)[:rune_size], (b.runes * 3)[:rune_size]]
    bfs = [present(a, picks[0], b), present(b, picks[1], a)][:n_bf]
    legends = [a.legend, b.legend]
    # 112/133.4 -- the Chosen Champion starts in the Champion Zone, so it is a
    # FIFTH element rather than a 40th card in `decks`. Callers that only want
    # the board setup unpack four and ignore it.
    champions = [a.champion, b.champion]
    return decks, runes, bfs, legends, champions
