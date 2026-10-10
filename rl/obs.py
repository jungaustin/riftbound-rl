"""Observation encoding -- PLAN.md §5.2.

Three rules, in priority order.

**1. The viewer is the spec.** `viewer.view(state, table, cfg, seat)` defines
what a seat may legitimately see. This module encodes that and nothing more:
own hand, both trashes as counts, deck and opponent-hand counts but never
contents, and the Facedown Zone as a public slot with a private card (107.3.f).
`privileged` is the one deliberate exception -- it is critic-only and training-
only (asymmetric actor-critic, §6.4), and it is a separate field precisely so
that "did this leak?" is answerable by looking at which field a number is in.

**2. Canonicalize.** Everything is encoded from the acting seat's point of view:
"mine" and "theirs", "own base" and "enemy base", never "seat 0" and "seat 1".
One network then serves both seats. `rl/engine/mirror.py` exists to test this.

**3. Attributes, never card ids.** A card is its stat line, type, domains and
keywords. No id one-hots, so a card the net never saw in training still gets a
sensible embedding from its features -- which is the entire reason this can
become a deck evaluator rather than a bot for one decklist (§1.1).

Battlefields are *not* mirrored between seats: both are public ground and are
told apart by their card features, not by whose they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl.config import DOMAINS, Config
from rl.engine import actions as A
from rl.engine import chain, combat
from rl.engine.cardtable import CardTable
MAX_MODES = 4    # widest "Choose one --" in the pool; asserted in effects
# 372's replacement kinds ride the same mode slots (see `_action_row`); a fifth
# kind would fall off the end silently, so it is asserted here instead.
from rl.engine.combat import RK_DECLINE as _RK_DECLINE   # noqa: E402
assert _RK_DECLINE == MAX_MODES, "RK kinds below RK_DECLINE must fit MAX_MODES"
from rl.engine.effects import (TK_LOCATION, TK_SPELL, TK_TRASH_CARD, TK_UNIT,
                               TK_MODE,
                               unpack_trash)
from rl.engine.state import (MAX_SIDE, C_ABIL, C_CARD, C_CTRL, C_SRC, F_BUFFED, F_NON_UNIT,
                             F_NO_COMBAT_DAMAGE,
                             F_STUNNED, GRANTABLE, MAX_HAND, MAX_PERMS, N_BF, N_LOCATIONS,
                             N_DOMAINS, N_GRANTABLE, N_SEATS, P_ALIVE,
                             P_ARRIVED, P_CARD,
                             P_CTRL, P_DMG, P_FLAGS, P_LOC, P_MIGHT_MOD,
                             P_OWNER,
                             P_READY,
                             PHASE_NAMES, GameState, base_loc, bf_index,
                             bf_loc, fd_bf, fd_slots, is_battlefield, N_FD,
                             is_legend_src, legend_src_seat)

# --- per-row context block, appended to every card feature row --------------
# Uniform across zones so a single shared card encoder can process all of them
# (§6.1). Fields meaningless in a zone are zero there.
(CX_ZONE_HAND, CX_ZONE_BOARD, CX_ZONE_BF, CX_ZONE_FD,
 CX_MINE, CX_READY, CX_DMG, CX_STUNNED, CX_NODMG,
 CX_LOC,                       # N_LOCATIONS slots: own base, enemy base, B0..B2
 CX_AFFORD, CX_ARRIVED,
 CX_CTRL_MINE, CX_CTRL_OPP, CX_CTRL_NONE, CX_CONTESTED,
 CX_SCORED_MINE, CX_SCORED_OPP, CX_FD_PRESENT, CX_FD_LIVE,
 CX_ZONE_LEGEND,
 # --- Might, decomposed by how long each part lasts (see `_board`) ---
 CX_MIGHT, CX_MIGHT_MOD, CX_MIGHT_COMBAT, CX_BUFFED,
 # --- the Chain ---
 CX_ZONE_CHAIN, CX_CHAIN_DEPTH, CX_CHAIN_ABILITY,
 # --- attachment (716-719) ---
 CX_ATTACHED, CX_EQUIPPED,
 CX_GRANT) = (                 # N_GRANTABLE slots: granted keywords
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34)
# CX_LOC spans N_LOCATIONS (5: two bases, two chosen battlefields, and the
# Baron Pit's token slot), which is why CX_AFFORD starts at 14.
# --- the registered decklist (`_decks`) ---
# Defined after the tuple rather than inside it because their indices depend on
# N_GRANTABLE, which CX_GRANT spans.
CX_ZONE_DECK = 34 + N_GRANTABLE      # this row describes a DECK, not an object
CX_COPIES = CX_ZONE_DECK + 1         # copies in the registered list
CX_SEEN = CX_ZONE_DECK + 2           # copies already visible in public zones
CX_KNOWN = CX_ZONE_DECK + 3          # is CX_COPIES real, or is this a sighting?
# --- the Champion Zone (108.3) ---
CX_ZONE_CHAMP = CX_ZONE_DECK + 4     # this row is a Chosen Champion (103.2.a)
# --- Bo3 (D2/D2b) -- APPENDED, so every older column keeps its index and a
# --- network grown from a pre-Bo3 checkpoint reads the old ones unchanged ---
CX_ZONE_SIDE = CX_ZONE_DECK + 5      # this row is a card in MY sideboard
CX_OUTGOING = CX_ZONE_DECK + 6       # the main-deck card a swap is taking out
CX_MATCH = CX_ZONE_DECK + 7          # seen in an EARLIER game of this match
CTX_DIM = 42 + N_GRANTABLE

# Slot counts. Overflow is a bug, not a resize -- silently dropping a card from
# the observation would be invisible in training.
# Both derived from the engine's own caps for the same reason, twice learned:
# `HAND_SLOTS` was 16 ("spells draw cards; 12 was reachable") and real decks
# reached 23. Tracking `MAX_HAND` is the one value that cannot be outgrown
# without the engine itself refusing the state first.
HAND_SLOTS = MAX_HAND
# **Derived, not chosen.** This was 24, picked as "far above anything a legal
# game reaches" -- and then 806.3 landed, games got ~3x longer, and boards of 31
# live permanents appeared in ordinary random play. A hand-tuned headroom
# constant has to be re-tuned every time the game changes and only announces
# itself by crashing mid-run. `MAX_PERMS` is the engine's own hard cap, so
# tracking it is the one value that cannot be outgrown.
BOARD_SLOTS = MAX_PERMS

# The Chain, top-first. **Not** `MAX_CHAIN`, which is 152 -- that is the
# engine's structural ceiling (2*MAX_HAND + MAX_TRIGGERS), not a depth any game
# reaches, and a 152-row dense zone in every observation would cost more than
# everything else here combined. Measured over 56k decisions in 400 real-deck
# games the deepest Chain was 9.
#
# So unlike `hand` and `board`, overflow here is NOT a bug and must not assert:
# rows are encoded nearest-to-resolution first (340.4 -- only the top resolves
# next), so anything dropped is the part that matters least, and `n_chain` stays
# in the globals to tell the net that more is stacked than it can see.
CHAIN_SLOTS = 12

# **The** zone list. `nets.py` reads this through `Encoder.shapes()["zones"]`
# rather than keeping its own copy, because it kept one and it went stale: the
# `legends` zone was added in a8a7bf4 and the net's hardcoded four-tuple was
# never updated, so for every run since, the encoder built the Legend Zone and
# the trunk silently discarded it -- including `CX_READY`, which is whether the
# seat's "Exhaust:" ability is still available this turn. Nothing failed; the
# information simply was not there. One definition, so a new zone cannot be
# added to the observation and dropped from the network again.
# Distinct cards per deck, one row each. **Derived from the real decklists**,
# not chosen: measured over `deck_pool_deal`, a real 40-card list runs 14-22
# distinct cards (mean 16.5, p95 20) because constructed decks play 3-ofs.
# 32 is that with headroom.
#
# Unlike `hand` and `board`, overflow here must NOT assert -- not because it is
# expected, but because the dealers are not all equally realistic. `v1_deal`
# used to average 36.5 distinct cards: it sampled a large pool one card at a
# time and only rejected a 4th copy, so repeats were accidents and its decks
# were nearly singleton. It now picks distinct cards and assigns copy counts
# from the measured distribution (17.4 distinct, `ppo.COPY_WEIGHTS`), and
# `test_ppo` [8] holds it there. Rows are still ordered by copy count
# descending, so anything that does overflow is the thinnest part of the least
# realistic deck, and `n_deck_rows` in the globals tells the net how much it is
# not seeing.
DECK_SLOTS_PER_SEAT = 32
DECK_SLOTS = N_SEATS * DECK_SLOTS_PER_SEAT

# One row per seat, like `legends`: 108.3.a gives each player exactly one
# Champion Zone and 112 puts exactly one card in it. Its own zone rather than
# two more `legends` rows because 107.4 and 108.3 are different zones with
# different rules -- a Legend never leaves (107.4.d) while a Chosen Champion is
# meant to be played out of its zone (108.3.d) -- and a shared zone marker
# would have the net learn the difference from a single context column.
CHAMP_SLOTS = N_SEATS

# 601.1.c.1 -- 10 or fewer sideboard cards; `MAX_SIDE` is the engine's cap. Only
# the seat's OWN sideboard: 403.6 lets the opponent count theirs, which is a
# global, never the contents.
SIDE_SLOTS = MAX_SIDE

# `sideboard` is LAST so a network grown from a pre-Bo3 checkpoint keeps every
# older zone's pooled block at the same trunk offset (`nets.grow_state_dict`).
ZONES = ("hand", "board", "battlefields", "facedown", "legends", "chain",
         "decks", "champions", "sideboard")

# 5*N_DOMAINS: runes_ready + runes_spent for both seats (4), plus this seat's
# pool_power (1). The +1 is pool_power's [A] column -- see state.D_ANY.
# The +2 is `pending_ready_runes` for both seats -- see `_globals`.
# The +4 at the end is `_decks`' two knowledge flags and two row counts --
# masked mean/max pooling discards how MANY rows a zone had, and "how much
# of their deck have I actually seen" is the whole point of the zone.
# The +50 is `_standing` -- see that method for why each entry is there. A
# measured audit found 24 pieces of standing state that the ENGINE reads to
# decide the game and the encoder never touched, `victory_bonus` among them.
# The +4 after it is the suspended damage assignment -- who is assigning, the
# budget left, and how many kills are committed. See `_standing`'s tail.
# The +2 is 108.3.d: whether each player's Chosen Champion is still in its
# zone, which its `_champions` row no longer says now that it can be played.
# The final +4 is [Show Off], per seat: whether one is live this ply and the
# shown-off unit's Might.
# The final +17 is the Bo3 match context (`_match_globals`), appended last so a
# grown pre-Bo3 network reads every older global at its old index.
GLOBAL_DIM = 40 + 5 * N_DOMAINS + 1 + 2 + 7 + 4 + 51 + 4 + 2 + 4 + 17

# ---------------------------------------------------------------------------
# Every GameState field is either READ here or listed below with a reason.
#
# `victory_bonus` is why this exists. Aspirant's Climb raises the Victory Score
# by 1, `check_winner` honours it, and the observation never mentioned it -- so
# the agent played every game of that matchup toward the wrong finish line, with
# nothing failing. A measured probe (perturb a field, re-encode, see whether
# anything moves) found 24 more like it.
#
# The lesson is not "add 24 fields". It is that the engine grows and the encoder
# does not follow, silently, and no test noticed for months. So this set is the
# same device as `mirror.SEAT_AXIS`: a new field must be classified, or the gate
# in `test_env` fails and names it. Being listed here is a CLAIM -- that the
# policy either does not need the field or can already see its consequence --
# and the reason has to hold.
#
# The distinction that does the work: state which gates LEGALITY needs no entry
# in the globals, because the action rows carry what is on offer and absence is
# information. State that changes what a position is WORTH without changing what
# can be done in it has to be given.

OBS_UNREAD = frozenset({
    # Bo3. `pg_bf_pick` is HIDDEN from the opponent until the deal and the
    # seat's own pick is on the board (`bf_card`) from then on; `seen_game` is
    # the match dealer's bookkeeping -- this game's public cards are read from
    # the zones they are in.
    "pg_bf_pick", "seen_game",
    # Choosing which runes pay: who is asked is `to_move`, the options are the
    # action rows (`pay_opts`, `pay_card` are read there), and the rest is the
    # replay machinery in `actions._apply_choosing_payment`.
    "pend_pay", "pay_n_opts", "pay_force", "pay_force_n", "pay_force_i",
    "pay_act_kind", "pay_act_arg",

    # HIDDEN BY RULE -- a leak if encoded. 107.3.f / the deck's order.
    "deck", "mull_mask", "n_look", "rune_deck", "rune_head",

    # MID-RESOLUTION SCRATCH -- the ACTION ROWS carry the choice on offer, which
    # is the only part a policy needs. These live between a suspend and its
    # resume and are meaningless outside it.
    "amt_kind", "amt_loc", "amt_spell", "bonus_uid", "chain_from_trigger",
    "chain_targets", "chain_uid", "copy_pending", "cull_spell_seat",
    "dj_cat", "dj_first", "dj_hand_keep", "dj_keep", "dj_rune_keep",
    "dj_seat", "empower_src", "grenade_hits", "grenade_ply", "group_loc",
    "group_loc_opts", "hp_attach", "hp_cost", "hp_dest", "hp_discount",
    "hp_kw", "hp_max_energy", "hp_optional", "hp_spells", "hp_tag",
    "hp_types", "kill_disc_e", "kill_disc_p", "last_burned", "last_token",
    "last_token_n", "look_domain", "look_last_pick", "look_max_might",
    "look_min_energy", "look_multi", "look_optional", "look_pick_dest",
    "look_rest_dest", "look_reveal", "look_type_mask", "move_from",
    "move_to", "n_attached", "n_group_loc", "n_name_opts", "n_sarc",
    "name_src", "pend_altar", "pend_guard", "guard_cands",
    "n_guard_cands",
    "pend_ask_caster",
    "pend_ask_no",
    "pend_ask_yes", "pend_cost_recycle", "pend_cost_recycle_n", "pend_cull",
    "pend_cull_dest", "pend_cull_first", "pend_cull_keep", "pend_cull_mode",
    "pend_cull_skip", "pend_cull_type", "pend_discard", "pend_discard_ops",
    # Combat damage assignment. The candidate rows ARE the action rows, so the
    # target list and its count need no global; `pend_dmg_bf` is the battlefield
    # the Showdown is already at. `pend_dmg_kills` is the honest exception --
    # see `_standing`'s tail for why its identities are left out and what it
    # would cost to include them. `pend_dmg`, `pend_dmg_pool` and
    # `pend_dmg_n_kill` ARE read there -- the budget is the decision.
    "pend_dmg_targets", "pend_dmg_n_tgt", "pend_dmg_bf", "pend_dmg_kills",
    "pend_dmg_done",
    # [Show Off] has no entry here: `pend_show_off` is read to decide what an
    # action row means, and `show_off_perm` / `show_off_card` / `show_off_ply`
    # are read in `_standing`. What the policy gets is the shown-off unit's
    # SIZE, not its identity -- a genuine gap for the hand-revealed case,
    # where a human would remember the card; closing it means a hand-zone row
    # for a card the opponent does not hold, which is a shape change.
    "pend_discard_src", "pend_discard_tgt", "pend_double", "pend_grave",
    "pend_grave_dest", "pend_grave_owner", "pend_hide", "pend_kill_play_loc",
    "pend_kill_play_seat", "pend_mull", "pend_order", "pend_phase",
    "pend_play_seat", "pend_repeat_bound", "pend_repeat_card",
    "pend_repeat_hand", "pend_repeat_seat", "pend_repeat_tgts",
    "pend_reveal", "pend_reveal_xp", "pend_split", "pend_tax_cost",
    "pend_then", "pf_cards", "pf_first", "pf_stage", "resolving_bonus",
    "resolving_paid", "resume_bound", "resume_card", "resume_ctx",
    "resume_ctx2", "resume_hand", "resume_idx", "resume_kind", "resume_op",
    "resume_seat", "resume_src", "resume_subj", "resume_tgts",
    "reveal_hold_return", "reveal_play_loc", "rp_armed", "rp_cost",
    "rp_discount", "rp_empower", "rp_from_sarc", "rp_here", "rp_kill",
    "rp_owner", "rp_power", "rp_zone", "sarc_cards", "split_alloc",
    "split_left", "split_loc", "split_spell", "split_xp", "steal_seat",
    "steal_slot", "steal_stage", "steal_targets", "steal_uid",

    # PER-PERMANENT DETAIL -- reflected in the board row's True Might / flags, or
    # in whether the action is offered at all. A spent once-per-turn ability
    # is visible as an A_ACTIVATE that is simply not there.
    "altar_ply", "repl_pick", "repl_ply",
    "armory_ply", "banish_death_ply", "base_might_ply",
    "base_might_val", "block_next_ply", "combat_might_ply",
    "combat_might_val", "conquer_ply", "copy_of", "copy_via", "ctrl_link",
    "death_shield_ply", "desig", "desig_seat", "double_dmg_ply", "eot_kind",
    "eot_ply", "extra_buffs", "flow_grant_banish", "flow_grant_card",
    "flow_grant_e", "flow_grant_p", "flow_grant_ply", "foe_dmg",
    "granted_card", "granted_ply", "guillotine_ply", "hold_return",
    "kw_grant_turn", "legend_pile", "mark_ply", "mark_seat", "mark_slot",
    "might_hi", "mode_used_mask", "mode_used_ply", "move_count", "move_ply",
    "n_hold_return", "named", "once_used", "shield_amt", "shield_ply",
    "tag_grant", "zero_cards", "zero_owner",

    # TERMINAL OR STRUCTURAL -- the episode is over, or a count the zone MASK
    # already carries.
    "bf_conquered_ply", "bf_contester", "bf_first_use", "bf_prev_ctrl",
    "bf_replaced", "big_spell_ply", "buff_bonus_n", "buff_bonus_ply",
    "chose_enemy_n", "chose_enemy_ply", "decl_mask", "died_in_beginning",
    "discarded_ply", "draw_ply", "equip_played_ply", "excess_attacking",
    "excess_ply", "focus", "free_gear_ply", "free_hide_ply",
    "next_spell_bonus_ply", "next_spell_repeat_ply", "next_unit_ready_ply",
    "no_cards", "no_effect_damage_ply", "pending_add_any", "played_types",
    "pool_rstr_e", "pool_rstr_p", "power_spent_ply", "rune_recycled_n",
    "second_draw_ply", "showdown_step", "trig", "truncated", "unit_died_ply",
    "winner",

})


# Action-row layout after the kind one-hot and card block.
ACT_EXTRA = N_LOCATIONS + 1 + 3 + 1 + 1 + 1 + 1 + 2 + 1 + MAX_MODES + 2 + N_DOMAINS + 2
# The last N_DOMAINS (2026-10-07) are a payment option (`A_PAY_SPLIT`): how
# many runes of each domain it recycles, /3. Appended, so older checkpoints
# grow into it (`nets.grow_state_dict`).
# The final 2 (2026-10-07) describe the PERMANENT an action is about, when it is
# about one: *it is mine*, and *it is in the current combat* (at the showdown
# battlefield during a showdown). The board rows carry ownership (`CX_MINE`)
# but are pooled before the action head, so a target row could not say whose
# unit it named -- the AI cast Discipline on the ENEMY and En Garde on a unit
# at base while its lone defender died (`docs/REVIEW_HUMAN_GAMES.md`). Nothing
# is forbidden: an enemy target stays legal, it is just no longer invisible.
# The final 2 describe a TAG being named (BACKLOG 7b): how many enemy and how
# many friendly units on the board carry it. A tag option used to be encoded
# only as an index in the four mode slots, which every tag past the fourth fell
# off -- the policy could not tell "Poro" from "Pirate".
# loc, is_bf, ctrl(3), counts, might, cost, whose, mode one-hot
#
# The last slot is "whose thing is this", and it exists because a target's
# owner is not always recoverable from the rest of the row. A permanent target
# carries its controller in the location block; a card in a TRASH does not, and
# "recycle a card from trashes" reaches both piles. Recycling your own card
# refills your deck, recycling theirs denies them a redraw -- opposite plays
# from one action kind, indistinguishable without this bit.


@dataclass(frozen=True)
class Obs:
    """One decision point, as the acting seat sees it."""
    zones: dict[str, np.ndarray]      # {name: [slots, row_dim] float32}
    zone_mask: dict[str, np.ndarray]  # {name: [slots] bool}
    globals: np.ndarray               # [GLOBAL_DIM] float32
    legal_actions: np.ndarray         # [A_max, act_dim] float32
    action_mask: np.ndarray           # [A_max] bool
    to_move: int
    n_legal: int
    privileged: np.ndarray | None     # critic-only, training-only

    def public_bytes(self) -> bytes:
        """Everything the policy may see, flattened. Used by the leak test."""
        parts = [self.globals.tobytes(), self.legal_actions.tobytes(),
                 self.action_mask.tobytes()]
        for k in sorted(self.zones):
            parts.append(self.zones[k].tobytes())
            parts.append(self.zone_mask[k].tobytes())
        return b"".join(parts)


def _loc_slot(loc: int, seat: int) -> int:
    """Location as 0=own base, 1=enemy base, 2+i=battlefield i. Canonical."""
    if is_battlefield(loc):
        return 2 + bf_index(loc)
    return 0 if loc == base_loc(seat) else 1


class Encoder:
    """Builds `Obs` from `GameState`. Holds the frozen card feature matrix."""

    def __init__(self, table: CardTable, cfg: Config,
                 privileged: bool = True) -> None:
        self.table = table
        self.cfg = cfg
        self.want_privileged = privileged
        self.cards = table.features().astype(np.float32)     # [n_cards, D]
        self.card_dim = int(self.cards.shape[1])
        self.row_dim = self.card_dim + CTX_DIM
        self.act_dim = len(A.KIND_NAMES) + 1 + self.card_dim + ACT_EXTRA
        self.a_max = cfg.max_actions
        # One row per hand slot and one per FACEDOWN SLOT -- a battlefield may
        # hold two (Bandle Tree), and the belief head is asked about each card.
        self.priv_dim = (HAND_SLOTS + N_FD) * self.card_dim

    # -- rows ------------------------------------------------------------

    def _row(self, card: int, zone: int) -> np.ndarray:
        r = np.zeros(self.row_dim, np.float32)
        if card >= 0:
            r[:self.card_dim] = self.cards[card]
        r[self.card_dim + zone] = 1.0
        return r

    def _hand(self, state: GameState, seat: int):
        """Cards in hands this seat may see.

        Normally that is its own hand alone. A card that reveals the opponent's
        hand (Scuttle Crab) adds THEIR cards to the same zone rather than a new
        one: the rows already carry `CX_MINE`, so ownership is a feature the
        net reads rather than a shape it has to be rebuilt for. That keeps the
        observation shape -- and every checkpoint -- valid.
        """
        z = np.zeros((HAND_SLOTS, self.row_dim), np.float32)
        m = np.zeros(HAND_SLOTS, bool)
        seats = [seat]
        if int(state.saw_hand[seat]) == int(state.ply):
            seats.append(1 - seat)
        k = 0
        for owner in seats:
            n = int(state.n_hand[owner])
            if owner == seat:
                # The seat's OWN hand must always fit -- MAX_HAND == HAND_SLOTS
                # by construction, so this is a real invariant.
                assert n <= HAND_SLOTS, (
                    f"hand of {n} exceeds HAND_SLOTS={HAND_SLOTS}")
            else:
                # The revealed half is EXTRA information the policy would not
                # have at all without the card, so if two hands somehow cannot
                # both fit, showing fewer revealed cards is strictly better
                # than an assertion that takes down a training run. Measured
                # peak for both hands together is 15 against 60 slots, so this
                # is headroom rather than a live cap -- but MAX_HAND is 60, and
                # hand-picked capacities in this engine have a history of being
                # overtaken.
                n = min(n, HAND_SLOTS - k)
            for j in range(n):
                card = int(state.hand[owner, j])
                r = self._row(card, CX_ZONE_HAND)
                r[self.card_dim + CX_MINE] = float(owner == seat)
                # Affordability is a genuine feature, not a shortcut: it is the
                # single fact that most changes what a hand card means right
                # now, and the net would otherwise have to rederive
                # max(energy, power) against the rune board from scratch.
                # Read against the card's OWNER -- for your own hand that is
                # you, and for a revealed one "can they actually cast it" is
                # the question worth asking.
                r[self.card_dim + CX_AFFORD] = float(
                    A.plan_payment(state, self.table, owner, card) is not None)
                z[k] = r
                m[k] = True
                k += 1
        return z, m

    def _board(self, state: GameState, seat: int):
        """Live permanents, with Might decomposed by PERSISTENCE.

        The card block carries PRINTED Might, and for a long time that was the
        only Might in the observation -- so a unit's true Might was wrong on
        8.5% of board rows in real games, by anything from -5 to +7. The policy
        could see `CX_DMG`, the damage marked on a unit, but not the number
        that damage has to reach to kill it. That is the single most important
        quantity in a combat.

        Encoding only "current Might" alongside it would fix the arithmetic and
        still lose the planning, because `combat.might` sums four things that
        disappear on four different schedules:

          P_MIGHT_MOD        "this turn" -- gone in the end-of-turn cleanup
          Buff counter       a counter (703); survives cleanup, gone when the
                             unit leaves play (705)
          static_might       continuous; gone when its SOURCE leaves, and where
                             an Equipment's Might Bonus will land
          combat_role_bonus  [Assault]/[Shield]; exists only inside a Combat
                             (807.1.c, 814.1.c)

        So three columns rather than one, and printed Might already in the row:

          CX_MIGHT          what it is now -- decides this combat
          CX_MIGHT_MOD      the part that evaporates at end of turn
          CX_MIGHT_COMBAT   the part that evaporates when the Combat ends

        which lets the net recover "what this unit resets to" by subtraction,
        and tells apart a real 5-Might body from a 2-Might body wearing a pump
        -- indistinguishable to any pair of (printed, current) numbers, because
        neither says which part decays.

        One inexactness, deliberately left: 143.2.b floors Might at 0, so on a
        unit modified below zero the parts no longer sum to `CX_MIGHT`. Encoding
        the unfloored total instead would misstate the number that actually
        decides lethality, which is the more expensive of the two errors.
        """
        z = np.zeros((BOARD_SLOTS, self.row_dim), np.float32)
        m = np.zeros(BOARD_SLOTS, bool)
        k = 0
        for i in range(state.n_perms):
            row = state.perms[i]
            if row[P_ALIVE] != 1:
                continue
            assert k < BOARD_SLOTS, f"more than {BOARD_SLOTS} live permanents"
            r = self._row(int(row[P_CARD]), CX_ZONE_BOARD)
            c = self.card_dim
            r[c + CX_MINE] = float(row[P_CTRL] == seat)
            r[c + CX_READY] = float(row[P_READY])
            r[c + CX_DMG] = float(row[P_DMG]) / 5.0
            r[c + CX_STUNNED] = float(bool(row[P_FLAGS] & F_STUNNED))
            r[c + CX_NODMG] = float(bool(row[P_FLAGS] & F_NO_COMBAT_DAMAGE))
            r[c + CX_LOC + _loc_slot(int(row[P_LOC]), seat)] = 1.0
            r[c + CX_ARRIVED] = float(int(row[P_ARRIVED]) == state.ply)
            # Never `table.might` here -- PLAN.md §1.3.d, and the whole point.
            r[c + CX_MIGHT] = combat.might(state, self.table, i) / 5.0
            r[c + CX_MIGHT_MOD] = float(row[P_MIGHT_MOD]) / 5.0
            r[c + CX_MIGHT_COMBAT] = combat.combat_role_bonus(
                state, self.table, i) / 5.0
            r[c + CX_BUFFED] = float(bool(row[P_FLAGS] & F_BUFFED))
            # 718 -- Attached and Top-Most are two different things and a row
            # can be either, so they are two features rather than one.
            #
            # Without these an Attached gear encodes exactly like a free one,
            # and the difference is not cosmetic: 718.2 makes its whole printed
            # text Inactive, so the policy would see an activatable ability the
            # action layer will never offer. That is the same failure the
            # battlefield rows had, where 66 distinct cards collapsed to three
            # identical rows.
            r[c + CX_ATTACHED] = float(state.is_attached(i))
            r[c + CX_EQUIPPED] = float(bool(state.attachments(i)))
            # Granted keywords (kw_grant) -- a unit handed [Tank] is assigned
            # combat damage first and is otherwise identical to one that was
            # not, so without this the two encode the same.
            g = state.kw_grant[i]
            for j in range(N_GRANTABLE):
                if g[j]:
                    r[c + CX_GRANT + j] = float(g[j])
            z[k] = r
            m[k] = True
            k += 1
        return z, m

    def _battlefields(self, state: GameState, seat: int):
        z = np.zeros((N_BF, self.row_dim), np.float32)
        m = np.zeros(N_BF, bool)
        for i in state.live_bfs():
            m[i] = True
            r = self._row(int(state.bf_card[i]), CX_ZONE_BF)
            c = self.card_dim
            ctrl = int(state.bf_ctrl[i])
            r[c + (CX_CTRL_NONE if ctrl < 0 else
                   CX_CTRL_MINE if ctrl == seat else CX_CTRL_OPP)] = 1.0
            r[c + CX_MINE] = float(ctrl == seat)
            r[c + CX_CONTESTED] = float(state.bf_contested[i])
            r[c + CX_SCORED_MINE] = float(state.bf_scored[seat, i])
            r[c + CX_SCORED_OPP] = float(state.bf_scored[1 - seat, i])
            # Any facedown card here, and any LIVE one: a battlefield may hold
            # two (Bandle Tree), and what the battlefield row says is whether
            # the ground is threatening at all.
            r[c + CX_FD_PRESENT] = float(any(int(state.fd_owner[k]) >= 0
                                             for k in fd_slots(i)))
            r[c + CX_FD_LIVE] = float(any(
                int(state.fd_owner[k]) >= 0
                and int(state.fd_ply[k]) < int(state.ply) for k in fd_slots(i)))
            r[c + CX_LOC + _loc_slot(bf_loc(i), seat)] = 1.0
            z[i] = r
        return z, m

    def _facedown(self, state: GameState, seat: int):
        """107.3.f -- the zone is public, the card is not.

        A slot the opponent occupies is encoded as *present, identity unknown*:
        the mask says a card is there, the card block stays zero. That gap is
        the belief head's target in Phase 6, so it must be a real hole in the
        observation rather than a quietly filled-in one.
        """
        z = np.zeros((N_FD, self.row_dim), np.float32)
        m = np.zeros(N_FD, bool)
        for k in range(N_FD):
            i = fd_bf(k)
            owner = int(state.fd_owner[k])
            if owner < 0:
                continue
            mine = owner == seat
            # "You can look at their facedown cards this turn" fills exactly
            # this hole -- the slot was already public (107.3.f), only the
            # identity was not.
            seen = mine or int(state.saw_fd[seat]) == int(state.ply)
            r = self._row(int(state.fd_card[k]) if seen else -1, CX_ZONE_FD)
            c = self.card_dim
            r[c + CX_MINE] = float(mine)
            r[c + CX_FD_PRESENT] = 1.0
            # 811.1.b -- a card hidden this turn is not playable until the next
            # one. Public information (everyone saw when it was hidden), and
            # decisive for whether the threat is real right now.
            r[c + CX_FD_LIVE] = float(int(state.fd_ply[k]) < int(state.ply))
            r[c + CX_LOC + _loc_slot(bf_loc(i), seat)] = 1.0
            z[k] = r
            m[k] = True
        return z, m

    def _chain(self, state: GameState, seat: int):
        """What is waiting to resolve -- 340.4, top of the Chain first.

        The observation used to carry `n_chain / 4.0` and nothing else, so on
        26.9% of real decisions something was on the Chain and the policy could
        not see WHAT. On 12.7% the actor was holding a card it could have
        responded with. That is the counterspell decision -- and the bluffing
        behaviour Phase 6/7 exists to measure -- made blind: `env.should_auto_pass`
        deliberately refuses to collapse those windows precisely because they
        are the interesting ones, and then the encoder described them as a
        single scalar.

        Naming the cards leaks nothing. A Chain Item is public the moment it is
        played (337.1); what stays hidden is the Facedown Zone, which has its
        own zone and its own deliberate hole.

        `CX_CHAIN_DEPTH` is distance from the TOP, not array position, because
        "resolves next" is what decides whether a response can still catch it.
        """
        z = np.zeros((CHAIN_SLOTS, self.row_dim), np.float32)
        m = np.zeros(CHAIN_SLOTS, bool)
        n = int(state.n_chain)
        for k in range(min(n, CHAIN_SLOTS)):
            i = n - 1 - k                     # top of the Chain first
            item = state.chain[i]
            r = self._row(int(item[C_CARD]), CX_ZONE_CHAIN)
            c = self.card_dim
            r[c + CX_MINE] = float(int(item[C_CTRL]) == seat)
            r[c + CX_CHAIN_DEPTH] = float(k) / 4.0
            # 337.2 -- an ABILITY on the Chain is not a spell, so "counter a
            # spell" cannot touch it. Same card id, completely different
            # question, and nothing else in the row tells them apart.
            r[c + CX_CHAIN_ABILITY] = float(int(item[C_ABIL]) >= 0)
            src = int(item[C_SRC])
            if 0 <= src < state.n_perms:
                r[c + CX_LOC + _loc_slot(int(state.perms[src, P_LOC]),
                                         seat)] = 1.0
            z[k] = r
            m[k] = True
        return z, m

    def _legends(self, state: GameState, seat: int):
        """107.4 -- the Legend Zone, one Champion Legend per seat.

        Public and permanent: 174.2.b establishes it at the start of the game
        and it never leaves (107.4.d), so unlike every other zone here there is
        no hidden information and no arrival to track. What DOES change is
        `CX_READY` -- nearly every legend charges "Exhaust:" for its ability, so
        whether it is spent is the single fact that decides what the seat can do
        this turn.

        Not a location (107.4.b), so no `CX_LOC` block is set.
        """
        z = np.zeros((N_SEATS, self.row_dim), np.float32)
        m = np.zeros(N_SEATS, bool)
        for k, owner in enumerate((seat, 1 - seat)):     # canonical: mine first
            card = int(state.legend[owner])
            if card < 0:
                continue
            r = self._row(card, CX_ZONE_LEGEND)
            c = self.card_dim
            r[c + CX_MINE] = float(owner == seat)
            r[c + CX_READY] = float(state.legend_ready[owner])
            z[k] = r
            m[k] = True
        return z, m


    def _champions(self, state: GameState, seat: int):
        """108.3 -- the Champion Zone, one Chosen Champion per seat.

        **This is the other half of the pre-game archetype signature.** 103.1.a
        puts the Champion Legend in a Public zone and `_decks` already reads it;
        103.2.a.2 then requires the Chosen Champion to share that Legend's
        champion tag, and 108.3.e makes this zone Public too. So before a card
        is drawn both players know each other's Legend AND which build of that
        champion is being piloted -- Darius - Trifarian and Darius - Reaper of
        Noxus are the same Legend and very different decks.

        Unconditional, with no `deck_known` gate: unlike the 40-card list, this
        is public by rule in every game, in Bo1 game 1 as much as Bo3 game 3.

        Not counted in `_decks`. 103.2 registers it inside the 40 but 133.4
        starts it OUTSIDE the Main Deck, so folding it into the deck rows would
        claim a card can still be drawn that is already on the table. The two
        zones are complementary, and 11 of the 30 real lists run further copies
        in the deck proper, which `_decks` reports on its own.

        `CX_READY` is deliberately unset. A card in a zone is not a Game Object
        with a ready state (133.4), and 705.1 has it hold no Buffs either.
        """
        z = np.zeros((CHAMP_SLOTS, self.row_dim), np.float32)
        m = np.zeros(CHAMP_SLOTS, bool)
        for k, owner in enumerate((seat, 1 - seat)):     # canonical: mine first
            # **`champion_reg`, not `champion`.** 108.3.d lets the card be
            # played out of the zone, and reading occupancy here made the
            # archetype signature vanish the moment it was cast -- the policy
            # would lose half of what tells it which deck it is piloting,
            # exactly when the board is most committed. Identity is public and
            # permanent (108.3.e, and once played it is on the board or in a
            # trash); whether it is still AVAILABLE is two globals in
            # `_standing`.
            card = int(state.champion_reg[owner])
            if card < 0:
                continue          # a random-pool deal has no decklist (112)
            r = self._row(card, CX_ZONE_CHAMP)
            r[self.card_dim + CX_MINE] = float(owner == seat)
            z[k] = r
            m[k] = True
        return z, m


    def _sideboard(self, state: GameState, seat: int):
        """This seat's OWN sideboard (403), one row per distinct card.

        Empty outside a match. Never the opponent's: 403.6 lets a player count
        the cards in an opponent's sideboard, which `_match_globals` carries,
        but not see them -- and the leak test perturbs exactly that.
        """
        z = np.zeros((SIDE_SLOTS, self.row_dim), np.float32)
        m = np.zeros(SIDE_SLOTS, bool)
        counts: dict[int, int] = {}
        for c in state.side[seat, :int(state.n_side[seat])]:
            counts[int(c)] = counts.get(int(c), 0) + 1
        for i, card in enumerate(sorted(counts)[:SIDE_SLOTS]):
            r = self._row(card, CX_ZONE_SIDE)
            r[self.card_dim + CX_MINE] = 1.0
            r[self.card_dim + CX_COPIES] = float(counts[card]) / 3.0
            z[i] = r
            m[i] = True
        return z, m

    def _decks(self, state: GameState, seat: int):
        """What this seat legitimately knows about each player's DECK.

        **Why this zone exists.** The deck used to be one number -- how many
        cards were left -- so a policy could not tell whether it was piloting
        an aggressive list or a grindy one, and "play to your deck's plan" was
        not a behaviour it could express. It saw four cards in hand and the
        number 31.

        **Three grades of knowledge, and the rules fix which applies.**

        Your own list you always know. Against an opponent, 103.1.a puts their
        Champion Legend in the Legend Zone at the start of the game and
        355.10.a.1 makes that zone Public, while 103.1.b.2 has the Legend fix
        the deck's entire Domain Identity -- so you always know the *shape* of
        their deck before a card is drawn, and at competitive level a Legend
        implies most of a list. On top of that, cards of theirs that have been
        seen in public zones are known individually. Only the full registered
        40 is conditional, and `state.deck_known` carries it: true in a match
        after game 1, false against a stranger.

        So each row is a distinct card with three numbers rather than one:

          CX_COPIES   how many are in the registered list  (known lists only)
          CX_SEEN     how many are already sitting in public zones
          CX_KNOWN    whether CX_COPIES means anything at all

        `CX_COPIES - CX_SEEN` is then "how many could still be hidden", which is
        the quantity a player actually counts, and the net can form it. With an
        unknown list only `CX_SEEN` is filled, and the row is a sighting.

        **This is also where the trash finally arrives.** A trash is Public
        (108.5) and is the single richest evidence for what remains in an
        opponent's deck, but the observation carried it as `n_trash / 20` --
        one scalar, contents discarded. Folding it in here rather than giving
        it a zone of its own is the honest framing: a card in their trash is
        not interesting as a trash card, it is interesting as a card you now
        know their deck contained and no longer holds.

        **Read as a multiset, never in order.** `state.decklist` is stored in
        registration order and the live `deck` is in draw order; either read
        positionally would leak the shuffle. Counting copies per distinct card
        is order-free by construction, which is what keeps `test_env`'s
        deck-order perturbation invisible.
        """
        z = np.zeros((DECK_SLOTS, self.row_dim), np.float32)
        m = np.zeros(DECK_SLOTS, bool)
        counts = np.zeros(2, np.float32)
        for k, owner in enumerate((seat, 1 - seat)):     # canonical: mine first
            known = (owner == seat) or bool(state.deck_known[owner])
            reg: dict[int, int] = {}
            if known:
                n = int(state.n_decklist[owner])
                for j in range(n):
                    c = int(state.decklist[owner, j])
                    if c >= 0:
                        reg[c] = reg.get(c, 0) + 1
            seen = self._seen_cards(state, owner)
            # Bo3: cards of THEIRS seen in earlier games of the match. Known to
            # exist in the list, not where they are now -- so they are rows of
            # their own kind (CX_MATCH) rather than more CX_SEEN. Empty in Bo1.
            match_seen = ({int(c) for c in state.seen_match[owner] if int(c) >= 0}
                          if owner != seat else set())
            outgoing = (int(state.sb_out) if owner == seat
                        and int(state.pg_seat) == seat else -1)
            # Deterministic order -- replay is bit-identical and must stay so.
            # Copy count descending, so an overflowing procedural deck drops
            # its singletons rather than its core.
            cards = sorted(set(reg) | set(seen) | match_seen,
                           key=lambda c: (-(reg.get(c, 0) + seen.get(c, 0)), c))
            base = k * DECK_SLOTS_PER_SEAT
            for i, card in enumerate(cards[:DECK_SLOTS_PER_SEAT]):
                r = self._row(card, CX_ZONE_DECK)
                c = self.card_dim
                r[c + CX_MINE] = float(owner == seat)
                r[c + CX_COPIES] = float(reg.get(card, 0)) / 3.0
                r[c + CX_SEEN] = float(seen.get(card, 0)) / 3.0
                r[c + CX_KNOWN] = float(known)
                r[c + CX_MATCH] = float(card in match_seen)
                r[c + CX_OUTGOING] = float(card == outgoing)
                z[base + i] = r
                m[base + i] = True
            counts[k] = float(len(cards))
        return z, m, counts

    def _seen_cards(self, state: GameState, owner: int) -> dict[int, int]:
        """Cards of `owner`'s that ANY player has legitimately seen, by count.

        Public zones only, so this is safe to show either seat: the Trash
        (108.5), the Banish pile (108.6.e), live permanents on the board, and
        the Legend Zone (355.10.a.1). Deliberately NOT the hand, the facedown
        card or the undrawn deck -- those are the hidden information the belief
        head is meant to predict, and putting them here would be the leak the
        zone is otherwise built to avoid.
        """
        out: dict[int, int] = {}

        def add(card: int) -> None:
            if card >= 0:
                out[card] = out.get(card, 0) + 1

        for j in range(int(state.n_trash[owner])):
            add(int(state.trash[owner, j]))
        for j in range(int(state.n_banished[owner])):
            add(int(state.banished[owner, j]))
        for i in range(int(state.n_perms)):
            if (int(state.perms[i, P_ALIVE]) == 1
                    and int(state.perms[i, P_OWNER]) == owner):
                add(int(state.perms[i, P_CARD]))
        add(int(state.legend[owner]))
        return out

    # -- globals ---------------------------------------------------------

    def _standing(self, state: GameState, seat: int) -> list[float]:
        """Standing state the ENGINE reads and the policy could not see.

        **Found by measurement, not by reading.** Perturbing a field and
        re-encoding says definitively whether the policy can see it -- the same
        trick as `test_env`'s leak test, inverted. 24 fields survived that probe
        invisible, every one of them something a player at the table tracks.

        The distinction that matters: state which gates LEGALITY is already
        visible, because the action rows carry what is on offer and the agent
        reads the consequence from an action's absence. What is not visible is
        state that changes **what a position is worth** without changing what
        can be done in it right now. `victory_bonus` is the pure case -- it
        moves the finish line and touches no legal action.

        Two encoding rules:
          * a `*_ply` field holds a ply stamp, so what matters is `== state.ply`
            ("is this live right now"), never the raw number. A stamp keyed on
            anything else covered the opponent's turn too, which is the bug
            [[riftbound-turn-is-a-round-ply-is-a-turn]] records.
          * counters are scaled by roughly what a card counts TO, not by a
            theoretical maximum, so the useful range fills the unit interval.
        """
        foe = 1 - seat
        ply = int(state.ply)
        # The REAL target, bonus included. `check_winner` computes exactly this
        # (`state.victory_score + victory_bonus`), so anything else here would
        # be the observation disagreeing with the rule that ends the game --
        # which is why this reads the STATE field and not `cfg.victory_score`
        # (BACKLOG #10). `new_game` copies the cfg into the field, so the value
        # is unchanged; what changes is that there is now one place to move it.
        # Aspirant's Climb makes it 9, and because turns-to-win is
        # `ceil((V - points) / 2)`, moving V by one INVERTS which scores are
        # tempo-efficient. An off-by-one here is a whole wasted turn.
        v = int(state.victory_score) + int(state.victory_bonus)
        g = [float(v) / 8.0]
        # Turns to win at ~2 points a turn, both seats. A LOWER BOUND, not a
        # promise: non-Conquer sources (471.1.a.1) and a third battlefield both
        # beat 2 a turn. Given exactly because it is exact arithmetic the policy
        # would otherwise have to carve out of a smooth `points / vs` scalar --
        # and because it is what decides whether a single point is worth the
        # unit it costs. At 6 of 8 a point saves a turn; at 7 of 8 it saves
        # nothing, and the unit that walked into a losing showdown to get it was
        # spent for free.
        for sd in (seat, foe):
            need = max(0, v - int(state.points[sd]))
            g.append(float((need + 1) // 2) / 4.0)
        # Triggers queued but not yet resolved (383.3.d orders them). Something
        # is about to happen that the board does not show yet.
        g.append(min(1.0, float(state.n_trig) / 8.0))
        # Consecutive passes: how close this priority window is to closing, and
        # so whether the top of the Chain is about to resolve (340.4).
        g.append(min(1.0, float(state.passes) / 2.0))
        for sd in (seat, foe):
            g += [
                float(state.extra_turns[sd]),              # Time Warp
                float(state.hold_points[sd]) / 2.0,         # bonus per Hold
                float(state.hold_points_ply[sd] == ply),
                float(state.legend_emp[sd]),                # 441.1.a, binary
                float(state.legend_once[sd] == ply),        # already used
                float(state.death_guard[sd] == ply),        # 136.2.d promise
                float(state.riches_on[sd]),                 # Endless Riches
                float(state.unit_tax_ply[sd] == ply),       # units cost more
                float(state.units_enter_ready_turn[sd] == ply),
                float(state.xp_gained_ply[sd] == ply),      # Wily Newtfish
                float(state.next_spell_discount[sd]) / 3.0,
                float(state.next_discount[sd, 0]) / 3.0,    # energy, and power
                float(state.next_discount[sd, 1]) / 3.0,    # separately: runes
                float(state.n_zero[sd]) / 4.0,              # The Zero Drive
                float(state.legend_pile_n[sd]) / 4.0,       # Jhin counts to 4
                float(state.spells_played[sd]) / 6.0,
                float(state.cards_completed[sd]) / 6.0,
                float(state.power_spent[sd]) / 6.0,
                float(state.draw_count[sd]) / 6.0,
                float(state.excess_amt[sd]) / 6.0,
                float(int((state.delayed[sd] >= 0).sum())) / 4.0,
                float(state.recycled_n[sd]) / 6.0,
                float(state.banished_n[sd]) / 6.0,
            ]
        # A suspended combat-damage assignment (465.2.c.2). The candidates are
        # the action rows, so what they cannot carry is the BUDGET: how much of
        # the pool is left to spend, and how many kills are already committed.
        # Without those two the policy is choosing its second kill unable to
        # tell a unit it has already doomed from one it can no longer afford --
        # both simply stop being offered.
        #
        # Known gap, deliberately left: WHICH units this seat has already
        # chosen is still invisible, because nothing is marked on the board
        # until both seats answer (that is what keeps the sequential ask from
        # leaking). Encoding it properly means a per-row "doomed by my pending
        # assignment" flag, which widens `row_dim` for every zone; the two
        # numbers here are the first-order part of the decision.
        # 108.3.d -- is each player's Chosen Champion still available? The
        # ROW in `_champions` names which champion it is either way, because
        # that identity is public and permanent; what changes is whether it is
        # still a threat to come. Not inferable from the action rows: a
        # champion that is merely unaffordable this turn is equally absent from
        # them, and "they still have their champion" is exactly the standing
        # fact that changes what a position is worth.
        g += [float(int(state.champion[seat]) >= 0),
              float(int(state.champion[foe]) >= 0)]
        # [Show Off] (RAD). Public both ways: revealing from hand is the price
        # of showing off, and a picked friendly unit was always visible. What
        # the policy needs is whether a unit was shown off and how big it is --
        # Primordial Roar deals damage equal to exactly that Might, so it is
        # the number that decides what the spell threatens.
        #
        # **The card's IDENTITY is not encoded, only its Might**, and that is
        # a real gap: a revealed hand card is public and a human would
        # remember which card it was, not just how large. Encoding it properly
        # means a row in the hand zone for a card the opponent does not hold,
        # which is a shape change; recorded in OBS_UNREAD.
        from rl.engine.resolve import show_off_might
        for sd in (seat, foe):
            # The two slots are read here rather than left to the helper, so
            # that "was a unit shown off" is computed from the same pair
            # `COND_SHOWED_OFF` tests -- and so the coverage gate in test_env,
            # which scans this file for `state.<field>`, can see that they are
            # encoded at all. A field read only through a helper in another
            # module looks unread to that scan, which is the whole point of it.
            perm = int(state.show_off_perm[sd])
            shown = int(state.show_off_card[sd])
            live = (int(state.show_off_ply[sd]) == ply
                    and (perm >= 0 or shown >= 0))
            g.append(float(live))
            g.append(float(show_off_might(state, self.table, sd)) / 8.0
                     if live else 0.0)
        asg = int(state.pend_dmg)
        g += [float(asg == seat),
              float(asg == foe),
              (float(int(state.pend_dmg_pool[seat])
                     - combat._dmg_spent(state, self.table, seat)) / 6.0
               if asg == seat else 0.0),
              float(int(state.pend_dmg_n_kill[seat])) / 4.0 if asg == seat else 0.0]
        return g

    def _globals(self, state: GameState, seat: int,
                 deck_rows: np.ndarray) -> np.ndarray:
        foe = 1 - seat
        cfg = self.cfg
        # The state field, not the cfg -- see `_standing` and BACKLOG #10.
        vs = float(state.victory_score)
        g: list[float] = [
            float(state.points[seat]) / vs,
            float(state.points[foe]) / vs,
            float(state.points[seat] - state.points[foe]) / vs,
            float(state.turn) / float(cfg.turn_cap),
        ]
        phase = [0.0] * len(PHASE_NAMES)
        phase[int(state.phase)] = 1.0
        g += phase
        g += [
            float(state.active == seat),
            float(state.priority == seat),
            float(state.is_open),
            float(state.showdown_bf >= 0),
            # ...and WHICH kind. A Combat Showdown ends in the damage step; the
            # Non-Combat one 344.2 opens ends in somebody Conquering (348.2.a).
            # Both present as "a showdown is open with me holding priority",
            # and the right play in them is completely different -- answer the
            # damage, or answer the Conquer -- so the policy needs the bit.
            float(bool(state.showdown_combat)),
            float(state.attacker == seat),
            float(state.attacker == foe),
            float(state.showdown_bf == 0),
            float(state.showdown_bf == 1),
            float(state.declaring),
            float(state.decl_dst == bf_loc(0)),
            float(state.decl_dst == bf_loc(1)),
            float(len(state.declared())) / 4.0,
            float(state.pend_play >= 0),
            float(state.n_hand[seat]) / 10.0,
            float(state.n_hand[foe]) / 10.0,
            float(state.n_deck[seat] - state.deck_ptr[seat]) / 30.0,
            float(state.n_deck[foe] - state.deck_ptr[foe]) / 30.0,
            float(state.n_trash[seat]) / 20.0,
            float(state.n_trash[foe]) / 20.0,
            float(state.burned_out[seat]),
            float(state.burned_out[foe]),
            float(state.rune_left[seat]) / float(cfg.rune_deck_size),
            float(state.rune_left[foe]) / float(cfg.rune_deck_size),
            float(state.pool_energy[seat]) / 5.0,
            # The Chain. Depth matters: an item five deep is respondable only
            # after the ones above it clear (340.4).
            float(state.n_chain) / 4.0,
            float(state.pend_slot >= 0 or state.pend_cost_kill >= 0
                  or state.pend_kill_play >= 0 or state.pend_tax >= 0
                  or state.pend_ask >= 0 or state.pend_group_loc >= 0),
            # Facedown zones. Presence is public (107.3.f), identity is not --
            # the *contents* stay out of the observation, which is what the
            # belief head will be asked to predict.
            float(any(state.fd_owner[k] == seat for k in range(N_FD))),
            float(any(state.fd_owner[k] == foe for k in range(N_FD))),
            float(any(state.fd_owner[k] == foe
                      and int(state.fd_ply[k]) < int(state.ply)
                      for k in range(N_FD))),
            # --- standing per-seat resources the engine reads and the
            # --- observation did not carry at all ---
            #
            # XP. `[Level N]` is "while you have N+ XP", 17 `COND_LEVEL` sites
            # in `effects.py` gate on it, and Scorchclaw's whole card is two
            # clauses keyed to it. The policy could not see the resource that
            # decides whether half its deck is on or off. Max observed 6.
            float(state.xp[seat]) / 3.0,
            float(state.xp[foe]) / 3.0,
            # [Legion] -- "if you have played another card this turn" (812).
            # Live on 53.1% of real decisions, and it is the single fact that
            # decides whether a Legion card in hand is worth its cost right
            # now. Both seats: whether the OPPONENT's Legion is on changes what
            # their open mana threatens.
            float(state.cards_played[seat]) / 4.0,
            float(state.cards_played[foe]) / 4.0,
            # A spell lockout is total while it lasts -- every spell in hand is
            # dead -- so it cannot be inferred from a shrunken action list
            # without first trying to act.
            float(state.no_spells[seat]),
            float(state.no_spells[foe]),
            # Elder Dragon: "any amount of your damage is enough to kill enemy
            # units" rewrites lethality board-wide, so every Might number above
            # means something different while it is set.
            float(bool(state.any_damage_kills)),
        ]
        # Runes are on the board face up, so both boards are public.
        for s in (seat, foe):
            g += list(state.runes_ready[s].astype(np.float32) / 4.0)
            g += list(state.runes_spent[s].astype(np.float32) / 4.0)
        g += list(state.pool_power[seat].astype(np.float32) / 3.0)
        # Runes promised at end of turn (Targon's Peak). Public -- the conquer
        # that banked them happened in the open -- and it changes what tapping
        # out costs, which is the single decision runes drive. Left out, the
        # whole point of the card ("delayed, so the refund lands before the
        # OPPONENT's turn") would be invisible to the policy.
        g += [float(state.pending_ready_runes[seat]) / 2.0,
              float(state.pending_ready_runes[foe]) / 2.0]
        # `_decks`, which masked pooling cannot carry on its own. The flags say
        # whether each registered list is known -- theirs is what the policy
        # conditions on, and MINE matters too, because a list the opponent has
        # seen has no surprise left in it. The counts say how many distinct
        # cards each deck row-set actually held, which is "how much of their
        # deck have I seen" and is lost to mean/max pooling.
        g += [float(state.deck_known[foe]),
              float(state.deck_known[seat]),
              float(deck_rows[0]) / float(DECK_SLOTS_PER_SEAT),
              float(deck_rows[1]) / float(DECK_SLOTS_PER_SEAT)]

        g += self._standing(state, seat)
        g += self._match_globals(state, seat)

        out = np.asarray(g, np.float32)
        assert out.size == GLOBAL_DIM, f"{out.size} globals, expected {GLOBAL_DIM}"
        return out

    def _match_globals(self, state: GameState, seat: int) -> list[float]:
        """Bo3 context (D2/D2b): 17 numbers, all zero in a Bo1 game.

        Everything here is either public or this seat's own: the score and game
        number, which pre-game decision is open, whether this seat plays first,
        how many sideboard swaps IT has made, the SIZE of both sideboards
        (403.6) and how many of each seat's battlefields are burned (the burned
        ones were played in the open). Never the opponent's swaps or pick.
        """
        foe = 1 - seat
        stage = [0.0] * 4
        stage[int(state.pg_stage)] = 1.0
        decided = float(int(state.pg_stage) == 0 or int(state.pg_stage) > 3)
        return [
            float(int(state.m_bo) > 1),
            float(state.m_game) / 3.0,
            float(state.m_wins[seat]) / 2.0,
            float(state.m_wins[foe]) / 2.0,
            float(state.m_last_draw),
            *stage,
            # Who goes first is only settled once the play-order choice is.
            float(int(state.first_player) == seat) * decided,
            decided,
            float(int(state.m_chooser) == seat),
            float(state.sb_swaps[seat]) / 10.0,
            float(state.n_side[seat]) / 10.0,
            float(state.n_side[foe]) / 10.0,
            float(3 - int(state.pg_bf_avail[seat].sum())) / 3.0
            if int(state.m_bo) > 1 else 0.0,
            float(3 - int(state.pg_bf_avail[foe].sum())) / 3.0
            if int(state.m_bo) > 1 else 0.0,
        ]

    # -- actions ---------------------------------------------------------

    def _open_slot_kind(self, state: GameState) -> int:
        """The TK_* kind of the target slot currently being filled."""
        return chain.open_slot_kind(state, self.table)

    def _action_row(self, act: A.Action, state: GameState,
                    seat: int) -> np.ndarray:
        r = np.zeros(self.act_dim, np.float32)
        nk = len(A.KIND_NAMES)
        r[act.kind] = 1.0

        card, loc, might = -1, -1, -1
        whose = -1.0
        mode = -1
        tag_extra = None
        pay_split = None
        perm = -1                         # the permanent this row is about
        k = act.kind
        if k == A.A_PLAY:
            # Nocturne's permission borrows this action while a look is
            # suspended, and its arg is a LOOK-BUFFER index there, not a hand
            # index -- the same overload `A_PICK` carries. Reading the hand
            # would have named some unrelated card, so the policy would have
            # been choosing "play Nocturne" off a feature describing whatever
            # happened to sit at that hand slot.
            # ...and 108.3.d adds a THIRD meaning: `CHAMPION_SRC` names the
            # Chosen Champion in the Champion Zone rather than any hand slot.
            # `played_card` is the one place that mapping lives.
            card = (int(state.look_cards[act.arg]) if state.pend_look >= 0
                    else A.played_card(state, seat, int(act.arg)))
        elif k in (A.A_PLAY_AT, A.A_PLAY_AT_FAST):
            loc = act.arg
            if state.rp_seat >= 0:
                card = int(state.rp_card)
            elif state.pend_hand_play >= 0 and int(state.hp_pick) >= 0:
                card = int(state.hand[int(state.pend_hand_play), int(state.hp_pick)])
            elif state.pend_play_look >= 0:
                card = int(state.look_cards[state.pend_play_look])
            elif state.pend_play >= 0:
                # `played_card` because 108.3.d makes the Champion Zone a
                # second source: `pend_play` may be `CHAMPION_SRC`, which is
                # deliberately out of range for the hand.
                card = A.played_card(state, seat, int(state.pend_play))
        elif k == A.A_DECLARE:
            loc = act.arg
        elif k == A.A_ACTIVATE:
            if A.is_legend_activate(int(act.arg)):
                # A legend activation names an ability index, not rows -- and
                # the Legend Zone is not a location (107.4.b), so there is no
                # `loc` to set. The card alone distinguishes the candidates
                # while a legend has one ability; the day one has two, the
                # index is what tells them apart and belongs here.
                card = int(state.legend[seat])
            else:
                # The arg may be a packed (permanent, donor) pair --
                # Heimerdinger borrows abilities, so the row alone no longer
                # says which one. The DONOR's card is what distinguishes the
                # candidates; the location is the activating permanent's.
                _perm, _donor, _k = A.unpack_activate(int(act.arg))
                card = int(state.perms[_donor, P_CARD])
                loc = int(state.perms[_perm, P_LOC])
                perm = int(_perm)
        elif k in (A.A_ADD, A.A_RETREAT):
            # The location a unit is *leaving*; the destination is fixed for the
            # whole decision, so it could not discriminate between candidates.
            card = int(state.perms[act.arg, P_CARD])
            loc = int(state.perms[act.arg, P_LOC])
            might = combat.might(state, self.table, act.arg)
            perm = int(act.arg)
        elif k == A.A_COMMIT:
            loc = int(state.decl_dst)
        elif k == A.A_HIDE:
            card = int(state.hand[seat, act.arg])
        elif k == A.A_HIDE_AT:
            loc = bf_loc(fd_bf(act.arg))          # the arg is a SLOT
        elif k == A.A_PLAY_FLOW:
            # The trash is public information (108.5), so naming the card here
            # leaks nothing.
            card = int(state.trash[seat, act.arg])
        elif k == A.A_PLAY_HIDDEN:
            # The card's identity is legitimate here: only its owner is ever
            # offered this action, and they know what they hid.
            card = int(state.fd_card[act.arg])
            loc = bf_loc(fd_bf(act.arg))          # the arg is a SLOT
        elif k == A.A_PICK and state.pend_show_off >= 0:
            # [Show Off]: the arg is a HAND INDEX, and the whole decision is
            # which unit -- so naming the card is the only thing that tells the
            # candidates apart. It leaks nothing: only the seat holding the
            # hand is ever offered this, and choosing it reveals the card
            # anyway.
            card = int(state.hand[seat, act.arg])
        elif k == A.A_PICK and state.pend_dmg >= 0:
            # Combat damage assignment: the arg is a PERMANENT ROW, not any
            # buffer index, so `pick_card` would describe the wrong card
            # entirely -- the same class of bug as `A_TARGET`'s below, where a
            # location arg was read as a permanent row. What discriminates the
            # candidates is the unit itself: what it is, where it stands, and
            # how much Might it takes off the board when it dies.
            card = int(state.perms[act.arg, P_CARD])
            loc = int(state.perms[act.arg, P_LOC])
            might = combat.might(state, self.table, act.arg)
            perm = int(act.arg)
        elif k == A.A_PICK:
            # The whole decision is WHICH card, so naming it is the only
            # feature that could discriminate between the candidates -- and it
            # leaks nothing in any of the three cases: the look buffer is this
            # player's own cards off their own deck, the revealed hand was
            # revealed to them by a card that says so, and a discard is out of
            # their own hand. Only the deciding seat is ever offered the
            # action.
            #
            # Which of the three the arg indexes is `pick_card`'s business, not
            # this encoder's -- see the note there for why that must be one
            # shared answer.
            card = A.pick_card(state, act.arg)
            # 372's replacement choice: the arg is a `combat.RK_*` kind, not a
            # card, and without this every option encoded IDENTICALLY -- the
            # policy could not tell "save it" from "banish it". The four kinds
            # fill the four mode slots exactly (a refusal is an A_DECLINE).
            if state.pend_repl >= 0:
                mode = int(act.arg)
            if state.pend_amount >= 0 or (state.pend_name >= 0
                                          and int(state.name_kind) == 0):
                mode = int(act.arg)             # the amount / tag IS the choice
            if state.pend_name >= 0 and int(state.name_kind) == 0:
                tag_extra = self._tag_counts(state, seat,
                                             int(state.name_opts[act.arg]))
        elif k == A.A_TARGET and state.pend_show_off >= 0:
            # ...and here the arg is a friendly PERMANENT ROW, chosen rather
            # than targeted, so there is no chain slot for `_open_slot_kind`
            # to read. Reading one would describe an unrelated card.
            card = int(state.perms[act.arg, P_CARD])
            loc = int(state.perms[act.arg, P_LOC])
            might = combat.might(state, self.table, act.arg)
            perm = int(act.arg)
        elif k == A.A_TARGET:
            # **`arg` means whatever the open slot's KIND says it means**: a
            # permanent row, a location, or a Chain Item uid. This read
            # `state.perms[act.arg]` unconditionally, so a location target
            # (Sprite Call, Ride The Wind) described permanent row 0-3 and a
            # counterspell target (Defy, Lilting Lullaby) described a row
            # chosen by a monotonic counter. Valid indices, meaningless
            # features -- on exactly the decisions that need them most.
            kind = self._open_slot_kind(state)
            if kind == TK_LOCATION:
                loc = int(act.arg)
            elif kind == TK_SPELL:
                i = chain.index_of_uid(state, int(act.arg))
                if i >= 0:
                    card = int(state.chain[i, C_CARD])
            elif kind == TK_MODE:
                # "Choose one --": the card is the one on the Chain being
                # announced, and the mode index is what tells candidates apart.
                item = chain.oldest_pending(state)
                if item >= 0:
                    card = int(state.chain[item, C_CARD])
                mode = int(act.arg)
            elif kind == TK_TRASH_CARD:
                # A packed (owner, card); the trash is public (108.2), so
                # naming it leaks nothing. There is no row and no location to
                # describe -- the choice is "which card, out of whose pile".
                owner, card = unpack_trash(int(act.arg))
                whose = 1.0 if owner == seat else 0.0
            elif is_legend_src(int(act.arg)):
                # Profiteer's "a legend": no row, so the card is all there is.
                card = int(state.legend[legend_src_seat(int(act.arg))])
                whose = 1.0 if legend_src_seat(int(act.arg)) == seat else 0.0
            else:
                card = int(state.perms[act.arg, P_CARD])
                loc = int(state.perms[act.arg, P_LOC])
                might = combat.might(state, self.table, act.arg)
                perm = int(act.arg)
        elif k == A.A_PAY_SPLIT:
            # Which runes pay for `pay_card`: the card says what is being
            # bought, the split says what is given up for it.
            card = int(state.pay_card)
            pay_split = state.pay_opts[int(act.arg)]
        elif k in (A.A_SB_OUT, A.A_SB_IN):
            card = int(act.arg)                  # a card id, the seat's own
        elif k == A.A_PICK_BF:
            card = int(state.pg_bf_reg[seat, int(act.arg)])
        elif k in (A.A_ACCEPT, A.A_DECLINE):
            # 383.3.a -- the choice is about one triggered ability, so the
            # candidates differ only by yes/no. What distinguishes the decision
            # is whose ability it is.
            #
            # ACCEPT/DECLINE also answers Hard Bargain's "unless its controller
            # pays" tax, and that decision is about a SPELL on the Chain, not
            # about a pending ability. Read `pend_may` there and the row would
            # describe -1, so the two candidates would be identical and the
            # policy would be choosing whether to save its own spell with no
            # feature saying which spell, or what refusing costs.
            #
            # The threatened spell is named; what refusing COSTS is not. The
            # amount is `pend_tax_cost`, and it is 2 on the only card in the
            # pool that taxes, so a column for it would be a constant -- and
            # the row already carries that spell's own Energy and Power, which
            # is the half of "is this worth 2 energy" that actually varies. A
            # second taxing card at a different price is what should add the
            # column, and it should be a real one rather than an overload of
            # the card-cost slot below.
            if state.rp_seat >= 0:
                card = int(state.rp_card)
            if state.pend_ask >= 0:
                card = int(state.pend_ask_card)
                subj_a = int(state.pend_ask_subj)
                if subj_a >= 0:
                    loc = int(state.perms[subj_a, P_LOC])
            if state.pend_tax >= 0:
                j = chain.index_of_uid(state, int(state.pend_tax_uid))
                if j >= 0:
                    card = int(state.chain[j, C_CARD])
            item = int(state.pend_may)
            if item >= 0:
                card = int(state.chain[item, C_CARD])
                src = int(state.chain[item, C_SRC])
                if src >= 0:
                    loc = int(state.perms[src, P_LOC])
                    perm = src

        if card >= 0:
            r[nk] = 1.0
            r[nk + 1:nk + 1 + self.card_dim] = self.cards[card]
        o = nk + 1 + self.card_dim

        L = N_LOCATIONS
        if loc >= 0:
            r[o + _loc_slot(loc, seat)] = 1.0
            r[o + L] = float(is_battlefield(loc))
            if is_battlefield(loc):
                i = bf_index(loc)
                ctrl = int(state.bf_ctrl[i])
                r[o + L + 1 + (2 if ctrl < 0 else 0 if ctrl == seat else 1)] = 1.0
                r[o + L + 4] = state.units_at(loc, 1 - seat).size / 4.0
                r[o + L + 5] = state.units_at(loc, seat).size / 4.0
                r[o + L + 6] = float(state.bf_contested[i])
        if might >= 0:
            r[o + L + 7] = might / 5.0
        if card >= 0:
            r[o + L + 8] = float(self.table.energy[card]) / 5.0
            r[o + L + 9] = float(self.table.power[card]) / 3.0
        if whose >= 0.0:
            r[o + L + 10] = whose
        if 0 <= mode < MAX_MODES:
            r[o + L + 11 + mode] = 1.0
        if tag_extra is not None:
            r[o + L + 11 + MAX_MODES] = tag_extra[0]
            r[o + L + 12 + MAX_MODES] = tag_extra[1]
        if pay_split is not None:
            for d in range(N_DOMAINS):
                r[o + L + 13 + MAX_MODES + d] = float(pay_split[d]) / 3.0
        if 0 <= perm < int(state.n_perms):
            q = o + L + 13 + MAX_MODES + N_DOMAINS
            r[q] = float(int(state.perms[perm, P_CTRL]) == seat)
            sb = int(state.showdown_bf)
            r[q + 1] = float(sb >= 0 and int(state.perms[perm, P_LOC]) == bf_loc(sb))
        return r

    def _tag_counts(self, state: GameState, seat: int,
                    tag_idx: int) -> tuple[float, float]:
        """(enemy, friendly) units on the board carrying tag `tag_idx`, /4."""
        from rl.engine.effects import tag_vocab
        tag = tag_vocab(self.table)[tag_idx]
        n = [0, 0]
        for i in range(state.n_perms):
            if (state.perms[i, P_ALIVE] == 1
                    and not state.has_flag(i, F_NON_UNIT)
                    and tag in combat.perm_tags(state, self.table, i)):
                n[int(int(state.perms[i, P_CTRL]) == seat)] += 1
        return n[0] / 4.0, n[1] / 4.0

    def _actions(self, legal: list[A.Action], state: GameState, seat: int):
        n = len(legal)
        assert n <= self.a_max, (
            f"{n} legal actions exceeds max_actions={self.a_max}; raise the cap "
            f"in Config after checking the distribution")
        z = np.zeros((self.a_max, self.act_dim), np.float32)
        m = np.zeros(self.a_max, bool)
        for i, act in enumerate(legal):
            z[i] = self._action_row(act, state, seat)
            m[i] = True
        return z, m

    # -- privileged ------------------------------------------------------

    def _privileged(self, state: GameState, seat: int) -> np.ndarray:
        """Exactly the information the policy is denied: the opponent's hand and
        every facedown card. Kept in its own field so a leak is a type error
        rather than an archaeology problem."""
        foe = 1 - seat
        out = np.zeros((HAND_SLOTS + N_FD, self.card_dim), np.float32)
        n = min(int(state.n_hand[foe]), HAND_SLOTS)
        for j in range(n):
            out[j] = self.cards[int(state.hand[foe, j])]
        for k in range(N_FD):
            if state.fd_owner[k] >= 0:
                out[HAND_SLOTS + k] = self.cards[int(state.fd_card[k])]
        return out.reshape(-1)

    # -- entry point -----------------------------------------------------

    def encode(self, state: GameState, seat: int,
               legal: list[A.Action]) -> Obs:
        builders = {"hand": self._hand, "board": self._board,
                    "battlefields": self._battlefields,
                    "facedown": self._facedown, "legends": self._legends,
                    "chain": self._chain, "decks": self._decks,
                    "champions": self._champions, "sideboard": self._sideboard}
        assert set(builders) == set(ZONES), "a zone has no builder"
        zones, masks = {}, {}
        for name in ZONES:
            if name == "decks":
                # The one builder that also returns a count, because pooling
                # destroys it and `_globals` needs it. Kept explicit rather
                # than stashed on `self`.
                zones[name], masks[name], deck_rows = self._decks(state, seat)
            else:
                zones[name], masks[name] = builders[name](state, seat)
        acts, amask = self._actions(legal, state, seat)
        return Obs(
            zones=zones,
            zone_mask=masks,
            globals=self._globals(state, seat, deck_rows),
            legal_actions=acts,
            action_mask=amask,
            to_move=seat,
            n_legal=len(legal),
            privileged=(self._privileged(state, seat)
                        if self.want_privileged else None),
        )

    def shapes(self) -> dict[str, tuple]:
        """Everything a network builder needs to size its layers."""
        return {
            # The names the network must pool over, in order. See `ZONES`.
            "zones": ZONES,
            "row_dim": self.row_dim,
            "card_dim": self.card_dim,
            "hand": (HAND_SLOTS, self.row_dim),
            "board": (BOARD_SLOTS, self.row_dim),
            "battlefields": (N_BF, self.row_dim),
            "facedown": (N_FD, self.row_dim),
            "legends": (N_SEATS, self.row_dim),
            "chain": (CHAIN_SLOTS, self.row_dim),
            "decks": (DECK_SLOTS, self.row_dim),
            "champions": (CHAMP_SLOTS, self.row_dim),
            "sideboard": (SIDE_SLOTS, self.row_dim),
            "globals": (GLOBAL_DIM,),
            "actions": (self.a_max, self.act_dim),
            "privileged": (self.priv_dim,),
        }
