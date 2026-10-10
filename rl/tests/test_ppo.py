"""Network and PPO tests -- the first three items on PLAN.md's Phase 4 debug list.

When PPO does not learn, §4 says to check, in order: (1) action mask alignment,
(2) reward sign and seat attribution, (3) observation canonicalization, (4)
advantage normalization, (5) learning rate -- and that it is one of the first
three about 90% of the time. Canonicalization is covered in `test_env.py`; the
other two are covered here, plus the pieces that would silently poison a run
rather than crash it.

  [1] Mask alignment -- index i really is `env.legal[i]`, end to end.
  [2] Masked logits -- illegal slots get zero probability, and no NaNs.
  [3] Seat attribution and GAE -- closed-form, hand-checkable.
  [4] Entropy normalization -- invariant to the number of legal actions.
  [5] Gradients -- finite, and reaching every parameter.
  [6] An untrained net is ~50% against random, so the eval harness is sane.
  [7] Hidden information reaches neither the policy nor the shared trunk.
  [8] The procedural dealer is as REDUNDANT as a real decklist.
 [10] Matchup-aware PFSP: the unflagged weights, a losing cell, an
      all-empty matrix, the per-cell decay and the resume round-trip.
 [11] ...and the joint choice reaches the slot it was dealt to.
 [16] The update on the Apple GPU (`update_device="mps"`) matches the CPU one,
      leaves acting on the CPU, and its optimizer state survives a resume.
"""
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from rl.config import Config
from rl.engine import game
from rl.engine.cardtable import full_table
from rl.env import RiftboundEnv
from rl.eval import duel
from rl.nets import RiftboundNet, count_params, to_torch
from rl.obs import Encoder
from rl.ppo import (HP, MAIN_DECK_SIZE, Step, Trainer, deal_stats,
                    deck_pool_deal, finish_episode, v0_deal, v1_deal)
from rl.vec import batch

T = full_table()
CFG = Config().at_victory_score(3)
DEAL = v0_deal(T)


def ok(name):
    print(f"  \033[32mPASS\033[0m {name}")


def die(name, msg):
    print(f"  \033[31mFAIL\033[0m {name}: {msg}")
    sys.exit(1)


enc = Encoder(T, CFG)
net = RiftboundNet(enc.shapes())
print(f"net: {count_params(net):,} parameters")

# ---------------------------------------------------------------------------
print("\n[1] action index i is env.legal[i], through the whole stack")
checked = 0
for seed in range(30):
    env = RiftboundEnv(T, CFG, encoder=enc)
    obs = env.reset(seed, *DEAL(seed))
    rng = np.random.default_rng(seed)
    while obs is not None:
        # The row the net scores at slot i must be the featurization of the
        # engine action the env will apply for i. Anything else and the policy
        # is learning to score one action and play another -- which trains
        # perfectly happily and never improves.
        for i, act in enumerate(env.legal):
            want = enc._action_row(act, env.state, obs.to_move)
            if not np.array_equal(obs.legal_actions[i], want):
                die("alignment", f"seed {seed}: row {i} != featurized "
                                 f"{act!r}")
        checked += len(env.legal)
        obs = env.step(int(rng.integers(obs.n_legal))).obs
ok(f"{checked} action rows matched their engine action exactly")

# And the sampler never returns a padded slot.
env = RiftboundEnv(T, CFG, encoder=enc)
obs = env.reset(0, *DEAL(0))
gen = torch.Generator().manual_seed(0)
for _ in range(300):
    if obs is None:
        obs = env.reset(1, *DEAL(1))
    t = to_torch(batch([obs]), "cpu")
    idx, logp, ent, val = net.act(t, generator=gen)
    i = int(idx.item())
    if not 0 <= i < obs.n_legal:
        die("alignment", f"sampled index {i} outside {obs.n_legal} legal actions")
    obs = env.step(i).obs
ok("300 samples all landed inside the legal prefix")


# ---------------------------------------------------------------------------
print("\n[2] masked logits carry no probability and no NaNs")
env = RiftboundEnv(T, CFG, encoder=enc)
obs = env.reset(3, *DEAL(3))
t = to_torch(batch([obs] * 4), "cpu")
logits, value, value_aux = net(t)     # two critics since the symmetric head
p = torch.softmax(logits, -1)
n = obs.n_legal
assert torch.isfinite(logits).all(), "non-finite logit"
assert torch.isfinite(value).all(), "non-finite value"
if p[:, n:].abs().max().item() > 1e-8:
    die("mask", f"illegal slots hold {p[:, n:].abs().max().item():.2e} probability")
assert abs(p[:, :n].sum(-1).mean().item() - 1.0) < 1e-5, "legal mass != 1"
ok(f"{p.shape[1] - n} padded slots hold <1e-8 probability; legal mass sums to 1")


# ---------------------------------------------------------------------------
print("\n[3] seat attribution and GAE inside a seat's own subsequence")
hp = HP(gamma=1.0, gae_lambda=0.95)

# Interleaved seats, zero value estimates: advantages are then pure discounted
# terminal reward, so the arithmetic is checkable by hand.
pend = [Step(obs={}, action=0, logp=0.0, value=0.0, seat=s)
        for s in (0, 1, 0, 1, 0, 1)]
finish_episode(pend, (1.0, -1.0), hp)
s0 = [s.adv for s in pend if s.seat == 0]
s1 = [s.adv for s in pend if s.seat == 1]
want0 = [0.95 ** 2, 0.95, 1.0]
if not np.allclose(s0, want0):
    die("gae", f"seat 0 advantages {s0}, expected {want0}")
if not np.allclose(s1, [-x for x in want0]):
    die("gae", f"seat 1 advantages {s1}, expected {[-x for x in want0]}")
ok("winner's advantages are positive, loser's are their exact negation")

# The opponent's value estimates must not enter a seat's bootstrap. Give seat 1
# wildly wrong values and check seat 0's advantages do not move at all.
pend2 = [Step(obs={}, action=0, logp=0.0, value=(0.0 if s == 0 else 99.0), seat=s)
         for s in (0, 1, 0, 1, 0, 1)]
finish_episode(pend2, (1.0, -1.0), hp)
if not np.allclose([s.adv for s in pend2 if s.seat == 0], want0):
    die("gae", "seat 0's advantages changed when seat 1's critic changed -- "
               "GAE is bootstrapping across the opponent's moves")
ok("seat 0's GAE is unaffected by seat 1's value estimates (gotcha 2)")

# A game nobody won pays nobody.
pend3 = [Step(obs={}, action=0, logp=0.0, value=0.0, seat=s) for s in (0, 1, 0)]
finish_episode(pend3, (0.0, 0.0), hp)
if any(s.adv != 0.0 for s in pend3):
    die("gae", "a truncated game produced non-zero advantages")
ok("a truncated game pays zero to both seats")

# One seat acting alone (possible when a game ends before the other replies).
solo = [Step(obs={}, action=0, logp=0.0, value=0.5, seat=0)]
finish_episode(solo, (1.0, -1.0), hp)
assert np.isclose(solo[0].ret, 1.0), f"single-step return {solo[0].ret}"
ok("a one-decision episode returns exactly the terminal reward")


# ---------------------------------------------------------------------------
print("\n[4] entropy is normalized by log(n_legal)")
for n in (2, 3, 5, 10, 40):
    lg = torch.zeros(1, enc.a_max)
    mask = torch.zeros(1, enc.a_max, dtype=torch.bool)
    mask[0, :n] = True
    e = RiftboundNet.entropy(lg.masked_fill(~mask, -1e9), mask).item()
    if abs(e - 1.0) > 1e-5:
        die("entropy", f"uniform over {n} actions gave {e:.4f}, expected 1.0")
ok("a uniform policy scores 1.0 whether it has 2 legal actions or 40")

lg = torch.zeros(1, enc.a_max)
lg[0, 0] = 20.0
mask = torch.zeros(1, enc.a_max, dtype=torch.bool)
mask[0, :5] = True
assert RiftboundNet.entropy(lg.masked_fill(~mask, -1e9), mask).item() < 0.05, \
    "a near-deterministic policy should score near 0"
ok("a near-deterministic policy scores near 0")


# ---------------------------------------------------------------------------
print("\n[5] gradients are finite and reach every parameter")
tr = Trainer(T, CFG, DEAL, HP(n_envs=8, rollout=64, minibatches=2,
                              update_epochs=1), seed=0)
roll = tr.collect()
assert roll.episodes > 0, "no episode completed in a 64-transition rollout"
before = [p.detach().clone() for p in tr.net.parameters()]
stats = tr.update(roll, 0.0)
for k, v in stats.items():
    if not np.isfinite(v):
        die("grad", f"stat {k} is {v}")
moved = sum(not torch.equal(a, b)
            for a, b in zip(before, tr.net.parameters()))
total = len(before)
if moved < total:
    stuck = [n for (n, p), b in zip(tr.net.named_parameters(), before)
             if torch.equal(p.detach(), b)]
    die("grad", f"{total - moved}/{total} parameter tensors never moved: {stuck}")
ok(f"{roll.episodes} episodes, {len(roll.steps)} transitions, "
   f"all {total} parameter tensors updated")
print(f"    stats: {({k: round(v, 4) for k, v in stats.items()})}")

# Awkward transition counts must not produce a size-1 minibatch, whose
# advantage std is undefined. That NaN'd the policy loss silently, spread to
# every parameter through one optimizer step, and did not raise.
for nmb in (2, 3, 4, 7):
    t2 = Trainer(T, CFG, DEAL, HP(n_envs=4, rollout=17, minibatches=nmb,
                                  update_epochs=1), seed=1)
    st = t2.update(t2.collect(), 0.0)
    for k, v in st.items():
        if not np.isfinite(v):
            die("grad", f"minibatches={nmb} produced {k}={v}")
ok("odd transition counts split evenly and stay finite for 2/3/4/7 minibatches")


# ---------------------------------------------------------------------------
print("\n[6] an untrained net is a coin flip against random")
rate = duel(net, game.random_agent, T, CFG, DEAL, n=60)
if not 0.35 <= rate <= 0.65:
    die("eval", f"untrained net scored {rate:.1%} vs random -- the eval harness "
                f"or the seat swap is wrong")
ok(f"untrained net {rate:.1%} vs random (seat-swapped, so ~50% is correct)")


# ---------------------------------------------------------------------------
# The claim: the agent cannot learn to play by cheating. Two halves, and both
# are architectural rather than conventional, so both can be tested directly.
print("\n[7] hidden information cannot reach the policy, or the trunk")

env = RiftboundEnv(T, CFG, encoder=enc)
t = to_torch(batch([env.reset(11, *DEAL(11))]), "cpu")
assert "privileged" in t, "the test needs the privileged field to perturb"

# (a) Rewriting what ONLY the critic sees must not move a single action score.
# `test_env` proves the observation excludes hidden information; this proves
# the network has no second path to it.
base_logits, _, _ = net(t)
cheat = {k: (v if k != "privileged" else torch.randn_like(v))
         for k, v in t.items()}
cheat_logits, _, _ = net(cheat)
if not torch.equal(base_logits, cheat_logits):
    die("cheating", "scrambling the privileged input changed the action "
                    "logits -- hidden information reaches the policy")
ok("scrambling the opponent's hand and facedowns leaves every action score identical")

# ...and the perturbation is not a no-op, or (a) proves nothing.
if torch.equal(net(t)[2], net(cheat)[2]):
    die("cheating", "the privileged critic did not move either -- the "
                    "perturbation was inert and the test above is vacuous")
ok("...while the privileged critic does move, so the control is live")

# (b) The spare critic is fitted to the same returns, so its gradient exists --
# but it must not reach the shared trunk. Otherwise switching the privileged
# critic off would still let hidden-information gradients shape the features
# the policy reads, which is the entire point of switching it off.
net.zero_grad(set_to_none=True)
_, _, aux = net(t)
aux.sum().backward()
trunk_grad = [p.grad for p in net.trunk.parameters() if p.grad is not None]
enc_grad = [p.grad for p in net.card_enc.parameters() if p.grad is not None]
leaked = [g for g in trunk_grad + enc_grad if g.abs().sum().item() > 0.0]
if leaked:
    die("cheating", f"the auxiliary critic put gradient on {len(leaked)} shared "
                    f"trunk tensors -- `values` is not detaching it")
ok("the auxiliary critic's gradient stops at its own head; the trunk is untouched")

head_grad = sum(p.grad.abs().sum().item()
                for p in net.value_priv.parameters() if p.grad is not None)
if head_grad <= 0.0:
    die("cheating", "the auxiliary critic got no gradient at all, so the "
                    "detach test above is vacuous")
ok("...and its own weights do get gradient, so it is genuinely being trained")
net.zero_grad(set_to_none=True)

# ---------------------------------------------------------------------------
print("\n[8] the procedural dealer builds decks, not samples of a pool")
# `v1_deal` drew 39 cards one at a time from a 339-card pool and rejected a 4th
# copy, so a repeat was an accident: 36.5 distinct cards of 39, against 16.7 for
# the real corpus. Nothing failed -- and nothing tested it -- but the `decks`
# observation zone exists to condition on draw consistency, on what a trash
# implies about what is left, and on archetype, and a near-singleton deck
# misrepresents all three. Measured against the corpus, not against a guess.
_real = deal_stats(T, deck_pool_deal(T, 0.0), 120)
_proc = deal_stats(T, v1_deal(T), 300)
print(f"    real  decks: {_real['distinct_mean']:.1f} distinct of "
      f"{MAIN_DECK_SIZE}, copies 1/2/3 = {_real['copy_frac']}")
print(f"    v1_deal    : {_proc['distinct_mean']:.1f} distinct of "
      f"{MAIN_DECK_SIZE}, copies 1/2/3 = {_proc['copy_frac']}")

if _proc["max_copies"] > 3:
    die("dealer", f"103.2.b -- {_proc['max_copies']} copies of one card")
# Within 3 of the corpus mean. Loose on purpose: the target is "a deck shaped
# like a deck", not a specific number, and the spell/unit split forces two
# truncations per deck which leaves a small excess of 1-ofs.
if abs(_proc["distinct_mean"] - _real["distinct_mean"]) > 3.0:
    die("dealer", f"v1_deal runs {_proc['distinct_mean']:.1f} distinct cards "
                  f"against the corpus' {_real['distinct_mean']:.1f} -- decks "
                  f"are not redundant like real lists")
# The shape, not just the count: most of a real deck's distinct cards are maxed
# out. A dealer could hit the mean with all-2-ofs and still be wrong.
if _proc["copy_frac"][2] < 0.35:
    die("dealer", f"only {_proc['copy_frac'][2]:.0%} of distinct cards are "
                  f"3-ofs, against {_real['copy_frac'][2]:.0%} for real decks")
ok(f"{_proc['distinct_mean']:.1f} distinct of {MAIN_DECK_SIZE} "
   f"(corpus {_real['distinct_mean']:.1f}), {_proc['copy_frac'][2]:.0%} of them 3-ofs")

# The reason the old dealer was not simply replaced by the real one: spell
# density is what `v1_deal` is FOR, and it has to survive the rewrite.
if not 0.28 <= _proc["spell_frac_mean"] <= 0.67:
    die("dealer", f"spell density {_proc['spell_frac_mean']:.0%} left the "
                  f"real range 28-67%")
if _proc["deck_size"] != [MAIN_DECK_SIZE]:
    die("dealer", f"deck sizes {_proc['deck_size']}, not {MAIN_DECK_SIZE}")
ok(f"...at {_proc['spell_frac_mean']:.0%} mean spell density, every deck exactly "
   f"{MAIN_DECK_SIZE} cards")



# ---------------------------------------------------------------------------
# The coverage METRIC, not the engine. `plays_as_printed` is what
# `--min-coverage` filters the training pool on, so a false negative silently
# shrinks the pool and a false positive silently admits a deck whose cards do
# nothing. Both directions are asserted here.
print("\n[9] the coverage guard reads every registry a card can be scripted in")

from rl import decks as _D                                        # noqa: E402
from rl.engine import effects as _E                               # noqa: E402

_t = full_table()
_bf = [c for c in range(_t.n) if _t.is_type(c, "Battlefield")]
_lg = [c for c in range(_t.n) if _t.is_type(c, "Legend")]
_bad_bf = [_t.names[c] for c in _bf if not _D.plays_as_printed(_t, c)]
_bad_lg = [_t.names[c] for c in _lg if not _D.plays_as_printed(_t, c)]
# All 66 battlefields and all 49 legends are scripted. 15 battlefields live in
# registries that are neither BF_STATICS nor BF_ABILITIES -- cost rules, score
# rules, play costs, death replacements, extra hides, move permissions, granted
# abilities -- and 2 legends carry an [Empower] cost in LEGEND_ABILITIES. The
# guard once consulted the two ability tables only and reported all 17 as
# uncovered.
if _bad_bf or _bad_lg:
    die("coverage", f"scripted cards read as uncovered: "
                    f"battlefields {_bad_bf}, legends {_bad_lg}")
ok(f"all {len(_bf)} battlefields and {len(_lg)} legends count as covered")

# ...and the credit is registry membership, not forgiveness. Take a battlefield
# out of the only table that holds it and it must fail again.
_name = "Marai Spire"
_rule = _E.BF_COST_RULES.pop(_name)
try:
    if _D.plays_as_printed(_t, _t.names.index(_name)):
        die("coverage", f"{_name} counts as covered with its cost rule removed "
                        f"-- the guard is forgiving the name, not reading a table")
finally:
    _E.BF_COST_RULES[_name] = _rule
ok("...and a battlefield whose registry entry is removed fails again")

# The [Empower] credit on a ground comes from the rule's own `kind`. Risen
# Altar prints "[Empower] costs ... cost less" and does not HAVE the keyword;
# the discount is what reads it. A rule of any other kind must not credit it.
_name = "Risen Altar"
_rule = _E.BF_COST_RULES[_name]
_E.BF_COST_RULES[_name] = dict(_rule, kind="card")
try:
    if _D.plays_as_printed(_t, _t.names.index(_name)):
        die("coverage", f"{_name}'s [Empower] is credited by a cost rule that "
                        f"has nothing to do with Empower")
finally:
    _E.BF_COST_RULES[_name] = _rule
ok("...and the keyword credit tracks the cost rule's kind, not the card's name")

# `coverage` must still mean the MAIN DECK only -- it is the training filter --
# with the battlefield half reported alongside rather than folded in.
_deck = max(_D.load_all(_t), key=lambda d: d.coverage)
_main_ok = sum(1 for c in _deck.main if _D.plays_as_printed(_t, c))
if abs(_deck.coverage - _main_ok / len(_deck.main)) > 1e-9:
    die("coverage", "`coverage` is no longer the main deck's as-printed share")
if "bf " not in _deck.report():
    die("coverage", "`report` quotes coverage without quoting bf_coverage")
ok(f"coverage stays main-deck-only ({_deck.coverage:.0%}) and report carries "
   f"bf {_deck.bf_coverage:.0%}")

# ---------------------------------------------------------------------------
print("\n[10] matchup-aware PFSP (BACKLOG #21)")

from dataclasses import replace as _dc_replace

from rl.pool import Entry, OpponentPool
from rl.ppo import MatchupDealer, select_decks


def _mkpool(matchup, n=4):
    pl = OpponentPool(capacity=20, p=2.0, matchup=matchup)
    for i in range(n):
        pl.entries.append(Entry(name=f"iter{i}", state_dict={}, iteration=i))
    return pl


# (a) The UNFLAGGED pool must weight opponents exactly as it did before. The
# old formula is reproduced here by hand rather than imported, so a future edit
# to `weights()` fails this instead of agreeing with itself.
_rng = np.random.default_rng(0)
_off, _on = _mkpool(False), _mkpool(True)
_hist = [(0, True), (0, True), (1, False), (2, True), (2, False), (3, False)]
for _p in (_off, _on):
    for _i, _won in _hist:
        _p.record(_p.entries[_i], learner_won=_won, deck_key="deck-A")
_hand = np.array([(1.0 - (e.wins + 1.0) / (e.games + 2.0)) ** 2.0
                  for e in _off.entries])
_hand /= _hand.sum()
if not np.allclose(_off.weights(), _hand, atol=0, rtol=1e-12):
    die("pfsp", f"per-opponent weights changed: {_off.weights()} vs {_hand}")
if any(e.cells for e in _off.entries):
    die("pfsp", "an unflagged pool recorded per-matchup cells; it must ignore "
                "the deck key entirely")
if not np.allclose(_off.weights(), _on.weights(), atol=0, rtol=1e-12):
    die("pfsp", "turning the flag on changed the per-opponent weights")
# ...and the same rng gives the same opponent sequence with the flag off.
_a = [_off.sample(np.random.default_rng(7)).name for _ in range(1)]
_b = [_off.sample(np.random.default_rng(7), deck_key="deck-A").name
      for _ in range(1)]
if _a != _b:
    die("pfsp", "a deck key changed the draw on a non-matchup pool")
ok("unflagged: weights match the old formula exactly, and no cells are written")

# (c) ALL-EMPTY degrades to uniform, not to one cell. (Checked before the
# losing-cell case, because this is the state the matrix spends most of its
# life in: 20 x 47 cells and a few hundred recorded games.)
_e = _mkpool(True, n=5)
_keys = [f"d{i}" for i in range(9)]
_w = _e.cell_weights(_keys)
if _w.shape != (5, 9):
    die("pfsp", f"cell_weights shape {_w.shape}")
if not np.allclose(_w, 1.0 / _w.size, atol=0, rtol=1e-12):
    die("pfsp", f"an empty matrix is not uniform: min {_w.min()} max {_w.max()}")
# and the marginals are what the unflagged code would have used
if not np.allclose(_w.sum(axis=1), _e.weights(), atol=0, rtol=1e-12):
    die("pfsp", "the opponent marginal of an empty matrix is not the "
                "unflagged per-opponent distribution")
_draws = [_e.sample_cell(np.random.default_rng(s), _keys) for s in range(200)]
if len({k for _, k in _draws}) < 7 or len({o.name for o, _ in _draws}) < 4:
    die("pfsp", f"an empty matrix collapsed onto few cells: "
                f"{len({k for _, k in _draws})} decks, "
                f"{len({o.name for o, _ in _draws})} opponents")
ok(f"all-empty matrix is exactly uniform over {_w.size} cells, and its "
   f"opponent marginal is the unflagged distribution")

# (b) A cell the learner LOSES must gain weight -- and specifically it must gain
# weight the per-opponent average cannot see, so the loser's opponent is given
# an offsetting winning record in another deck.
_m = _mkpool(True, n=3)
_keys = ["deck-A", "deck-B", "deck-C"]
_hard = (_m.entries[1], "deck-B")
for _ in range(10):
    _m.record(_hard[0], learner_won=False, deck_key=_hard[1])
    _m.record(_hard[0], learner_won=True, deck_key="deck-A")
_w = _m.cell_weights(_keys)
_flat = int(np.argmax(_w))
if (_flat // 3, _flat % 3) != (1, 1):
    die("pfsp", f"the losing cell is not the heaviest; argmax at {_flat}, "
                f"weights\n{_w}")
if _w[1, 1] <= 2.0 / _w.size:
    die("pfsp", f"the losing cell gained less than 2x uniform: {_w[1, 1]:.4f} "
                f"vs uniform {1 / _w.size:.4f}")
# The averaging that hid it: this opponent's OVERALL record is ~50%, so the
# unflagged weights rank it mid-pack while the matrix puts a cell of it on top.
_agg = _m.weights()
if _agg.argmax() == 1 and _agg[1] > 1.5 / len(_agg):
    die("pfsp", "the test case is not actually hidden by the average")
_cnt = {}
for _s in range(600):
    _o, _k = _m.sample_cell(np.random.default_rng(_s), _keys)
    _cnt[(_o.name, _k)] = _cnt.get((_o.name, _k), 0) + 1
if max(_cnt, key=_cnt.get) != (_hard[0].name, _hard[1]):
    die("pfsp", f"sampling does not favour the losing cell: {_cnt}")
ok(f"a losing cell reaches {_w[1, 1] / (1 / _w.size):.1f}x uniform weight and is "
   f"the modal draw ({_cnt[(_hard[0].name, _hard[1])]}/600), while its "
   f"opponent's AVERAGE weight is only "
   f"{_agg[1] / (1 / len(_agg)):.1f}x -- the thing the average hides")

# The optimistic prior: an unseen cell of a thoroughly-beaten opponent is still
# sampled. This is the property that makes a 940-cell matrix usable at all.
_seen = _m.entries[1].cell_winrate("deck-B")
_unseen = _m.entries[1].cell_winrate("deck-C")
if abs(_unseen - 0.5) > 1e-12:
    die("pfsp", f"an unseen cell does not read 0.5 ({_unseen})")
if _w[1, 2] <= 0:
    die("pfsp", "an unseen cell has zero weight -- it would never be sampled")
ok(f"an unseen cell reads exactly 0.5 and keeps weight {_w[1, 2]:.4f} "
   f"(seen-and-losing is {_w[1, 1]:.4f})")

# The decay is per-cell, on that cell's own visits. The global decay the
# per-opponent record uses would erase a cell faster than it can be revisited;
# rl/tools/matchup_decay.py measures that. Here: 20 wins in one cell must leave
# that cell near 1.0 even though 20 other games happened elsewhere.
_d = _mkpool(True, n=2)
for _ in range(20):
    _d.record(_d.entries[0], learner_won=True, deck_key="deck-A")
    _d.record(_d.entries[1], learner_won=False, deck_key="deck-B")
if _d.entries[0].cell_winrate("deck-A") < 0.9:
    die("pfsp", f"a cell with 20 straight wins reads "
                f"{_d.entries[0].cell_winrate('deck-A'):.2f} -- the cell record "
                f"is being decayed by games in other cells")
if _d.entries[1].cell_winrate("deck-B") > 0.1:
    die("pfsp", f"a cell with 20 straight losses reads "
                f"{_d.entries[1].cell_winrate('deck-B'):.2f}")
ok(f"per-cell decay survives interleaved games elsewhere "
   f"(20-0 cell reads {_d.entries[0].cell_winrate('deck-A'):.2f}, "
   f"0-20 reads {_d.entries[1].cell_winrate('deck-B'):.2f})")

# (d) resume round-trip, through torch.save like the real one.
import tempfile
from pathlib import Path

_path = Path(tempfile.mkdtemp()) / "resume.pt"
torch.save({"pool_records": _m.records()}, _path)
_rs = torch.load(_path, map_location="cpu", weights_only=False)
_fresh = _mkpool(True, n=3)
_hit = _fresh.load_records(_rs["pool_records"])
if _hit != 3:
    die("pfsp", f"load_records re-attached {_hit} of 3 entries")
for _a_, _b_ in zip(_m.entries, _fresh.entries):
    if abs(_a_.wins - _b_.wins) > 1e-12 or abs(_a_.games - _b_.games) > 1e-12:
        die("pfsp", f"{_a_.name}: aggregate record did not round-trip")
    if {k: tuple(v) for k, v in _a_.cells.items()} != \
       {k: tuple(v) for k, v in _b_.cells.items()}:
        die("pfsp", f"{_a_.name}: cells did not round-trip\n"
                    f"  {_a_.cells}\n  {_b_.cells}")
if not np.allclose(_m.cell_weights(_keys), _fresh.cell_weights(_keys)):
    die("pfsp", "the restored matrix samples differently")
# ...and it degrades rather than crashing when the names do not line up, which
# is what a resume whose snapshots were pruned looks like.
_odd = _mkpool(True, n=2)
_odd.entries[0].name = "iter99"
if _odd.load_records(_rs["pool_records"]) != 1:
    die("pfsp", "a partial name match did not re-attach exactly the one entry")
if _odd.entries[0].cells or abs(_odd.entries[0].learner_winrate - 0.5) > 1e-12:
    die("pfsp", "an unmatched entry did not keep the optimistic prior")
if "matchups:" not in _odd.summary() or "pool:" not in _odd.summary():
    die("pfsp", f"summary() hides the matchup state: {_odd.summary()!r}")
ok("records() round-trip through torch.save, and an unmatched entry keeps the "
   "prior instead of crashing")
if _m._matchup_summary() == "" or _off._matchup_summary() != "":
    die("pfsp", "the matchup summary is not gated on the flag")
print("       summary:", _m.summary().replace("\n       ", " | "))
ok("summary() surfaces the matrix only for a flagged pool")

# ---------------------------------------------------------------------------
print("\n[11] the joint choice reaches the right env slot")

# The handoff between `MatchupDealer` (which owns the decks) and `Trainer`
# (which owns the opponent) is a FIFO, so it is exactly one off-by-one away from
# labelling every record with another slot's deck -- which would look like it
# worked. This checks the label against the LEGEND the env was actually dealt.
_t2 = T
_cfg2 = _dc_replace(Config().at_victory_score(3), units_only=False)
_decks = select_decks(_t2, None, 0.6)
_dealer = MatchupDealer(_t2, _decks, 0.6)
_hp2 = HP(n_envs=6, rollout=16, matchup_pfsp=True)
_tr = Trainer(_t2, _cfg2, _dealer, _hp2, seed=3)
_byname = {d.name: d for d in _decks}


def _check_alignment(where):
    for _i in range(_hp2.n_envs):
        _nm = _tr.env_names[_i]
        if _nm is None:
            die("matchup-align", f"{where}: slot {_i} has no deck label")
        for _seat in (0, 1):
            _want = _byname[_nm[_seat]].legend
            _got = int(_tr.vec.envs[_i].state.legend[_seat])
            if _want != _got:
                die("matchup-align",
                    f"{where}: slot {_i} seat {_seat} is labelled "
                    f"{_nm[_seat]} (legend {_t2.names[_want]}) but the env was "
                    f"dealt {_t2.names[_got]} -- the FIFO is off by one")
        if _tr.env_deck[_i] != _nm[_tr.learner_seat[_i]]:
            die("matchup-align", f"{where}: slot {_i} records against "
                                 f"{_tr.env_deck[_i]} but the learner sits on "
                                 f"seat {_tr.learner_seat[_i]} with "
                                 f"{_nm[_tr.learner_seat[_i]]}")


_check_alignment("after construction")
if _dealer.queue:
    die("matchup-align", f"{len(_dealer.queue)} choices left queued after "
                         f"construction -- every later pop is offset by that")
ok(f"all {_hp2.n_envs} slots are labelled with the decks their env holds, and "
   f"the queue is drained")

# Now bind a pool and replay the vec contract: `_fresh` in ascending slot order
# (what `VecRiftbound.step` does for finished slots), then `_assign` in that
# same order (what `collect` does over `np.flatnonzero(done)`).
for _i in range(3):
    _tr.pool.add(_tr.net, _i, name=f"iter{_i}")
_dealer.bind(_tr.pool, _tr.rng, 1.0)
for _round in range(4):
    _dones = [1, 3, 4] if _round % 2 else [0, 2, 5]
    for _i in _dones:
        _tr.vec._fresh(_tr.vec.envs[_i], _i)
    for _i in _dones:
        _tr._assign(_i)
    _check_alignment(f"round {_round}")
    if _dealer.queue:
        die("matchup-align", "the queue did not drain one-for-one")
    for _i in _dones:
        if _tr.env_opp[_i] is None:
            die("matchup-align", "pool_frac 1.0 left a slot on self-play, so "
                                 "`vs_opp` would vanish from the log")
        _tr.pool.record(_tr.env_opp[_i], learner_won=(_i % 2 == 0),
                        deck_key=_tr.env_deck[_i])
_cells = sum(len(e.cells) for e in _tr.pool.entries)
if _cells == 0:
    die("matchup-align", "12 recorded games produced no cells at all")
ok(f"12 simulated episodes stay aligned and fill {_cells} cells "
   f"({_dealer.joint_deals} joint deals, {_dealer.self_play_deals} uniform)")

# And the same dealer with no pool bound is just `deck_pool_deal`: every deal
# uniform, nothing steered, which is what the first rollout of a flagged run is.
_d2 = MatchupDealer(_t2, _decks, 0.6)
for _s in range(40):
    _d2(_s)
if _d2.joint_deals or _d2.self_play_deals != 40:
    die("matchup-align", "an unbound dealer steered a deal")
_pick = {_d2.queue.popleft()["names"][0] for _ in range(40)}
if len(_pick) < 15:
    die("matchup-align", f"an unbound dealer only dealt {len(_pick)} distinct "
                         f"decks on seat 0 out of 40 deals")
ok(f"an unbound dealer is uniform ({len(_pick)} distinct seat-0 decks in 40)")

# ---------------------------------------------------------------------------
print("\n[12] sideboards parse, and the legality checks are not vacuous")
# ---------------------------------------------------------------------------
# BACKLOG D2b, first piece. The data was in the repo and unread: every list's
# `Sideboard:` section was dropped on the floor, the same shape as the Chosen
# Champion gap -- parsed by the corpus, ignored by the engine.
#
# **The corpus reports ZERO illegal sideboards, and a clean zero from a validator
# is exactly the shape that has fooled this project before** (a gate validation
# once reported "0 wins skipped" out of 0 wins). So each rule is also shown to
# fire on a list built to break it. `data/tournament_rules.txt` is the authority
# and the only source: Core never mentions sideboards.
from rl.decks import decklist_files as _dfs, load_deck as _ld, sideboard_legal as _sl

_files = _dfs()
_with = [(f, _ld(f, T)) for f in _files]
_have = [d for _, d in _with if d.sideboard]
_illegal = [(d.name, d.sideboard_notes) for _, d in _with if d.sideboard_notes]
if _illegal:
    die("sideboard corpus", f"real decklists reported illegal: {_illegal[:4]}")
elif len(_have) < 20:
    die("sideboard corpus",
        f"only {len(_have)} of {len(_files)} lists parsed a sideboard -- the "
        f"section is being dropped again")
# A list with no section registered NO sideboard, which is legal (0 cards) and
# means that player does not sideboard. It is not a gap to be filled, and it must
# not produce a legality note.
elif any(d.sideboard_notes for _, d in _with if not d.sideboard):
    die("sideboard corpus",
        "a list with no sideboard was reported illegal; 0 cards is legal")
else:
    _sizes = sorted({len(d.sideboard) for d in _have})
    _none = len(_files) - len(_have)
    ok(f"{len(_have)} of {len(_files)} decklists carry a legal sideboard "
       f"(sizes {_sizes}, 601.1.c.1 caps it at 10); the other {_none} registered "
       f"NO sideboard, which is legal and means they do not sideboard")

# Now prove the zero above means something.
_unit = next(i for i in range(len(T.names))
             if T.is_type(i, "Unit") and not T.is_token(i))
_unit2 = next(i for i in range(len(T.names))
              if T.is_type(i, "Unit") and i != _unit and not T.is_token(i))
_bf = next(i for i in range(len(T.names)) if T.is_type(i, "Battlefield"))
_rune = next(i for i in range(len(T.names)) if T.is_type(i, "Rune"))
_leg = next(i for i in range(len(T.names)) if T.is_type(i, "Legend"))

_checks = [
    ("601.1.c.1", "eleven cards", [], [_unit2] * 11, -1),
    ("601.1.c.2", "a battlefield", [], [_bf], -1),
    ("601.1.c.2", "a rune", [], [_rune], -1),
    ("601.1.c.2", "a legend", [], [_leg], -1),
    # 103.2.d spans BOTH zones: counting per zone would pass three in the main
    # plus three in the side, which is six copies.
    ("103.2.d", "a 4th copy in the sideboard", [_unit] * 3, [_unit], -1),
    # 103.2 puts the Chosen Champion inside the 40, and 403.4.a lets it be
    # re-designated from either zone, so it counts toward the limit too.
    ("103.2.d", "three in deck plus the champion", [_unit] * 3, [], _unit),
]
_fired = 0
for _rule, _what, _main, _side, _ch in _checks:
    _notes = _sl(T, _main, _side, champion=_ch)
    if not any(_rule in n for n in _notes):
        die("sideboard checks",
            f"{_what} did not trip {_rule}: got {_notes}")
    else:
        _fired += 1
# And the legal shapes must PASS -- a validator that rejects everything would
# also have produced a clean sweep above.
# Ten DISTINCT cards: ten copies of one would legitimately break 103.2.d, which
# is what the first version of this fixture did -- the validator was right and the
# test was wrong.
_units = [i for i in range(len(T.names))
          if T.is_type(i, "Unit") and not T.is_token(i)]
_ten, _mainc = _units[:10], _units[10]      # disjoint, or the main deck's three
if _sl(T, [_mainc] * 3, _ten):              # copies collide with the sideboard
    die("sideboard checks",
        f"a legal 10-card sideboard was rejected: {_sl(T, [_mainc] * 3, _ten)}")
if _sl(T, [_unit] * 3, []):
    die("sideboard checks", "an EMPTY sideboard was rejected; 0 is legal")
if _sl(T, [_rune] * 12, []):
    die("sideboard checks", "103.2.d.1 exempts Runes from the copy limit")
if _fired == len(_checks):
    ok(f"all {_fired} legality rules fire on a list built to break them, and "
       f"the legal shapes (10 cards, empty, 12 runes) all pass")

# The format setting both mechanics hang off.
from dataclasses import replace as _rp
if not Config().match_mode or _rp(Config(), best_of=1).match_mode:
    die("match mode", "best_of does not drive match_mode")
else:
    ok(f"best_of defaults to {Config().best_of} (Match, 486) and Bo1 reads as a "
       f"Duel (485), which is what makes the random battlefield pick correct")

# ---------------------------------------------------------------------------
print("\n[13] pinned opponents survive eviction and keep a sample floor")
# ---------------------------------------------------------------------------
# BACKLOG #22, cheap version. Both halves are load-bearing and they fail
# differently, so both are tested.
#
# EVICTION is by age, so a pinned ancestor would be the FIRST entry dropped --
# the exact opposite of what a pin is for.
#
# The FLOOR matters because PFSP weights by `(1 - winrate)^p`: an opponent the
# learner beats reliably sinks toward zero sample share. That is correct
# prioritization and it is also precisely how a population forgets how to beat
# something it used to beat, which is the failure `pin_floor` exists to stop.
from rl.pool import OpponentPool as _OP


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(2, 2)


_p = _OP(capacity=3, pin_floor=0.05)
_p.pin(_Tiny(), 0, "v7")
_p.pin(_Tiny(), 1, "v10")
for _i in range(2, 7):
    _p.add(_Tiny(), _i)
_names = [e.name for e in _p.entries]
if "v7" not in _names or "v10" not in _names:
    die("pool pins", f"a pin was evicted: {_names}")
elif len(_p.entries) > 3:
    die("pool pins", f"capacity 3 exceeded: {_names}")
else:
    ok(f"two pins survived five evictions at capacity 3 ({_names})")

# Crush v7 so pure PFSP would drive it to ~0.
_v7 = next(e for e in _p.entries if e.name == "v7")
for _ in range(60):
    _p.record(_v7, True)
_w = _p.weights()
_share = float(_w[[e.name for e in _p.entries].index("v7")])
_bare = (1.0 - _v7.learner_winrate) ** _p.p
if _share < 0.0499:
    die("pool pins", f"the floor did not hold: v7 at {_share:.4f}")
elif abs(float(_w.sum()) - 1.0) > 1e-9:
    die("pool pins", f"weights do not sum to 1: {_w.sum()}")
else:
    ok(f"v7 beaten {_v7.learner_winrate:.0%} of the time keeps "
       f"{_share:.1%} of the sample mass (unfloored PFSP score {_bare:.3f})")

# A floor that cannot fit must share out rather than silently half-apply.
_p2 = _OP(capacity=5, pin_floor=0.5)
_p2.pin(_Tiny(), 0, "a")
_p2.pin(_Tiny(), 1, "b")
_p2.add(_Tiny(), 2)
_w2 = _p2.weights()
if abs(float(_w2[:2].sum()) - 1.0) > 1e-9 or float(_w2[2]) > 1e-9:
    die("pool pins", f"an over-subscribed floor was applied partially: {_w2}")
else:
    ok("floors that cannot all fit are shared equally, not half-applied")

# With no pins at all, the weights must be byte-identical to the old formula.
_p3 = _OP(capacity=5, pin_floor=0.05)
for _i in range(4):
    _p3.add(_Tiny(), _i)
_p3.record(_p3.entries[0], True)
_expect = np.array([(1.0 - e.learner_winrate) ** _p3.p for e in _p3.entries])
_expect = _expect / _expect.sum()
if not np.allclose(_p3.weights(), _expect):
    die("pool pins", "an unpinned pool's weights changed")
else:
    ok("an unpinned pool is unchanged: the floor is inert without pins")

print("\n[14] BACKLOG B2 -- the opponent cache serves the entry it was asked for")
# `_opp_cache` was keyed on `id(entry)`. Once an Entry is freed CPython hands
# the next one the same address, so a NEW snapshot was served the FREED
# opponent's frozen net. Staged as tight as it gets -- nothing allocates
# between the free and the new Entry -- which failed 50 of 50 before the fix.
# (In training the net allocations in between usually take the address first,
# which is why no run was seen to hit it; "usually" is not a cache key.)
from rl.pool import Entry as _E14
_sd_a = {k: v.clone() for k, v in tr.net.state_dict().items()}
_sd_b = {k: v.clone() for k, v in tr.net.state_dict().items()}
_k0 = next(iter(_sd_a))
_sd_a[_k0].view(-1)[0] = 111.0
_sd_b[_k0].view(-1)[0] = 222.0
_stale = 0
for _ in range(50):
    tr._opp_cache = {}
    _a = _E14("a", _sd_a, 0)
    tr._opp_net(_a)
    del _a
    _b = _E14("b", _sd_b, 1)
    if float(tr._opp_net(_b).state_dict()[_k0].view(-1)[0]) != 222.0:
        _stale += 1
if _stale:
    die("B2", f"{_stale} of 50 lookups returned a freed entry's frozen net")
else:
    ok("a new Entry at a freed Entry's address gets its own net, 50 of 50")

print("\n[15] growing an older checkpoint into the current layout changes nothing")
# The new inputs are APPENDED and zero-weighted, so the grown net must compute
# the SAME logits and values as the old one on any position where the new inputs
# are silent. Two older layouts exist: pre-Bo3 (no sideboard zone, 17 fewer
# globals, 3 fewer row columns, 25 kinds, no tag or payment columns) and Bo3
# (31 kinds, no payment columns). Self-contained: each "old" net is built here
# at its layout rather than read from an untracked file.
from rl.engine import actions as _A15
from rl.engine.state import N_DOMAINS as _ND15
from rl.nets import RiftboundNet as _Net15, grow_state_dict as _grow15, \
    PRE_BO3_ACT_SPLIT as _PRE15, BO3_ACT_SPLIT as _BO15, PAY_ACT_SPLIT as _PAY15, \
    act_split_of as _aso15
from rl.obs import Encoder as _Enc15
from rl.env import RiftboundEnv as _Env15
from rl.vec import batch as _batch15

_enc = _Enc15(T, CFG)
_new = _enc.shapes()
_nk_new = len(_A15.KIND_NAMES)
for _label, _split, _tail, _pre_bo3 in (("pre-Bo3", _PRE15, 2 + _ND15 + 2, True),
                                        ("Bo3", _BO15, _ND15 + 2, False),
                                        ("payment", _PAY15, 2, False)):
    _old = dict(_new)
    if _pre_bo3:
        _old["zones"] = tuple(z for z in _new["zones"] if z != "sideboard")
        _old["row_dim"] = _new["row_dim"] - 3
        _old["globals"] = (_new["globals"][0] - 17,)
    _old["actions"] = (_new["actions"][0],
                       _new["actions"][1] - (_nk_new - _split[0]) - _tail)
    if _aso15(_old) != _split:
        die("grow", f"{_label}: act_split_of reads {_aso15(_old)}, not {_split}")
    torch.manual_seed(15)
    _old_net = _Net15(_old).eval()
    _new_net = _Net15(_new).eval()
    _new_net.load_state_dict(_grow15(_old_net.state_dict(), _old, _new,
                                     act_split_old=_split))

    def _strip(t):
        """The new-layout tensors, cut back to what the old layout carried."""
        o = {"zones": {z: t["zones"][z][..., :_old["row_dim"]]
                       for z in _old["zones"]},
             "zone_mask": {z: t["zone_mask"][z] for z in _old["zones"]},
             "globals": t["globals"][..., :_old["globals"][0]]}
        a = t["actions"]
        o["actions"] = torch.cat(
            [a[..., :_split[0]], a[..., _nk_new:a.shape[-1] - _tail]], -1)
        o["action_mask"] = t["action_mask"]
        if "privileged" in t:
            o["privileged"] = t["privileged"]
        return o

    _env15 = _Env15(T, CFG)
    _obs = _env15.reset(3, *DEAL(3))
    _worst = 0.0
    _rng15 = np.random.default_rng(15)
    for _ in range(60):
        if _obs is None:
            break
        _t = to_torch(_batch15([_obs]), "cpu")
        with torch.no_grad():
            _lo, _vo, *_ = _old_net(_strip(_t))
            _ln, _vn, *_ = _new_net(_t)
        _m = _t["action_mask"]
        _worst = max(_worst, float((_lo[_m] - _ln[_m]).abs().max()),
                     float((_vo - _vn).abs().max()))
        _r = _env15.step(int(_rng15.integers(_obs.n_legal)))
        _obs = None if _r.done else _r.obs
    if _worst > 1e-5:
        die("grow", f"{_label}: the grown net differs from the old one by {_worst:.2e}")
    else:
        ok(f"{_label} checkpoint grown == old net on 60 positions "
           f"(max |diff| {_worst:.1e})")

# ---------------------------------------------------------------------------
print("\n[16] the update on MPS matches the CPU update (BACKLOG P1)")
if not torch.backends.mps.is_available():
    print("  skip: MPS is not available on this machine")
else:
    import io
    # Minibatches of ~1,000 so MPS takes the chunked path (MPS_CHUNK=256): an
    # unchunked MPS backward got card_enc weight gradients wrong at ~800+
    # transitions on real data (2026-10-09), so a small-batch test cannot see it.
    _hp16 = HP(n_envs=8, rollout=2048, minibatches=2, update_epochs=2)
    _a = Trainer(T, CFG, DEAL, _hp16, seed=16)
    _roll16 = _a.collect()
    _b = Trainer(T, CFG, DEAL, _hp16, seed=16, update_device="mps")
    _b.net.load_state_dict(_a.net.state_dict())
    _b.gen.set_state(_a.gen.get_state())
    _start = {k: v.clone() for k, v in _a.net.state_dict().items()}
    _sa, _sb = _a.update(_roll16, 0.0), _b.update(_roll16, 0.0)
    _bad = [k for k in ("pg", "vf", "ent", "kl")
            if abs(_sa[k] - _sb[k]) > 1e-3 + 1e-2 * abs(_sa[k])]
    # Adam's first steps are ~lr * sign(grad), so a near-zero gradient that
    # rounds differently can move an element by up to ~2*lr. The test is that
    # the two updates are the same update, not bit-identical.
    _dw = max(float((p - q.cpu()).abs().max())
              for p, q in zip(_a.net.parameters(), _b.net.parameters()))
    _moved = max(float((p - _start[k]).abs().max())
                 for k, p in _b.net.named_parameters())
    _on_cpu = all(p.device.type == "cpu" for p in _b.net.parameters())
    if _bad:
        die("mps", "stats differ: " + ", ".join(
            f"{k} cpu {_sa[k]:.5f} mps {_sb[k]:.5f}" for k in _bad))
    elif _dw > 5 * _hp16.lr or _moved == 0.0:
        die("mps", f"weights differ by {_dw:.2e} (moved {_moved:.2e})")
    elif not _on_cpu:
        die("mps", "the acting net left the CPU")
    else:
        ok(f"stats agree, weights within {_dw:.1e} (update moved them "
           f"{_moved:.1e}), acting net still on the CPU")
    # Resume: the optimizer state is saved with MPS tensors and must load
    # into a fresh MPS trainer (via a CPU map_location, as ppo.py loads it).
    _buf = io.BytesIO()
    torch.save({"net": _b.net.state_dict(), "opt": _b.opt.state_dict()}, _buf)
    _buf.seek(0)
    _rs = torch.load(_buf, map_location="cpu", weights_only=False)
    _c = Trainer(T, CFG, DEAL, _hp16, seed=17, update_device="mps")
    _c.net.load_state_dict(_rs["net"])
    _c.opt.load_state_dict(_rs["opt"])
    _st = _c.update(_c.collect(), 0.5)
    _steps = {int(v["step"]) for v in _c.opt.state_dict()["state"].values()}
    if not all(np.isfinite(v) for v in _st.values()):
        die("mps-resume", f"non-finite stats after resume: {_st}")
    elif _steps != {2 * 2 * 2}:
        die("mps-resume", f"Adam step counts {_steps}, expected {{8}}")
    else:
        ok("optimizer state round-trips through a CPU checkpoint and keeps counting")

print("\n\033[32mall ppo tests passed\033[0m")
