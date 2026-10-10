# Design reasoning

Excerpted from the private project plan: the sections on the learning problem, the gym, the network and evaluation. The phases that built the rules engine are omitted.

## 1. How to make the idea better

Nine changes, ordered by how much they improve the project.

### 1.1 Reframe: this is a deck evaluator, not a game bot

You have a deck-building toolkit — 10 deck folders, ratings, archetypes, a meta
module, a legality checker. An agent that plays one mirror match well is a
science project you'll abandon. An agent that **pilots arbitrary decks well
enough to produce matchup win-rates** is a tool you use every week, and it plugs
directly into `cli.py`.

This changes one design decision: **the deck must be an input, never baked into
the network.** The brief's "card embeddings from attributes, not one-hot IDs"
already sets this up — this reframe is what makes it load-bearing rather than
nice-to-have.

It also gives you value *before the agent is good*. Two equally mediocre agents
still produce meaningful **relative** matchup numbers, because the badness
cancels. You get a usable deliverable at Phase 4 instead of Phase 8.

### 1.2 The critic is the deck evaluator — get it for free

Set `gamma = 1.0` with terminal-only ±1 reward. The value head then learns
literal **win probability from this state**. That gives you, at no extra cost:

- a win-probability curve to plot over any replay (an incredible debugging tool),
- a state evaluator you can query for "how good is this opening hand",
- the matchup oracle from §1.1.

This is why terminal-only reward is worth the slower learning: the critic becomes
a product, not just a variance-reduction trick.

### 1.3 Shrink the victory score for early training

`VICTORY_SCORE = 8` is already a constant (`state.py:20`). Train at **3 points**
first, then 5, then 8.

- Episodes get ~3× shorter → ~3× more games/hour on the same compute.
- Credit assignment over 40 decisions is far easier than over 150.
- The strategic skeleton (contest battlefields, hold them, tempo) is preserved.

Anneal it as a curriculum. This is the single cheapest speedup available and it
costs about four lines of config.

### 1.4 Measure exploitability, not just Elo

Self-play Elo hides cycling — a rock-paper-scissors league shows flat, healthy
looking Elo while the policy spins. The honest metric:

> Freeze checkpoint C. Train a **fresh best-response agent** against frozen C for
> a fixed budget. Report the BR's win-rate.

~55% means C is near-unexploitable at that budget. ~90% means C is a rock waiting
for paper. Run this every N checkpoints. **It is the only evidence that would
justify the bluffing claim**, and without it §8 of the brief is unfalsifiable.

### 1.5 The belief head needs a baseline or it proves nothing

Before building the neural belief head, build the **card-counting baseline**:
uniform distribution over cards not yet observed, given the known decklist,
constrained by hand size. In a mirror match with fixed decks this baseline is
strong and exactly computable.

The network's belief head must **beat that baseline in log-loss**. If it only
matches it, the head learned card counting and *zero behavioral inference* — the
entire hidden-information thesis has failed, silently. Build the baseline first
so you can never fool yourself.

### 1.6 Define bluffing operationally, up front

**The brief modelled the wrong channel — or rather, only one of two.** It assumes
the tell is *open runes*. The `[Hidden]` keyword (44 cards, present in **all ten**
of your decks) says otherwise. Reminder text, quoted from the cards:

> **`[Hidden]`** — *"Hide now for {any rune} to react with later for {0 energy}."*

So there are **two independent threat channels**:

| Channel | Threat is | Visible as |
|---|---|---|
| Runes open | A `[Reaction]` paid from hand | Ready rune count |
| Facedown card | A `[Reaction]` costing **{0 energy}** | A public object at a battlefield (107.3.f) |

A facedown card threatens interaction **with no runes open at all**. An agent that
reads only rune count will systematically misread the board, and so would a human
opponent model built on the brief's assumption.

**Hidden is also the cleanest bluffing substrate imaginable**, and it is native to
the game rather than something we hope emerges:

- Hiding is a **discrete, publicly visible commitment** — the opponent sees you
  spend a rune and place a card, but not which card.
- The bluff is *constructible*: hide something you will never usefully play,
  purely to represent a threat. It costs exactly one rune.
- Ground truth is perfect in self-play — we know what was hidden and whether it
  was ever used.

That is a far better-posed emergent-bluffing experiment than "did they keep runes
open", and it is why Phase 6 targets the Facedown Zone first.

**Metrics, fixed in advance. Log all of these from the first PPO run:**

- **Hide-bluff rate** — P(a hidden card is never played before it is lost).
- **Hide-respect rate** — P(opponent declines a might-favourable Move into a
  battlefield | a facedown card is present there).
- **Rune-bluff rate** — P(end turn holding >=2 ready runes | zero affordable
  `[Reaction]` in hand). The brief's original metric; keep it, it is the second
  channel.
- **Signal strength** — mutual information between each channel (facedown present;
  ready rune count) and the opponent's move-in decision. This is the one that
  actually matters: nonzero only if the signal genuinely informs behaviour.

Watching the two respect rates diverge is also how you will detect the agent
learning one channel and ignoring the other.

### 1.7 Build the vertical slice before the complete engine

The brief says engine-complete, then ML. But its own M5 warns that a failure to
beat random is almost always a masking or encoding bug. You want to discover
those in week 2, not month 5.

So: get a **deliberately crippled but end-to-end-complete** game running into PPO
first — no chain, no spells, units and combat only. Then thicken the engine
underneath a pipeline you already trust. The order is *vertical slice first,
horizontal completeness second*.

### 1.8 Python first; Rust at Phase 8, with Python as the oracle

The brief says Rust/C++ now, TypeScript as a correctness oracle. Invert it:

- The rules model is not settled in your head yet — the open questions in §10
  (showdown vs combat, forced vs declared attacks) are proof. You will
  restructure state three or four times. Each restructure costs ~4× more in Rust.
- The throughput requirement is real but binds at **scale-up**, not at v1. With
  victory score 3 and a 20-card pool, Python gets you to "beats greedy".

Keep the **new** Python engine forever as the differential-test oracle once Rust
lands. (`sim/engine/` cannot serve as the oracle — it has no combat, no chain and
no effects. It is a spec source, not a reference implementation.)

**Write the Python engine in a Rust-shaped way from line one** — fixed-size
arrays, integer IDs, no Python objects in the hot loop, explicit `clone()`. There
is no legacy to preserve, so the eventual port becomes a transliteration rather
than a redesign. This costs almost nothing now and saves weeks later.

**Port trigger, decided now so it isn't argued later:** port to Rust when
profiling shows the engine is >60% of wall-clock at your target batch size, *or*
when a run you want to do would take >48h. Not before.

### 1.9 Source the prior from your local corpus

Section 7 of the brief says "scrape decklists" for P(card | color, archetype).
riftdecks is Cloudflare-blocked — you already know this. Use instead:
`data/staples/`, `data/archetypes.md`, `riftbound/meta.py`, and the saved pages
you parse locally. Same feature, no scraping problem.

---


## 5. Setting up the gym — concrete

### 5.1 Do not use Gymnasium directly

Gymnasium is single-agent. Riftbound is two-player with alternating priority, and
your action space is a variable-length candidate list that fits no standard
`Space`. Forcing it produces hacks you'll fight for months.

**Recommendation: a plain custom env class with a PettingZoo-AEC-shaped API.**
You get correct semantics now and can adopt PettingZoo later if you want its
ecosystem. Self-play with one shared network doesn't need multi-agent plumbing.

```python
class RiftboundEnv:
    def reset(self, seed: int, deck_a: Deck, deck_b: Deck) -> Obs: ...
    def step(self, action_index: int) -> StepResult: ...
    @property
    def to_move(self) -> int: ...
```

`action_index` indexes into the `legal_actions` list the env just returned. The
policy never names an action; it picks a slot. This is what makes the
variable-length action space tractable.

### 5.2 The observation

```python
@dataclass
class Obs:
    zones: dict[str, np.ndarray]   # {"hand": [N_max, D_card], "board": ..., ...}
    zone_mask: dict[str, np.ndarray]  # [N_max] bool per zone
    globals: np.ndarray            # [G] scalars
    legal_actions: np.ndarray      # [A_max, D_act] — one featurized row per action
    action_mask: np.ndarray        # [A_max] bool
    to_move: int
    privileged: np.ndarray | None  # critic-only, training-only
```

**Card features** (`D_card`), all available in `data/cards.json`:
energy, power, might, type one-hot (5), domain multi-hot (6), the 30 keyword
flags from §0, tag hashes, plus a zone tag concatenated on. No card-ID one-hots —
that is what lets unseen cards get sensible embeddings.

**Globals** (`G`): both scores, turn number, phase one-hot, priority holder,
ready/exhausted rune counts by domain for both seats, opponent hand *count*,
deck counts, battlefield control flags.

**Canonicalize.** Always encode from the acting seat's perspective so one network
serves both seats. Verify with an assertion: a mirrored state must produce a
byte-identical observation.

### 5.3 Eight gotchas that will bite

**1. Reward attribution across seats.** The classic self-play bug is crediting the
final actor. Keep **two separate trajectory buffers**, one per seat, tag each
transition with its seat, and write ±1 into both at game end.

**2. GAE must skip opponent steps.** Between two of your seat's decisions, the
opponent acted. Treat each seat's trajectory as its own episode with the opponent
folded into the environment dynamics. Bootstrap from *your* next decision, never
across the opponent's.

**3. Auto-pass — keep it, but only the trivial case.** `actions.py:184` already
has `only_pass()`, which cuts a game from 400+ decisions to ~200. That is a real
speedup and it is safe *only* when the legal set is literally `[pass]` — that's a
no-op, not a decision. **Never auto-pass when a `[Reaction]` is affordable**, or
you destroy exactly the decision points where bluffing lives.

**4. Cap `A_max` and assert.** Worst case ≈ 10 hand uniques × 3 locations + 6
units × 3 move destinations + pass ≈ 50. Set `A_max = 64`, assert on overflow,
log the distribution. Keep the existing `_unique()` duplicate collapse
(`actions.py:176`) — identical copies are interchangeable, so collapsing them is
both correct and a free action-space reduction.

**5. `plan_payment` is a hidden policy decision.** `actions.py:62` auto-picks the
cheapest rune assignment. Which domains you keep ready is a *real* strategic
choice and the agent currently can't make it. Fine as a v1 approximation —
**record it as a known bias**, and promote it to its own decision point when you
factor multi-part actions.

**6. Determinism.** Separate RNG streams for shuffle vs in-game randomness, both
seeded from the episode seed, so a replay is stable even if you change how many
random draws an effect makes. Test: same seed twice → identical `state_hash`
trace.

**7. Truncation.** Cap at 30 turns → reward 0. Track the truncation rate as a
first-class metric: a rising rate means the agent found a stalling strategy, and
you'll want to know before it eats a training run.

**8. Damage assignment is simultaneous but *separable* — mask it anyway, cheaply.**
Confirmed: you cannot see the opponent's allocation when choosing yours. But the
project owner's read is that it almost never matters, and the rules say why:
**damage is dealt simultaneously and every unit contributes its full Might whether
or not it dies.** So their allocation cannot shrink your pool, and yours cannot
shrink theirs. The two optimisations are independent.

That means it is a simultaneous move with **no strategic interaction** — there is
no mixing to learn, and an earlier draft of this plan was wrong to call it an
equilibrium canary.

Two things still follow:

- **Keep the masking.** "Almost never" is not "never", and the assertion is nearly
  free: the observation handed to the second assigner must be byte-identical
  whether or not the first has chosen. Revisit if **Deflect** or **Shield** turn
  out to make incoming damage conditional — those are the keywords that could
  couple the two allocations.
- **In v1, do not make it a policy decision at all.** Because the problem is
  separable and (per 1.3.b) reduces to an exact-lethal knapsack, **solve it in the
  engine** behind a config flag. That deletes a whole class of decision points,
  shortens episodes, and costs essentially nothing in play strength. Expose it as
  a real decision at Phase 8, when keywords make it interesting.

### 5.4 Vectorization

Python engine, so: **8 worker processes × 32 envs each**, batching inference
either per-worker with numpy or through a central inference server. Start
per-worker; it's simpler and the difference won't matter until Phase 7.

Use CleanRL's `ppo_lstm` single file as the base — ~400 lines you'll fully
understand. **Do not use RLlib**: the candidate-scoring action head is
non-standard and you'll spend more time fighting the abstraction than writing the
algorithm.

---

## 6. Network architecture

### 6.1 Trunk

```
per-zone: card features -> shared MLP -> attention pool (DeepSets or 1-layer transformer)
concat all zone summaries + globals -> trunk MLP -> h  [256]
Phase 5+: h -> LSTM -> h_t
```

### 6.2 Candidate-scoring action head (the core mechanism)

```python
# a: [A_max, D_act] featurized legal actions
# h: [H] trunk output
x = concat(broadcast(h, A_max), a)        # [A_max, H + D_act]
logits = mlp(x).squeeze(-1)               # [A_max]
logits = logits.masked_fill(~action_mask, -inf)
pi = softmax(logits)
```

Two hidden layers of 128 is plenty. This handles variable action counts natively
and scores unseen cards by their features rather than by an index.

### 6.3 Normalize the entropy bonus — non-obvious, matters

Entropy of a softmax over `n` legal actions is bounded by `log(n)`. Since `n`
swings from 2 to 64 across states, a fixed `ent_coef` systematically pushes the
policy toward high-branching states — it will learn to keep its options open for
reasons that have nothing to do with winning. Normalize:

```python
ent_bonus = entropy / math.log(max(n_legal, 2))
```

### 6.4 Heads

- **Policy** — sees only legitimate information (opponent hand as a *count* plus
  public knowledge).
- **Value** — may see full state during training (OpenAI Five asymmetric critic).
  Discarded at inference, so policy validity is preserved.
- **Belief** (Phase 6) — strictly on the policy side. See §4 Phase 6.

### 6.5 Starting hyperparameters

```
gamma            1.0        # terminal-only reward; critic = win probability
gae_lambda       0.95
lr               3e-4 -> cosine decay
n_envs           256
rollout_steps    128
minibatches      4
update_epochs    4
clip_coef        0.2
ent_coef         0.01       # applied to the NORMALIZED entropy (§6.3)
vf_coef          0.5
max_grad_norm    0.5
```

---

## 7. Evaluation methodology

**Always evaluate with paired seeds and seat swap.** Play each matchup twice from
the same seed, once in each seat, and average. Detecting a 2% edge on unpaired
games needs thousands of matches; pairing cuts the variance enormously and
controls for first-player advantage (which rule 485.7's extra rune only partly
offsets — measure the actual random-vs-random seat split and record it).

**Dashboard from day one of Phase 4:**

| Metric | Why |
|---|---|
| Win-rate vs random / greedy / pool | Progress |
| BR-exploitability (§1.4) | The honest one |
| Belief log-loss vs counting baseline (§1.5) | Falsifies the thesis |
| Bluff rate, respect rate, rune↔attack MI (§1.6) | The actual goal |
| Truncation rate | Stalling detector |
| Mean episode length, mean `n_legal` | Engine health |
| Value calibration (predicted vs actual win rate) | Critic sanity |

---

