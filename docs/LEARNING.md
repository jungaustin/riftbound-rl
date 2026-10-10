# How the agent learns

The design decisions and their rationale live in `PLAN.md`. This is the
operating manual: what the algorithm is, how to read a run, and how to change
things and know whether you helped.

---

## 1. The algorithm: PPO

**Proximal Policy Optimization** — model-free, **on-policy**, actor-critic
**policy gradient**.

| part | what it is |
|---|---|
| actor | `π(a \| s)` — a *probability distribution* over legal actions, **sampled** |
| critic | `V(s)` — a win probability, used only as a variance-reduction baseline |
| advantage | GAE, `λ = 0.95` |
| discount | `γ = 1.0` (terminal reward only) |
| stability | clipped surrogate objective — PPO's trust region |
| opponent | self-play + a PFSP pool of frozen past selves |

### There is no Q

Nothing learns "the value of action *a* in state *s*", and nothing ever takes a
`max` over actions. That was deliberate: optimal play in a two-player
hidden-information game is generally a **mixed** strategy, and a deterministic
argmax cannot represent bluffing — it is maximally exploitable. See
`PLAN.md` and the `rl-training-design-thesis` memory for the full argument.

**The one thing that looks like Q-learning and is not.** The action head emits
one number per legal action:

```python
def logits(self, h, actions, action_mask):          # nets.py
    x = torch.cat([h.expand(...), actions], -1)
    return self.act_head(x).squeeze(-1).masked_fill(~action_mask, NEG_INF)
```

Those are **logits of a softmax**, normalized into a distribution and sampled —
never maxed. `play.py --deterministic` takes the argmax, but that is for
inspection only; training never does.

### The regression: what the critic is actually fitted to

This is the only supervised-learning piece in the whole system, and it is worth
being precise about, because "the critic predicts who wins" skips three steps.

**The loss** (`ppo.py`, `Trainer.update`):

```python
vf = 0.5 * (value - ret_all[mb]).pow(2).mean()
```

So it is a **squared-error (L2 / least-squares) regression** — ordinary
regression, not logistic, not classification. One scalar output, fitted by
gradient descent on mean squared error, with `vf_coef = 0.5` weighting it against
the policy loss.

**The target is not the ±1 result.** It is `ret`, the TD(λ) return, built in
`finish_episode`:

```python
delta = r + γ·V(next own decision) − V(now)
adv   = delta + γ·λ·adv                      # λ = 0.95, walked backwards
ret   = adv + V(now)
```

At the last decision of a game `r` is ±1 and `V(next)` is 0, so the target there
is exactly the result. Walking backwards, each earlier target is a blend of the
real outcome and the critic's own later estimates — λ controls the blend. λ = 1
would regress every decision directly onto the final ±1 (unbiased, very noisy);
λ = 0 would regress each onto the next state's estimate (low variance, biased by
whatever the critic currently believes). 0.95 sits near the unbiased end, which
is the right place when the reward is terminal-only and there is nothing else to
learn from.

**`γ = 1.0` is what makes the output a win probability.** No discounting means a
win in twenty turns is worth exactly a win in six, so the quantity being
regressed is `E[outcome]`. With outcome in {−1, +1} that is `2p − 1`, and
`net.win_prob()` exists to report it.

**Why least squares and not a classifier.** The honest answer is that it is what
PPO does, and the choice is defensible: MSE onto a {−1, +1} target recovers the
conditional *mean*, which is exactly `2p − 1`, so the estimator is right in
expectation. But the underlying question really is binary, and a Bernoulli head
with cross-entropy would be better **calibrated** near the extremes — squared
error on a bounded target has vanishing gradient pressure exactly where the
critic is most confident, which is where a win-probability display is most read.
Nothing here has measured which is better. **Listed in §8 as a gap rather than
asserted as a flaw.**

**The alternating-seat subtlety, which is easy to get wrong.** `V(next)` is this
seat's *next decision*, not the next decision in the game. A Riftbound turn
contains many decisions by one player, and bootstrapping across to the opponent's
estimate would have the critic predicting the wrong player's prospects. The
filter is `own = [s for s in pending if s.seat == seat]`, and both seats get a
terminal reward from the same game result — the seat that moved last has no
special status.

**What the policy loss is, for contrast.** Not a regression at all:

```python
ratio = exp(logp_new − logp_old)
pg = max(−a·ratio, −a·clamp(ratio, 1−0.2, 1+0.2)).mean()
```

That is the clipped surrogate: push up the probability of actions whose advantage
`a` was positive, but refuse to move more than ±20% per update, and take the
**pessimistic** of the clipped and unclipped terms. The critic's only job is to
supply the baseline that turns a raw return into an advantage.

### Why the reward is only ±1 at the end

`γ = 1.0` with terminal-only reward makes the critic a **literal win
probability**, which is the deck-evaluator deliverable and what `net.win_prob()`
returns. There is no "+0.1 per point" bonus, deliberately: shaped reward teaches
the shape, not the game.

If you ever add shaping, use the **potential-based** form
(`F = Φ(s') − Φ(s)`), which provably cannot change the optimal policy. And the
potential must be **turns-to-win**, not score — see §5.

---

## 2. The network

~1,008,000 parameters. Everything is sized from `Encoder.shapes()`, so a new
observation zone reaches the network automatically.

```
per-zone card rows ──► card_enc (shared MLP)  ──► masked mean ‖ max pool ─┐
                       ONE encoder for every zone                         │
                                                                          ▼
                                                 globals (135) ──► trunk ──► h
                                                                          │
                            ┌─────────────────────────────────────────────┤
                            ▼                                             ▼
        act_head(h ‖ each legal action row) ──► logits        value_sym(h) ──► V
                                                              value_priv(h ‖ hidden) ──► diagnostic
```

Zones: `hand`, `board`, `battlefields`, `facedown`, `legends`, `chain`,
`decks`, `champions`. `row_dim` 283, `card_dim` 237, `globals` 135.

**Two properties worth knowing:**

- **DeepSets, one shared card encoder across every zone.** That is why all zones
  must share `row_dim`.
- **No card-ID one-hots.** Cards are described by attributes and behaviour, so
  the net generalizes to cards it has never seen. This is also why a rules fix
  does not invalidate a checkpoint — it changes a card's *features*, not the
  vocabulary.

### The two critics

`value_sym` sees exactly what the policy sees and **drives learning by default**.
`value_priv` additionally sees the opponent's hand and facedowns, is trained on
the same returns as a diagnostic, and is fed a **detached trunk** so it cannot
shape the features the policy reads.

`hp.privileged_critic` flips which one leads. `test_ppo` [7] enforces the
guarantee: scrambling hidden information leaves every action logit bit-identical,
and the auxiliary critic's gradient never reaches a shared tensor.

`net.win_prob()` always returns the honest, policy-side number. That is the only
value ever shown to a person.

---

## 3. Hyperparameters (`HP` in `ppo.py`)

| | | | |
|---|---|---|---|
| `gamma` | 1.0 | `clip_coef` | 0.2 |
| `gae_lambda` | 0.95 | `ent_coef` | 0.01 (on **normalized** entropy) |
| `lr` | 3e-4, annealed | `vf_coef` | 0.5 |
| `n_envs` | 64 | `max_grad_norm` | 0.5 |
| `rollout` | 2048 transitions | `privileged_critic` | **False** |
| `minibatches` | 4 | `pool_frac` | 0.5 |
| `update_epochs` | 4 | `pool_every` / `pool_capacity` | 10 / 20 |

`rollout` is **transitions per update**, not steps per env.

---

## 4. Reading a training run

```
[ 40/150] steps= 106,048 eps= 59 len= 35.0 turns= 5.6 seat0=56% trunc=0.0%
          | ent=0.236 kl=0.0031 clip=0.03 ev=+0.68 | 320 st/s
```

| field | meaning | worry when |
|---|---|---|
| `ev` | explained variance — does `V(s)` predict outcomes? | **The one to watch.** Stuck near 0 → advantages are noise → nothing can learn |
| `ent` | normalized entropy, 1.0 = uniform, 0 = deterministic | crashes to ~0 in the first few iterations = premature collapse |
| `kl` | how far the policy moved this update | spikes → step too large, lower `lr` |
| `clip` | fraction of samples hitting the PPO clip | persistently > 0.3 → steps too large |
| `trunc` | games hitting the cap (reward 0 both sides) | anything high → training on garbage |
| `seat0` | self-play first-player win rate | **not a skill measure** — see §5 |

Entropy is normalized by `log(n_legal)`, so 1.0 means "uniform" whether there
are 2 legal actions or 40. Low entropy is not automatically bad: in a solved
combat position you *want* it near 0.

**When it does not learn**, check in this order (`PLAN.md` §4): action mask
alignment → reward sign and seat attribution → observation canonicalization →
advantage normalization → learning rate. It is one of the first three about 90%
of the time, which is why `test_env` and `test_ppo` cover all three permanently.

---

## 5. The trap: self-play win rate is meaningless

Both seats are the same network, so self-play sits near 50% by construction and
**can never tell you whether you are improving.**

Measured twice on this project, with the opponent pool switched on the second
time:

| run | best | last | drop |
|---|---|---|---|
| Phase 4 (procedural decks) | 69.0% vs greedy | 60.2% | **8.8 pts** |
| v3 (real decks, `pool_frac=0.5`) | 54.5% vs greedy | 47.0% | **7.5 pts** |

Late training made it **worse against a fixed opponent** while `vs_random` stayed
flat. That is self-play drift, and the PFSP pool did not prevent it — the pool is
one lineage of past selves, not a population.

**Always measure against a fixed external opponent, with paired seeds swapped
across seats.** At victory 3 going first is worth ~64%, so an unpaired number is
off by ~14 points. `eval.duel` already pairs.

### The measurement ladder, weakest to strongest

1. **vs random** — sanity only. 90% just means it is not broken.
2. **vs greedy** — a fixed external heuristic. Real signal.
3. **per-deck breakdown** (`eval.per_deck`) — the aggregate cannot distinguish
   "decent at all 30 decks" from "strong at 27, hopeless at 3".
4. **vs past selves** — Elo over the pool. Catches drift.
5. **exploitability** — train a fresh policy purely to beat a frozen one; how
   much it wins by is how exploitable you were. **The only real measure, and it
   is not built yet.**

### Tempo, and why the potential must not be score

Turns-to-win is `ceil((V − points) / 2)`, so scores **pair up**: at V=8, 6 and 7
are both one turn from winning and the 7th point buys no tempo at all. What
decides which scores are efficient is the parity of `(V − s)`, so it **inverts**
when V changes parity — Aspirant's Climb (V=9) makes 7 a milestone where normally
it is the wasted point.

Two consequences:

- Shaping on score difference asserts "a point is a point, linearly", which is
  false. Use turns-to-win as the potential.
- **The victory-3 curriculum teaches the inverted parity.** Do not expect tempo
  subtlety from it; it is a warm-start for mechanics only.

This is also the arithmetic behind "is pushing a unit in for one point worth
losing it?" At 6 of 8 the point saves a turn. At 7 of 8 it saves nothing and the
unit was spent for free. Both seats' turns-to-win are now input features.

---

## 6. Changing things

### A. Change what it sees — `obs.py`. Highest leverage.

The observation is the contract: the policy can never learn what does not reach
it. Recipe:

1. Add the column or zone — bump `CTX_DIM` / `GLOBAL_DIM`, extend `ZONES`, write
   the builder.
2. `nets.py` needs **nothing** — it reads `Encoder.shapes()`.
3. Run `test_env`. Four gates catch the real mistakes: mirrored positions must
   encode byte-identically, hidden information must not reach the policy, every
   declared zone must reach the network, and **every GameState field must be read
   or classified in `OBS_UNREAD`** ([4e]).
4. Old checkpoints are now invalid — the shape changed.

`OBS_UNREAD` exists because `victory_bonus` was invisible for months and nothing
failed. Being on that list is a *claim* that the policy does not need the field.

### B. Change what it plays — `config.py` and the dealers. Underrated.

`victory_score` (anneal 3 → 5 → 8), `deck_known_prob`, `units_only`, and
`v1_deal` (procedural) vs `deck_pool_deal` (the 30 real lists). The distribution
often matters more than the algorithm.

### C. Change how it learns — `HP`, `nets.py`. **Start here last.**

Least leverage per hour and hardest to attribute; you need many seeds to
distinguish a real gain from noise.

---

## 7. Commands

```bash
# generalist, all 30 real decks, with the per-deck breakdown at the end
python3 rl/ppo.py 150 --real-decks --victory 3 --per-deck 12 --out rl/runs/NAME

# anneal the curriculum: warm-start from a finished run
python3 rl/ppo.py 150 --real-decks --victory 5 --init rl/runs/NAME/best.pt --out rl/runs/NAME5

# narrow the pool: three archetypes, or one (a mirror-match specialist)
python3 rl/ppo.py 150 --real-decks --decks akali,darius,lillia-protector
python3 rl/ppo.py  60 --real-decks --decks fae-fawn --init rl/runs/NAME/best.pt
```

`--init` asserts the checkpoint's observation layout matches. A **rules fix** with
no observation change → loads fine, warm-start it. **Any** observation change →
refuses, loudly. Checkpoints record the engine commit in `ck["engine"]`.

One iteration ≈ 2048 transitions ≈ **9 seconds** on CPU, so 150 iterations is
roughly 22 minutes plus evaluation.

---

---

## 8. Glossary

Every term that appears in a training log, a duel, or one of these write-ups.

### The algorithm

| term | what it means here |
|---|---|
| **on-policy** | Learns only from games played by the *current* policy. Each batch is used for 4 epochs and then thrown away — it cannot be replayed later, because the policy that generated it no longer exists. This is why a run's cost is measured in games, not in gradient steps |
| **actor / policy** `π(a\|s)` | The distribution over legal actions. Sampled, never maxed |
| **critic / value** `V(s)` | One scalar: the expected game result from here, in [−1, +1]. Its only job in training is to be the baseline that turns a return into an advantage |
| **advantage** `A` | `return − V(s)`: how much better things turned out than the critic expected. Positive advantage pushes an action's probability up. **Normalized per minibatch** to mean 0, sd 1, so the update size does not depend on how extreme the batch happened to be |
| **GAE** | Generalized Advantage Estimation. The λ-weighted blend of multi-step returns described above. `λ = 0.95` |
| **γ (gamma)** | Discount. **1.0 here** — a win later is worth a win now, which is what makes the critic a win probability |
| **clipped surrogate / `clip_coef`** | PPO's trust region: the policy may not change any action's probability ratio by more than ±20% in one update. Without it, a single large advantage can collapse the policy onto one action |
| **entropy bonus / `ent_coef`** | A reward for keeping the action distribution spread out. Entropy is highest when every legal action is equally likely and 0 when one is certain. Subtracted from the loss, so the optimizer is paid to stay uncertain. Counteracts premature determinism |
| **epochs / minibatches** | Each rollout is reused `update_epochs = 4` times, split into `minibatches = 4` random chunks per pass: 16 gradient steps per iteration |

### Reading the log

| field | what it means | what to watch for |
|---|---|---|
| **`st/s`** | **Environment steps per second** — throughput. One "step" is one decision applied in one env, summed across all 64 envs. Purely a speed number; it says nothing about learning | Printed as instantaneous with the cumulative average beside it. A `STALL` note means an iteration took >3x the running median, which on this machine means macOS slept — see `BACKLOG` B5 |
| **`eps`** | Completed episodes (games) in this rollout | **Below ~70 the gradient is estimated from too few outcomes.** The reward is terminal-only, so the unit of signal is the episode, not the step |
| **`len`** | Mean decisions per episode | ~100 at victory 8 with spells; ~17-20 decisions per *turn* is the fingerprint that spells are actually on |
| **`kl`** | KL divergence between the policy before and after the update — **how far the policy actually moved.** Not a loss, a diagnostic | Decaying toward ~0.001 means the policy has stopped moving. This is the single number all five null results have in common |
| **`ent`** | Current policy entropy | Falling means it is becoming deterministic |
| **`clip`** (clipfrac) | Fraction of samples where the ratio hit the ±20% clip | High means the update wanted to move further than PPO allowed |
| **`ev`** | Explained variance of the critic: `1 − Var(ret − V)/Var(ret)`. 1.0 is perfect prediction, 0 is no better than guessing the mean | ~0.75-0.85 here. If it goes negative the critic is actively misleading the advantage |
| **`vs_opp`** | Win rate against the *frozen pool opponents* only, with an ewma | **If this vanishes from the log, the pool is empty and a "pool run" has silently become pure self-play.** That failure has happened and this is the tell |
| **`ewma`** | Exponentially weighted moving average — a running mean that forgets old values geometrically. Used wherever a single noisy eval would otherwise be read as a trend |

### Opponents

| term | what it means |
|---|---|
| **self-play** | Both seats are the *current* policy. Free, and the trap is that the win rate is pinned at 50% by construction — it cannot tell you anything |
| **fictitious self-play** | Train against a population of your own **frozen past selves**, so there is a fixed target to beat and progress is measurable |
| **PFSP** | **Prioritized** Fictitious Self-Play. Rather than sampling past selves uniformly, weight them by how badly you are doing: `w ∝ (1 − learner_winrate)^p`, `p = 2` here. Spend practice where you are losing. `rl/pool.py` |
| **pool / snapshot** | The frozen past selves. One is added every `pool_every = 10` iterations, capacity 20, oldest unpinned evicted |
| **`pool_frac`** | Fraction of envs playing a frozen opponent rather than self-play. 0.5 by default; `--exploit` forces 1.0 |
| **exploiter** | Pool pinned to **one** frozen checkpoint, `pool_frac = 1.0`, `pool_every = 0` so the learner is never snapshotted into its own pool. Warm-started from that same checkpoint, so it begins at a measured 50% and `vs_opp` becomes a direct **exploitability** reading of the champion |
| **pinned opponent** | `--pool-init`: a checkpoint in the pool for the whole run, never evicted and never below `pin_floor = 5%` of the sampling mass. The floor matters because PFSP drives an opponent you beat reliably toward zero share, which is also how a population forgets how to beat it |

### Measurement statistics

These come up because this project has twice drawn a conclusion from a number
that could not support it. `MEASURING.md` is the fuller treatment.

| term | what it means |
|---|---|
| **Elo** | A rating where a difference maps to an expected score: +100 Elo ≈ 64%, +200 ≈ 76%. Converted from a win rate by `400·log10(p/(1−p))`. Useful because it is additive and comparable across pairings; misleading if the sample is small, since a 55% result over 20 games is ±22% |
| **±4.0%** | A 95% confidence interval on a win rate: `1.96·sqrt(p(1−p)/n)`. **If the interval straddles 50%, the duel found nothing** — and the width tells you what size of effect the test could have detected at all. A ±4.0% duel can see a 45-Elo effect; a ±20% duel cannot see anything useful |
| **chi² / dof** | Chi-squared per degree of freedom: the test for **"is this variation more than sampling noise?"** Each group contributes `(observed − expected)² / expected variance`; under pure noise the sum is about equal to the number of groups, so the **ratio is ~1.0 for noise, >1 for real differences, <1 for less spread than chance**. v13's pool read **1.02 by deck and 0.51 by opponent**, which is how we know its opponents were genuinely all equally hard and PFSP had nothing to prioritize |
| **excess variance** | Observed variance of a set of rates, minus the variance that sampling alone would produce. **If it is ≤ 0, there is no detectable real difference between the groups** — the spread you can see is the spread you would expect from coin flips. This is the number that retracted the "15% to 89% matchup spread": the extremes were cells with ~9 games each |
| **power** | Whether a test could have seen the effect, had it been there. A null is only informative with this attached. The v13 duel's interval excluded anything beyond +29 Elo while v10's exploiter had moved ~45, so that null meant something; a 20-game result would not have |
| **cluster bootstrap** | Resampling whole *games* rather than individual decisions when computing an interval. Decisions inside one game are correlated, so treating them as independent understates the error. Used in `rl/tools/behaviour.py` |
| **saturated metric** | A measurement pinned near a ceiling, so real improvement cannot show. `vs_greedy` sits at ~71% with sd ~3 whatever happens, because greedy only attacks when it can wipe every defender and therefore cannot punish sequencing. **It is an anchor, not a yardstick** |

---

## 9. The run lineage: what each one tested

Runs are named `v<N>-<what varied>`. The suffix is the **independent variable**,
and the discipline is one variable per run against a named baseline — which is
why the iteration count, rollout, warm start and holdout are held fixed even when
that costs wall-clock.

| run | the one change | result |
|---|---|---|
| `v7-victory8` | First run at the **rulebook victory score of 8**, rather than the 3 → 5 curriculum. Cold start, because the observation layout had changed and every older checkpoint was dead | The big jump. **+91 Elo over greedy**, and most of the lineage's total distance |
| `v8-postfix` | A re-run **after an engine fix** ("post-fix"), to see whether the fixed rules changed what was learnable | **+87.** Tied with v7 — the fix did not move strength |
| `v9-exploiter` | Intended as an exploiter run | **INVALID.** Launched without `--spells`, so it trained on the units-only game against a champion trained with spells. Nothing caught it, because `ck["shapes"]` is byte-identical either way: **a rules difference is invisible to a tensor shape.** This is why configs are now fingerprinted into checkpoints |
| `v10-exploiter` | **Exploiter mode**: pool pinned to one frozen champion, `pool_frac = 1.0`, warm-started from it so `vs_opp` starts at a measured 50% | **+133, the champion, and the only lever that ever moved — ~45 Elo.** Which is also the evidence that headroom exists |
| `v11-pool` | **More ordinary self-play** from the strongest net, with the snapshot pool. Tests simply "does more training help" | **Null.** +28 Elo ±39 over 31,000 games. Three independent readings agree. This is the diminishing-returns result |
| `v12-entropy` | **`--ent-coef 0.02`** against v11's 0.01 — double the exploration pressure, testing whether the policy going deterministic is what stops progress | **Null.** It worked briefly and then decayed to the same place: kl 0.0043 → 0.0006. Duel 51.7% ±5.7% |
| `v13-matchup` | **`--matchup-pfsp`**: record a (frozen opponent × deck the learner pilots) matrix and sample the pair jointly toward losing cells | **Null, 50.2% ±4.0% over 600 games — and the matrix explained why.** Its pool's difficulty surface was flat (chi²/dof 1.02), because the pool was 20 snapshots of its own run ten iterations apart. PFSP reweights a difficulty surface; on a flat one it is a no-op by construction. **The null was guaranteed before the run started** |
| `v14-diverse` | **`--pool-init`**: four pinned opponents spanning ~46 Elo, so the pool has real variance in it | **Inconclusive.** Reached only 112/300 iterations (macOS slept through 47% of its wall clock). At *matched* iterations it is 48.8% ±4.0% — a tie. Needs a full-length re-run |

**The thread through all of it:** every null shares the same kl curve, decaying to
~0.001. Whatever stops these policies moving is upstream of opponent sampling,
entropy, and inference-time search. The two untried levers are the **LR anneal**
(`--no-anneal` exists, never run) and **real population diversity** at full
length.

**And the measurement lesson, which cost two runs:** establish that the variance
exists before building a mechanism to exploit it. #21 was built the other way
round.

---

## 10. Known gaps

- **108.3.d — the Chosen Champion cannot be played from the Champion Zone.**
  Every deck is understated by one guaranteed threat. The biggest engine gap.
- **Truncation pays 0 while losing pays −1**, so a losing player has an incentive
  to stall. Not yet biting (`trunc=0.0%`) but it will once the policy is strong,
  and victory 8 gives it far more room. Fix: decide a capped game on points.
- **No sterile-loop detection.** `state_hash` makes it cheap — an exact repeat
  with the same seat to move is provably a livelock.
- **No exploitability probe**, so "is it actually good" is still unanswered.
- **The per-deck spread conflates** agent weakness with deck strength. Needs a
  greedy-vs-greedy per-deck baseline to separate them.

### The critic is fitted by least squares onto a binary outcome

`vf = 0.5·(V − ret)²` recovers the conditional mean, which with outcome in
{−1, +1} is exactly `2p − 1`, so it is right in expectation. But the question is
binary, and a Bernoulli head with cross-entropy would be better **calibrated**
near the extremes: squared error on a bounded target has vanishing gradient
pressure precisely where the critic is most confident, which is where
`win_prob()` is most read. **Unmeasured** — it would be an ablation, not a fix,
and `ev` is currently healthy at ~0.75-0.85.
