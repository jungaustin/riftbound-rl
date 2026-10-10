# Training run log

See `BACKLOG.md` for what is queued. Append a row per run. A checkpoint without its engine commit is unreproducible,
which is why `ck["engine"]` exists — a `-dirty` suffix means uncommitted changes,
so the exact engine is only recoverable from the run directory.

All numbers are 100 deals x 2 seats, paired and seat-swapped, against fixed
external opponents. **Self-play win rate is not in this table because it is ~50%
by construction and measures nothing.**

---

## `v11-pool` — ordinary self-play from the strongest net: no detectable gain

**DONE 2026-10-01, stopped by the clock at iteration 280/300, 30,710 games, 4.54M
decisions.** Survived a laptop reboot at iteration 79 and resumed from its
iteration-60 checkpoint. Warm-started from `v10-exploiter/best.pt`, ordinary
self-play with the snapshot pool, rollout 12288 (110 games per update).

**Three independent readings, all the same:**

| measurement | result |
|---|---|
| `vs_greedy`, 14 evals | mean **71.2%, sd 2.7** — no trend across 280 iterations |
| ladder: `v11@280` vs `v10-exploiter` | **+137 vs +136** — indistinguishable |
| duel, 300 games: `v11/last.pt` vs `v10/best.pt` | **54.0% ±5.6** — does not exclude 50% |

So four hours and ~31,000 games of ordinary self-play from the strongest policy
bought at most **+28 Elo ±39**: a positive point estimate that cannot be
distinguished from nothing.

Read alongside the baseline scale — the lineage spans ~46 Elo while greedy-to-v7 is
+91 — **this is the diminishing-returns result.** Incremental self-play is not the
path from here. The remaining candidates are structural: matchup-aware PFSP (#21),
population diversity (#22), or search at inference (#28).

**Two methodological findings, both now fixed, both worth more than the run:**

1. **The snapshot ladder was underpowered and said nothing.** `--deals 15` gives
   ~90 games per rating, and the output gave *identical* Elo to different
   snapshots because their records collide. Sizing table in `MEASURING.md`; the
   short version is ~600 games per rung to resolve 30 Elo, so prefer few rungs
   with many games, or one direct duel.
2. **`best.pt` was selecting on noise.** The max of `vs_random + vs_greedy` over
   14 evals of a saturated metric picked **iteration 20 of 280**. Now an ewma,
   with the chosen value recorded as `selected_on`.

## `v10-exploiter` — the champion IS exploitable, and `vs_greedy` is saturated

**DONE 2026-10-01, stopped by `--max-hours 4` at iteration 334/400, 4.16M steps,
288 st/s.** The first VALID exploitability measurement in this project. Config
verified before launch (see the pre-flight table below) and the result verified
after by an independent duel with a control — the two checks v9 failed.

**The measurement.** `vs_opp`, mean of the raw rate per 50-iteration block:

| iterations | `vs_opp` | 95% CI |
|---|---|---|
| 1-50 | 49.2% | ±1.5 |
| 51-100 | 50.0% | ±1.8 |
| 101-150 | 49.0% | ±1.4 |
| 151-200 | 50.3% | ±1.5 |
| 201-250 | 51.8% | ±1.4 |
| 251-300 | 52.9% | ±1.3 |
| 301-334 | **53.0%** | ±1.9 |

**Extended to 400 on 2026-10-01**, which answered the open question and
corrected my reading of it:

| iterations | `vs_opp` | 95% CI |
|---|---|---|
| 201-250 | 51.8% | ±1.4 |
| 251-300 | 52.9% | ±1.3 |
| 301-350 | 52.5% | ±1.4 |
| 351-400 | 53.3% | ±1.3 |

**It flattened.** I said at iteration 334 that the climb "had not flattened" --
that was reading noise inside an already-flat stretch. The real shape is: flat at
~49.5% for 200 iterations, a climb to ~53% between 200 and 250, and **a plateau at
53% from 250 onward**, 150 iterations with no trend.

First 100 against last 100 is **+3.8pp ±1.4**, so the gain is real. It is also
small and it **saturates quickly** -- which is the useful part: the champion has a
shallow exploitable seam, not a deep one.

**Verified independently, which is the whole point:**

| pairing | result |
|---|---|
| v10 best.pt (iter 320) vs champion, 300 games | **56.3% ±5.6** |
| **champion vs itself, same seeds** (control) | **50.3% ±5.7** — harness unbiased |
| training-time figure for the same checkpoint | 54.4% |

Training and independent play **agree**. In v9 they differed by 17 points, which
is what exposed the config bug.

**It cost nothing in general strength.** v10 best.pt evaluates at **vs_random
100%, vs_greedy 68.3%** against the champion's 97.5% / 70.5%, and `vs_greedy`
across all 16 evals was 64.2-71.7% with no downward trend. So this is not the
rock-paper-scissors drift v9 appeared to show — the policy got genuinely better
against a peer while staying as good at everything else.

### The real finding: `vs_greedy` is a saturated yardstick

The policy improved enough to beat its own predecessor **56.3%** head-to-head,
and `vs_greedy` **did not move** — 70.5% before, 68.3% after, inside noise.

**BACKLOG 15c predicted exactly this**: greedy attacks only when it wipes every
defender, so it can never punish bad sequencing, never make a deliberate trade
and never kill a specific blocker to unlock a line. A policy can improve at all
of those and `vs_greedy` cannot see it.

So **"71% is the plateau" was the wrong conclusion** — the metric plateaued, not
the policy. There is real headroom and self-play can reach it.

**What follows from this:**

1. **More training is justified after all**, which the v9 artefact had argued
   against. The exploiter was still climbing at 4 hours.
2. **The opponent pool / PFSP work (#21, #22) is now motivated by evidence**
   rather than by hunch: a peer can beat this policy by 6 points, so there is
   something for a pool to teach.
3. **A new yardstick is needed before any of it can be measured** (#15c). The
   cheap version is an Elo ladder across checkpoints — each run's `best.pt` beats
   the last one's — rather than one fixed heuristic.
4. `resume.pt` is at iteration 334, so the same command continues this run to
   400. Given the trend was still rising, that is now worth doing.

## `v9-exploiter` — INVALID, a missing flag (2026-09-30)

**This run measured nothing. It was launched without `--spells`, so it trained on
the units-only game while its frozen champion had been trained with spells.**
`--real-decks` did not imply spells at the time, and nothing in the log said which
game was being played.

What that did: the champion was handed a game it was never trained for, with a
large fraction of every decklist dead in hand. The learner adapted to the cheaper
game over 7.09M steps. `vs_opp` duly climbed 51.6% -> 57.5%, which is a real
measurement of "can a net adapt to a spell-less game faster than a frozen net
that cannot" — and of nothing anybody wanted to know.

**The tell was in the very first eval and I did not read it.** At iteration 20,
with kl ~0.005 and the policy essentially unmoved from the champion's weights,
vs_greedy already read **55.0%** against the champion's **70.5%**. A policy that
has not changed cannot lose 15 points by drifting; an instant drop is an
environment change. Three more hours of probing followed before I checked the
config.

**Fingerprint, for diagnosing this in any future log — decisions per turn:**

| run | len / turns | per turn | spells |
|---|---|---|---|
| v5-warm | 42.6 / 6.2 | 6.9 | — |
| v6-overnight | 43.8 / 6.1 | 7.2 | — |
| **v7-victory8** | 151.4 / 9.2 | **16.5** | **on** |
| **v8-postfix** | 136.8 / 6.9 | **19.8** | **on** |
| **v9-exploiter** | 36.7 / 7.0 | **5.2** | **OFF** |

**v7 and v8 are unaffected** — both documented as `--spells` and both showing
16-20 decisions per turn, 3-4x v9's. Every conclusion resting on them stands,
including the per-deck table and the v7-vs-v8 head-to-head (`ab_checkpoints`
runs `units_only=False`, which matches how they trained). v5/v6 cannot be
settled from length alone because the engine changed enormously before v7, and
it does not matter: those checkpoints are dead on observation layout anyway.

**RETRACTED: "chasing one frozen opponent cost ~13 points of general skill."**
That was this run's headline and it was a config artifact, not drift — the policy
was never at 70.5% in that environment to begin with. It said nothing about
single-opponent training, and nothing about the user's per-deck specialist plan,
which I argued against partly on this basis.

**Fixed** so it cannot recur: `--real-decks` now implies spells, `--units-only`
is the explicit escape hatch, the two contradict with an error, and every run's
header now prints which game it is playing.

**The exploitability question is untouched and still open.** The corrected run,
pre-flighted 2026-10-01:

```
PYTHONPATH="$PWD" python3 rl/ppo.py 400 --real-decks --victory 8 \
    --holdout 7 --seed 0 --envs 64 --rollout 6144 --lr 1e-4 \
    --eval-every 20 --eval-games 60 --max-hours 4 \
    --init rl/runs/v8-postfix/best.pt --exploit rl/runs/v8-postfix/best.pt \
    --out rl/runs/v10-exploiter
```

**Pre-flight, all measured rather than assumed:**

| check | result |
|---|---|
| champion under this config | **97.1% / 70.0%** vs its recorded 97.5% / 70.5% — matches |
| champion with spells OFF | 83.8% / 45.8% — the v9 artefact, reproduced on demand |
| header states the game | `game: units + spells  decks: real decklists` |
| `vs_opp` at iteration 1 | 53.2%, ewma settling to 50.5% — the required 50% by construction |
| episodes per update | **86-94**, inside v7's healthy 74-84 band |
| first eval | **vs_greedy 68.3%, PHASE 4 EXIT MET** — against v9's 55.0%, which was the tell |
| throughput | 336 st/s, ~12,500 steps/iteration = ~37s, plus ~96s per eval |
| sizing | ~**340 iterations in 4h**, so 400 is a near-fit and the clock is the backstop |

**Glance at the first eval when it appears (iteration 20).** `vs_greedy` near
**68-70%** means the config is right. Anything near 55% means it is not, and the
run should be killed rather than left overnight — that one number is what four
hours hangs on.

## Summary

| run | victory | decks | iters | best vs random | best vs greedy | notes |
|---|---|---|---|---|---|---|
| Phase 4 | 3 | procedural (`v1_deal`) | 150 | 93.1% | **69.0%** | pre-deck-conditioning; not comparable to real decks |
| `v3-deckcond` | 3 | 30 real | 150 | 81.5% | **54.5%** @ iter 80 | first real-deck run; deck + champion zones |
| `v4-standing` | 3 | 30 real | 150 | 81.0% | **50.5%** @ iter 140 | + 51 standing-state globals, turns-to-win |
| `v5-warm` | 5 | 30 real | 150 | 79.0% | **54.5%** @ iter 110 | warm-started from v4 `best.pt`; **best so far**, barely drifted. Peak `vs_greedy` was 55.5% at a later eval, but `best.pt` is chosen on `vs_random + vs_greedy` and ties go to the earlier checkpoint |

The 65% exit bar in `ppo.py` was calibrated against `v1_deal`'s procedural decks.
The real-deck distribution is far harder and more diverse; **Phase 4's 69% and
v3's 54.5% are not comparable numbers.**

---

## Self-play drift — measured twice

| run | best | last | drop | pool |
|---|---|---|---|---|
| Phase 4 | 69.0% (iter 100) | 60.2% (iter 150) | **8.8 pts** | off |
| `v3-deckcond` | 54.5% (iter 80) | 47.0% (iter 150) | **7.5 pts** | **on**, `pool_frac=0.5` |
| `v5-warm` | 55.5% (peak eval) | 54.5% (iter 150) | **1.0 pt** | **on**, `pool_frac=0.5` |

`v5-warm` barely drifted. Two plausible reasons, not yet separated: the longer
victory-5 games give more signal per episode, and it started from a competent
warm-start rather than from noise, so it spent no iterations flailing. Worth
re-testing at victory 8 before treating it as solved.

Late training made the policy measurably **worse against a fixed opponent** while
`vs_random` stayed flat, with non-overlapping intervals. The PFSP opponent pool
did **not** prevent it: the pool is one lineage of past selves, not a population,
so every entry shares the same blind spots.

Two practical consequences:

- **Always use `best.pt`, not `last.pt`.**
- Picking "best" by periodic eval is itself mild overfitting to the eval set. The
  honest headline is the best number, but it is not a stable capability until
  an exploitability probe exists.

---

## v4 per-deck breakdown (vs greedy, 12 deals x 2 seats x 30 decks)

**A 75-point spread, well beyond the noise.** The generalist is *not* uniformly
competent, which is the empirical answer to "should I train specialists": yes for
a handful of weak decks, warm-started — not thirty separate agents.

| worst | | best | |
|---|---|---|---|
| `meta/ven-1-azir-calm-order` | **4.2%** ±8.0 | `meta/reksai-fury-order` | **79.2%** ±16.2 |
| `meta/sivir-body-chaos` | 8.3% ±11.1 | `meta/darius-fury-order` | 79.2% ±16.2 |
| `meta/lillia-sprite-tempo-netdeck` | 20.8% ±16.2 | `meta/annie-chaos-fury` | 79.2% ±16.2 |
| `meta/lillia-galewinds-unleashed` | 25.0% ±17.3 | `empower-chaos/body-chaos-v1` | 70.8% ±18.2 |
| `meta/lux-mind-order` | 29.2% ±18.2 | `meta/ven-3-masteryi-body-calm` | 66.7% ±18.9 |

Median around 50%. Full table in `rl/runs/v4-standing/train.log`.

**v5 reproduced the spread** (70.8%, noise ±27.3%) with almost the same decks at
each end -- `sivir-body-chaos` and `ven-1-azir-calm-order` worst at 8.3%,
`darius-fury-order` and `annie-chaos-fury` best at 79.2%. Two independent runs
agreeing on *which* decks are badly served is much stronger evidence than one
run's spread, and it makes the deck-strength confound (caveat 1) the next thing
to rule out.

**Two caveats before acting on this.**

1. **It conflates agent weakness with deck strength.** "Bad with Azir" and "Azir
   is bad against the field" produce the same number. Separating them needs
   greedy piloting each deck as a per-deck baseline. **Run 2026-09-28 — see the
   v7-victory8 entry below; it was deck strength.**
2. **n=24 per deck**, so ±8–20% each. The *spread* is real; individual rankings
   are not. Do not read the ordering as meaningful.

Checked and ruled out: all five worst decks are at **100% main-deck coverage**, so
this is not a scripting gap.

---
## v7-victory8 — the first run to clear Phase 4 (2026-09-28)
```
500/500 iters   8.23M env-steps   15.6h on CPU   engine 0ed3cf7
best.pt = iter 500:  vs_random 98.5%   vs_greedy 71.0%   livelocks 0
trunc 0.0%   loop 0.0%   ovf 0.0%
```

`--victory 8 --spells --real-decks --holdout 7 --envs 64 --rollout 12288`

**Victory 8 was the fix, and it was the curriculum rather than the compute.** v6
plateaued near 54.8% at victory 5 by iteration ~300 of 1500 and never reached the
65% bar. This run crossed it at ~iteration 240 and kept climbing; six of the last
nine evals were over it. The standing suspicion in `V8_SETUP.md` — that tempo
parity inverts at odd victory scores, so v3/v5 taught the wrong tempo — is now
the best explanation on the table.

**Generalization is real and was measured, not assumed.** The 7 held-out decks
averaged **69.0%** against **68.6%** for the 41 trained on. A v5 checkpoint fell
from 54.5% to 48.0% when unseen decks were added; this one does not move.

### Specialists: the answer is no

`eval.per_deck` alone cannot answer this, and its 53.8% spread reads like a case
FOR forking. `eval.per_deck_baseline` (greedy on both sides, no network, so a
property of the decks and the engine rather than of a checkpoint) separates
"the agent is bad with this deck" from "this deck is bad against the field" —
the gap the previous entry in this file flagged as *"Not yet run"*.

Run at n=30 deals x 2 seats on the same split. **The agent beats the greedy
baseline on 47 of 48 decks, mean lift +23.5%**, and the lift is *largest* on the
weakest decks. The bottom of the `per_deck` table was deck quality throughout.

| deck | agent | greedy | lift |
|---|---|---|---|
| `owned-lillia/v1` | 86.2% | 30.0% | +56.2% |
| `meta/ornn-calm-mind` | 77.5% | 26.7% | +50.8% |
| `meta/sg-035-defender-of-tomorrow` | 71.2% | 28.3% | +42.9% |
| `meta/lillia-sprite-tempo-netdeck` | 82.5% | 40.0% | +42.5% |
| `meta/sg-383-matriarch-of-war` | 70.0% | 30.0% | +40.0% |
| `meta/sg-007-wuju-bladesman` | 82.5% | 43.3% | +39.2% |
| `meta/viktor-mind-order` | 75.0% | 38.3% | +36.7% |
| `meta/sg-011-pridestalker` | 78.8% | 43.3% | +35.5% |
| `meta/sg-040-fire-below-the-mountain` *(held out)* | 67.5% | 33.3% | +34.2% |
| `meta/sg-069-bashful-bloom` *(held out)* | 72.5% | 38.3% | +34.2% |
| `meta/sg-008-grand-duelist` | 86.2% | 53.3% | +32.9% |
| `renata-gutter-palace/v2` | 63.7% | 33.3% | +30.4% |
| `meta/ven-1-azir-calm-order` | 40.0% | 10.0% | +30.0% |
| `meta/sg-311-chem-baroness` | 85.0% | 55.0% | +30.0% |
| `meta/akali-fury-calm` | 76.2% | 46.7% | +29.5% |
| `lillia-fae-fawn-blind/v9` | 81.2% | 51.7% | +29.5% |
| `lillia-sprite-fortress/v1` | 86.2% | 56.7% | +29.5% |
| `ambessa-empower/v1` | 78.8% | 50.0% | +28.8% |
| `meta/sg-380-grandmaster-at-arms` | 52.5% | 25.0% | +27.5% |
| `meta/sg-384-butcher-of-the-sands` | 77.5% | 50.0% | +27.5% |
| `meta/sg-010-emperor-of-the-sands` | 38.8% | 11.7% | +27.1% |
| `meta/sg-207-keeper-of-the-hammer` | 71.2% | 45.0% | +26.2% |
| `meta/sg-001-rogue-assassin` | 81.2% | 55.0% | +26.2% |
| `lillia-protector/v1` | 73.8% | 48.3% | +25.5% |
| `lillia-fae-fawn/v7` | 83.8% | 60.0% | +23.8% |
| `meta/sg-003-wuju-bladesman` | 76.2% | 53.3% | +22.9% |
| `meta/sg-116-virtuoso` | 83.8% | 61.7% | +22.1% |
| `ornn-swain/v1` | 40.0% | 20.0% | +20.0% |
| `meta/sg-033-gloomist` | 76.2% | 56.7% | +19.5% |
| `meta/sg-343-piltover-enforcer` | 71.2% | 53.3% | +17.9% |
| `meta/sg-109-wuju-master` *(held out)* | 67.5% | 50.0% | +17.5% |
| `lillia-sprite-conquest/v2` *(held out)* | 73.8% | 56.7% | +17.1% |
| `meta/sg-131-curator-of-the-sands` | 58.8% | 41.7% | +17.1% |
| `meta/sg-364-green-father` | 81.2% | 65.0% | +16.2% |
| `meta/sg-041-purifier` | 72.5% | 58.3% | +14.2% |
| `annie-strip-and-steal/v1` | 58.8% | 45.0% | +13.8% |
| `meta/sg-017-deceiver` | 78.8% | 65.0% | +13.8% |
| `meta/sg-395-mechanized-menace` | 92.5% | 80.0% | +12.5% |
| `meta/kennen-koko-ccs-7th` *(held out)* | 66.2% | 55.0% | +11.2% |
| `meta/reksai-fury-order` | 73.8% | 63.3% | +10.5% |
| `meta/ven-3-masteryi-body-calm` | 67.5% | 58.3% | +9.2% |
| `meta/kennen-postban-sacrifice` *(held out)* | 53.8% | 45.0% | +8.8% |
| `meta/sg-450-eye-of-twilight` | 50.0% | 41.7% | +8.3% |
| `meta/ven-2-kaisa-fury-mind` *(held out)* | 81.2% | 73.3% | +7.9% |
| `meta/kennen-gastaldon-convergence-5th` | 66.2% | 58.3% | +7.9% |
| `empower-chaos/body-chaos-v1` | 70.0% | 65.0% | +5.0% |
| `meta/sg-014-void-burrower` | 73.8% | 71.7% | +2.1% |
| `meta/kennen-postban-keeper-of-balance` | 45.0% | 50.0% | -5.0% |

mean agent **71.2%** · mean greedy **47.7%** · mean lift **+23.5%**

The only negative is `meta/kennen-postban-keeper-of-balance` at -5.0%, with
±12.6% on each side of the comparison — noise, not a finding. Worth a look if it
repeats at higher n, not worth a fork.

**What this does NOT say.** Both columns are one checkpoint against one
heuristic. It says the generalist pilots every list in the pool above baseline;
it does not say a specialist could not do better still. The honest specialist
experiment (pilot deck D against the whole field, gradient from D's seat only)
is still unbuilt — `--decks lillia` gives a mirror match instead, which is the
wrong thing. See `select_decks`.

---

## v8-postfix — the engine fixes are free (2026-09-28)

```
180/180 iters   2.93M env-steps   7h23m   engine 69332e0
warm-started from v7-victory8/best.pt, --seed 0 (identical holdout split)
best.pt = iter 180:  vs_random 97.5%   vs_greedy 70.5%   pass
```

The point of this run was not a higher number: it was to check that the three
rules fixes v7 never trained against (the Heron discount split, 373, and
136.2.d) had not broken something the policy was relying on. 136.2.d in
particular measurably reorders events, so this was a real risk.

**They are free. Three measurements, all on identical deals:**

| | v7 (iter 500) | v8-postfix (iter 180) |
|---|---|---|
| aggregate eval vs greedy | 71.0% | 70.5% |
| per-deck mean vs greedy | 71.2% | **68.4%** |
| head-to-head, 600 games | 47.8% | **52.2%** (±4.0) |

The middle row and the bottom row disagree, and that is the interesting part.
It is **not** noise cancelling out: the per-deck comparison is paired across 48
decks on fixed seeds, and -2.9pp with a per-deck sd of 6.2 is about t=3. The
honest reading is **opponent-specific strength**: v8 gave up a little of v7's
greedy-exploitation and gained a little against a peer. That is what training
against a snapshot pool rather than only the current policy would predict, and
the `pool:` lines in `train.log` confirm older snapshots were in the mix.

So a win rate against one fixed heuristic is not a total order over policies.
Anything that matters should be decided head-to-head (`rl/tests/ab_checkpoints.py`).

Two smaller findings:

* **The warm-start dip is real and recovers.** 67.5 -> 61.0 -> 60.5 -> back over
  the bar by iteration 100. `--init` restarts the LR and entropy anneal, so a
  continuation run spends its first third undoing that. Budget for it, or the
  first eval will look like a regression.
* **Per-deck spread narrowed**, sd 13.0% -> 12.1%, and 47 of 48 decks stay above
  the greedy baseline. The one that fell below is
  `empower-chaos/body-chaos-v1` (-7.5%), worth a look if it repeats.

### The plateau

Two runs, 11.2M steps, both in a 65-71% band vs greedy, neither trending up at
the end. **71% is this architecture's ceiling at victory 8 until something
changes besides compute** -- see BACKLOG #5's residual (damage assignment cannot
see which units this seat has already doomed), #6 (718.3 Equipment text), or
search at inference.

---

## Reproducing a run

```bash
cat rl/runs/<name>/PROVENANCE.txt     # engine commit, date, dirty state
git apply rl/runs/<name>/engine.patch # when the tree was dirty
head -3 rl/runs/<name>/train.log      # the exact config line
```

`train.log` holds every iteration line and eval, so the trend can be re-read
without rerunning anything.

## `v13-matchup` — matchup-aware PFSP: the fifth null, and the best-powered one

**DONE 2026-10-05, stopped by the clock at iteration 268/300, 7h02m, ~30,500 games,
4.38M decisions.** One variable against `v11-pool`: same warm start
(`v10-exploiter/best.pt`), same `--rollout 12288 --envs 64 --holdout 7`, with
`--matchup-pfsp` added. Iterations were deliberately left at 300 because the LR
anneal schedules off the TOTAL, so raising it would have changed the learning-rate
curve and cost the comparison.

| measurement | result |
|---|---|
| duel, **600 games** vs `v11/last.pt` | **301-299, 50.2% ±4.0%** |
| as Elo | **+1, CI [−27, +29]** |
| `vs_greedy`, 26 evals | 69.7% mean, sd 3.0 (v11: 71.2 / 2.7) |

**This is the null worth taking seriously, because it had the power to see the one
effect we know is real.** The interval excludes any true edge beyond +29 Elo, and
v10's exploiter moved ~45. So this is not an underpowered shrug.

### The matrix explained the null, and refuted the first thing written about it

This run was first written up as "the mechanism was working — matchups spanning 15%
to 89%, a 74-point spread the average was hiding." **That was wrong, and it was a
small-sample artefact of exactly the kind `MEASURING.md` exists to catch.** The
extremes were cells holding ~9 effective games each.

Tested for variance beyond what sampling alone produces, on the matrix the run
actually built:

| aggregation | groups | eff. games each | chi2/dof | excess |
|---|---|---|---|---|
| per (opponent, deck) cell | 191 (n>=10) | ~12 | **0.75** | none |
| per deck piloted | 41 | ~112 | **1.02** | real sd ~**0.5pp** |
| per opponent | 13 | ~354 | **0.51** | none; resolves 2.3pp |

chi2/dof of 1.0 *is* pure sampling noise. So the learner's win rate against its pool
is **flat** — the same against every deck to within half a point, and the same against
every opponent to within the 2.3pp this test can resolve.

**PFSP's premise is that some opponents deserve more practice than others. On a flat
difficulty surface, reweighting is a no-op by construction.** v13 did not fail to
exploit the spread; there was no spread. The null was guaranteed before the run
started, and nothing in the implementation could have changed it.

### Why the surface is flat, and what that points at

The pool was `iter70 … iter260` — **twenty snapshots of v13's own run, ten iterations
apart.** A sliding window of recent selves is twenty near-copies of one policy, and
near-copies cannot differ in difficulty. The late kl was ~0.0015 per update, so ten
iterations apart is almost no distance at all.

That makes **#22 (population diversity) evidence-backed rather than a guess by
elimination**, and suggests the cheap version first: pin *actually different*
opponents in the pool — v7, v8, v10, greedy and random, with a PFSP floor so they are
never forgotten — and measure whether the difficulty surface becomes non-flat *before*
building any weighting on top of it. **The order matters: establish the variance,
then exploit it.** #21 was built the other way round.

**What all five nulls share is the kl curve.** v13 went 0.0464 → 0.0014 on the same
path as v11 (→0.0005) and v12 (→0.0006). Whatever is stopping these policies from
moving is upstream of opponent sampling, entropy and inference-time search.
**`--no-anneal` is the only untried single-variable lever**, and the LR anneal is the
mechanism that most obviously produces that curve: by iteration 268 of a 300-iteration
cosine, the rate is near its 0.1× floor whatever else is true.

A throughput note, not a result: st/s fell 345 → 173 over the run because other work
on this machine competed for CPU, which is why it reached 268 rather than 300. v11
reached 280/300, so the two sit at a comparable point on the anneal.

## `v14b-diverse` — opponent diversity: the first positive since v10

**DONE 2026-10-06, all 300/300 iterations in 5h51m, ~37,000 games, 4.96M decisions.**
One variable against `v11-pool`: **the pool's composition.** Same warm start
(`v10-exploiter/best.pt`), same `--rollout 12288 --envs 64 --holdout 7`, same
300-iteration anneal schedule, plus `--pool-init` pinning four genuinely different
opponents — v7 (+91), v8 (+87), v10 (+133), v12 — spanning ~46 Elo on the ladder.

**Ran under `caffeinate -dimsu` and did not sleep once**, which is why it finished
the schedule where `v14` reached 112/300. That is the whole difference between the
two runs' lengths; see BACKLOG B5.

| measurement | result |
|---|---|
| duel, 600 games vs `v11/last.pt` | **52.8% ±4.0%** — Elo +20, CI [−8, +48] |
| duel, 600 games vs `v10/best.pt` | **53.7% ±4.0%** — Elo +26, CI [−2, +54], p = 0.07 |
| `vs_opp`, start → end | **37.5% → 49.1%** (ewma) |

### Settled at 1,200 games: this IS a gain

The 600-game duels were both positive and both straddled 50%, which is the blind
spot a ±4.0pp instrument has against a 3-point effect. Re-run at 1,200 games
(±2.8pp):

| duel | result | Elo | |
|---|---|---|---|
| vs `v11-pool/last.pt` | **642-558 = 53.5% ±2.8%** | **+24, CI [+5, +44]** | **p = 0.015, excludes 50%** |
| vs `v10-exploiter/best.pt` | 630-570 = 52.5% ±2.8% | +17, CI [−2, +37] | p = 0.083 |

**The v11 duel is the one that counts.** It is the clean single-variable
comparison — identical warm start, rollout, envs, holdout and anneal schedule,
with only the pool's composition different — and **v11 was never in v14b's pool.**
v10 *was* a pinned opponent, so beating it is partly an exploiter effect and is
the weaker evidence of the two.

**The two sets of duels must NOT be pooled.** `ab_checkpoints` deals `seed0 + i`
for `i in range(n)`, so the 1,200-game sample *contains* the 600-game sample's
deals; adding them would double-count 600 games. The 1,200-game result supersedes
rather than confirms.

### What IS new, and is not a coin flip

**`vs_opp` climbed from 37.5% to 49.1%.** The learner began losing badly to the four
pinned ancestors and reached parity over 300 iterations.

No previous run produced that trajectory. v11 and v13 sat flat near 50% for their
whole length, because their pools were snapshots of themselves taken ten iterations
apart — near-copies, with no gap to close. v13's matrix measured exactly that: a
**flat** difficulty surface, chi2/dof 1.02 by deck. A diverse pool has a gap, and
this run closed one.

That is the prediction #22 made, and it is independent of the duels: it comes from
the training log rather than from a post-hoc comparison.

### The methodological lesson, which is bigger than the run

**A 600-game duel is ±4.0pp and cannot resolve a 3-point effect.** v14b's two duels
land exactly in that blind spot — and so did several earlier results that were
recorded as nulls. The cost asymmetry is stark: the run took **6 hours**, and the
measurement that decides whether it meant anything takes **45 minutes**.

**Size the duel to the effect you care about before concluding.** 1,200 games is
±2.8pp; 2,400 is ±2.0pp. Pair this with the sizing table in `MEASURING.md`.

## `v15-bo3` — Bo3 matches: a clear gain in the format it trained on

**DONE 2026-10-07, all 300/300 iterations in ~6h, 4.94M decisions, 0% truncated.**
One variable against `v14b-diverse`: **`--bo3`** (BACKLOG D2/D2b). Same recipe —
`--real-decks --victory 8 --envs 64 --rollout 12288 --holdout 7`, pinned v7/v8/v10/v12
— warm-started FROM `v14b/last.pt`, grown into the Bo3 observation layout
(`nets.grow_state_dict`; the grown net plays exactly as v14b did). Every slot plays
back-to-back matches; the between-game decisions (sideboarding, battlefield pick,
play order) are the policy's own, and the reward stays per game.

kl 0.074 on the first update (vs 0.81 for the W1 spike at HEAD — the grown warm
start is gentle) decaying to 0.001; `vs_opp` ewma 44.8% → 54.2%; final `vs_greedy`
75.0%, its best. Two ~150s stalls (171, 291) despite the wake lock; harmless.

| duel (sampled, `ab_checkpoints`) | result | Elo | |
|---|---|---|---|
| **Bo3, 600 matches** (`--bo3`) | **334-266 = 55.7% ±4.0%** | **+40, CI [+12, +68]** | **p ≈ 0.005, excludes 50%** |
| games inside those matches | 789-657 = 54.6% | | CI overstated: a match's games share decks |
| Bo1, 2,400 games | 1241-1159 = 51.7% ±2.0% | +12, CI [−2, +26] | p ≈ 0.09 |

Matches went to 2 games 354×, 3 games 246×; nothing undecided.

**Read it carefully.** v14b's pregame decisions were never trained — its grown
net has never sideboarded, picked a battlefield or chosen a seat — so part of the
Bo3 margin measures *knowing the Bo3 decisions* rather than *playing games better*.
The Bo1 duel is the control for that: v15 lost nothing at plain games and probably
gained a little. Bo3 is the format the UI and real play use, so **v15-bo3 was
promoted to `rl/runs/current.pt`** on 2026-10-07.

`ab_checkpoints --bo3` is new with this run: one match per deal and seat, the
match dealt by `Bo3Dealer.for_slot`, scored by matches with games alongside.

## `v16-pay` — the payment choice: learned weakly, worth nothing measurable yet

**DONE 2026-10-07, 120/120 iterations in ~2.2h (resumed once at 70), 1.99M decisions,
0% truncated.** v15's recipe exactly — `--real-decks --victory 8 --envs 64 --rollout
12288 --holdout 7 --bo3`, pinned v7/v8/v10/v12 — warm-started FROM `v15-bo3/last.pt`,
with one new thing: **the payment choice (D6) ON** (`Config.choose_payment`, action
`A_PAY_SPLIT` — which runes pay a Power cost). It is the first run that could learn it;
pinned opponents and v15 itself play option 0, the old engine default
(`nets.untrained_choice`). 120 iterations so the LR anneal completes.

kl 0.074 on the first update → 0.020 (it 10) → 0.0066 (70) → 0.0014 (120), the usual
decay; `vs_opp` ewma between 48% and 54% throughout, final 53.0%; `vs_greedy` 67.5–76.5%
across the twelve evals, final 71.5% (saturated — not a yardstick). The `STALL` flag
fired on every post-eval iteration, so it is timing the eval, not sleep.

| duel vs v15-bo3 (sampled, `ab_checkpoints`) | result | Elo | |
|---|---|---|---|
| **Bo3, 600 matches** (`--bo3`) | **302-298 = 50.3% ±4.0%** | **+2, CI [−26, +30]** | **p ≈ 0.87** |
| games inside those matches | 762-761 = 50.0% ±2.5% | +0, CI [−17, +18] | CI overstated: a match's games share decks |
| Bo1, 2,400 games | 1168-1232 = 48.7% ±2.0% | −9, CI [−23, +5] | p ≈ 0.19 |

Matches went to 2 games 277×, 3 games 323×; nothing undecided. Logs
`rl/runs/v16_vs_v15_{bo3,bo1}.log`.

**Payment probe** (100 argmax self-play games, v16 both seats, real decks, payment
choice on): `knows_payment` is True for v16 and False for v15. **141 prompts (1.4 per
game), 60.3% chose something other than option 0** — 131 of them were 2-way, 9 3-way,
1 4-way. Zilean - Time Mage is half of all prompts (71, 46 deviating), then Ambessa (12),
Shuriken Flip (10), Counter Strike and Lightning Rush (7 each, both 7/7 deviating). A
typical case: Zilean, options *pay with Chaos* / *pay with Order* → chose Order at
p=0.71; Mirror Image, *Mind x2* / *Mind + Order* → chose Mind + Order at p=0.53. **But
the preferences are mild**: mean probability on option 0 is 0.46 against v15's exact
0.50 (untrained, uniform), and most chosen options sit at p=0.52–0.65. It moved off the
default; it has not formed strong opinions.

**Read it carefully.** Nothing here distinguishes v16 from v15 at either format: the
Bo3 interval excludes neither +25 nor −25 Elo, and Bo1 leans slightly the wrong way
without significance. So the payment decision — as learned in 120 iterations — is
worth **less than ~30 Elo in Bo3**, which is what you would expect from a decision that
arises 1.4 times a game and was half-learned. This does not show the decision is
unimportant in principle (Lillia's Lilting Lullaby case is real); it shows this run did
not turn it into strength. **Not promoted**: no evidence of a gain, a small hint of a Bo1
loss, and v15 stays `current.pt`. If the decision matters, a longer run or one weighted
toward decks that face it (Zilean, Lillia) is the way to find out.

## `v17-targets` — target rows say "mine" and "in combat": the first gain over v16

**DONE 2026-10-08, 300/300 iterations in 6h23m (22:29 → 04:52 PDT, inside `--max-hours
6.5`), 4.94M decisions.** v15's recipe exactly — `--real-decks --victory 8 --envs 64
--rollout 12288 --holdout 7 --bo3`, pinned v7/v8/v10/v12, payment choice on —
warm-started FROM `v16-pay/last.pt`, grown into the new action-row layout. The one new
thing: **every action about a permanent now says whether that unit is MINE and whether
it is IN THE CURRENT COMBAT** (`test_target_rows`). That bit was missing, and it is the
cause named in `REVIEW_HUMAN_GAMES.md` for the decisive blunders (Discipline on the
enemy instead of its own Crab; En Garde on a base unit while Yi defended alone).

kl 0.052 on the first update → 0.020 (it 10) → 0.014 (100) → 0.0049 (200) → 0.0015
(300); `vs_opp` ewma 49.5–56.6% throughout, final 56.1%; `vs_greedy` 64.0–78.5% across
30 evals, final 77.5% (saturated — not a yardstick). Truncation 0% except one overflow
episode at iteration 111 (1 of 103). The 29 `STALL` flags fall on iterations 11, 21, …,
291, the iteration after each eval, so they are timing the eval, not sleep.

**Damage rule.** Commit 6234b17 (engine files written 06:01 on 2026-10-08) made combat
damage lethal-in-full (465.2.c.3): "choose none" is offered only when one unit can absorb
the rest. **v17 trained under the OLD rule.** The complaint baseline (22:40), v17's
complaint run (05:49–05:51) and the Bo3 duel (process started 05:50) all imported the
old code too, so they compare like with like. **Only the Bo1 duel (started ~06:16) ran
under the new rule**, which v16 and v17 both face untrained.

| duel vs v16-pay (sampled, `ab_checkpoints`) | result | Elo | |
|---|---|---|---|
| **Bo3, 600 matches** (`--bo3`, old damage rule) | **330-270 = 55.0% ±4.0%** | **+35, CI [+7, +63]** | **p ≈ 0.014** |
| games inside those matches | 804-713 = 53.0% ±2.5% | +21, CI [+3, +38] | CI overstated: a match's games share decks |
| **Bo1, 2,400 games** (new damage rule) | **1309-1091 = 54.5% ±2.0%** | **+32, CI [+18, +46]** | **p ≈ 1e-5** |

Matches went to 2 games 283×, 3 games 317×; nothing undecided in either duel. Logs
`rl/runs/v17_vs_v16_{bo3,bo1}.log`. **Both intervals exclude 50%. This is the largest
measured gain since v10**, and v16 was itself no worse than v15, so v17 is about +30 Elo
over the promoted v15 by transitivity. That transitive figure has not been measured.

**Austin's complaints** (`behaviour.py`, 100 self-play games per agent, seeds
600000..600099 identical across agents; ratio, 95% cluster bootstrap over games, n/d).
v17 rows in `rl/runs/behaviour_complaints_v17.{log,json}`, the rest in
`behaviour_complaints_baseline.*`:

| | random | greedy | v14b | v15 | v16 | **v17** |
|---|---|---|---|---|---|---|
| buff given to an ENEMY unit | 37.7% [32,44] 132/350 | 67.5% [59,75] 168/249 | 17.9% [12,24] 35/196 | 18.4% [13,24] 38/206 | 14.1% [9,19] 29/205 | **10.3% [6,16] 24/233** |
| own-unit buff, never fights that ply | 75.7% [69,82] 165/218 | 75.3% [63,86] 61/81 | 47.2% [37,56] 76/161 | 49.4% [40,57] 83/168 | 41.5% [33,49] 73/176 | **49.8% [43,57] 104/209** |
| replay that changes nothing | 35.3% 6/17 | 60.0% 3/5 | 50.0% 1/2 | 42.9% 3/7 | 20.0% 1/5 | 80.0% 4/5 |
| must-win attack leaving units home | 35.2% [25,45] 25/71 | 40.7% [29,53] 22/54 | 17.9% [10,25] 21/117 | 25.2% [16,33] 29/115 | 27.3% [20,34] 33/121 | **25.7% [17,35] 28/109** |
| opponent's hold left uncontested | 63.2% [60,67] 385/609 | 42.5% [36,49] 237/558 | 33.1% [29,37] 224/677 | 30.5% [27,34] 198/650 | 27.5% [24,31] 173/628 | **22.8% [19,27] 148/650** |

Paired differences (same deals; resample game indices jointly, 4,000 draws):

| v17 minus | enemy buff | idle buff | must-win home | hold uncontested | combat not wiped |
|---|---|---|---|---|---|
| v16 | −3.8pp [−8.9, +1.3] p≈0.13 | +8.3pp [−1.1, +18.1] p≈0.09 | −1.6pp [−12.7, +9.4] | **−4.8pp [−9.1, −0.4] p≈0.03** | −5.2pp [−10.6, +0.3] p≈0.06 |
| v15 | **−8.1pp [−15.0, −0.9] p≈0.03** | +0.4pp [−9.6, +10.2] | +0.5pp | **−7.7pp [−11.9, −3.4] p≈0.001** | −1.7pp |

**Read it carefully.** The win rate moved a lot. The complaints the fix was aimed at
moved less clearly. **Buff on an enemy fell 14.1% → 10.3%.** That is the right direction
and the lowest of any agent, but it is not significant against v16 (p≈0.13). It is
significant against v15 (−8pp), and v15 is the champion Austin played. **The idle buff
did NOT fall: 41.5% → 49.8%**, back to the v14b/v15 level, which is a nominal worsening
(p≈0.09). The "in combat" bit has not taught it to time pump spells. This metric counts
any own-unit buff on a unit that does not fight that ply, so some of those may be
legitimate: lasting buffs, or buffs played as the defender. Must-win attacks are
unchanged. The pointless replay is too rare to read (n=5). **Contesting improved**:
holds left uncontested fell 27.5% → 22.8%, significant against v16 and v15. v17 also
fights a little more (35.5% of plies vs 34.4%) and commits fewer non-wiping attacks
(25.1% vs 30.4%).

**Promotion: recommended, not done** (Austin's call — `rl/ui/promote.py
rl/runs/v17-targets/last.pt`). It is the first checkpoint since v15 with a significant
win over its predecessor in BOTH formats. Its target choices are measurably better than
the champion's, and contesting improved. Idle buffs are the open complaint. **Step 4
(a hold-and-score punisher) is not needed now.** Contesting is the one complaint that
has improved monotonically v14b → v17 (33 → 31 → 27.5 → 23%) without it, and the scripted
holder opponent is too weak to train against anyway (CONTEXT: "loses to greedy 37%, to v16 23%").

## `v18-fairdmg` — the corrected damage rule and a refreshed pool: a small gain

**DONE 2026-10-08, 290/300 iterations (stopped on `--max-hours 6.5`; the last 10
would have run at an almost fully annealed LR, kl 0.0012), 0% truncated.** v17's
recipe warm-started FROM v17, with two changes: it is the first run under the
corrected combat-damage rule (465.2.c.3, `combat.dmg_may_stop`: no "kill
nothing" decline unless one unit can absorb the rest), and the pinned pool adds
**v15-bo3 and v16-pay** to v7/v8/v10/v12, because v17 beat the old four ~56% and
they had stopped teaching. First update kl 0.31 (the decline it relied on was
gone), 0.03 by iteration 2; `vs_opp` ewma 50% → ~53%; final `vs_greedy` 75.0%.

| duel vs v17-targets (sampled) | result | Elo | |
|---|---|---|---|
| Bo3, 600 matches | 310-290 = 51.7% ±4.0% | +12, CI [−16, +40] | p ≈ 0.41 |
| games inside those matches | 771-720 = 51.7% ±2.5% | | CI overstated |
| **Bo1, 2,400 games** | **1253-1147 = 52.2% ±2.0%** | **+15, CI [+1, +29]** | **p ≈ 0.03** |

Complaint rates (`behaviour.py`, 100 self-play games, all under the corrected
rule, idle-buff metric refined to cast spells whose buff serves nothing else):

| | v16 | v17 | v18 |
|---|---|---|---|
| buff given to an ENEMY unit | 14.1% | 10.3% | **8.6%** |
| buff spell, unit never fights that ply | 33.1% | 44.8% | 50.0% |
| must-win attack leaving units home | 28.1% | 25.2% | **20.6%** |
| opponent hold left uncontested | 27.3% | 22.7% | 23.8% |

**The idle-buff rise is mostly Discipline cast for its draw.** By spell (60
sampled self-play games each): Punch First wasted 9/18 (v16) → **0/17** (v18), En
Garde 14/18 → 10/23, while Discipline-as-cantrip rose 13/45 → 18/42. Pure-buff
waste fell; the metric does not yet separate a cantrip from a pure buff.

**Read:** a small, real gain in Bo1 and a positive lean in Bo3 that a 600-match
duel cannot resolve, plus the best enemy-buff and must-win rates of any version.
Logs `rl/runs/v18_vs_v17_{bo3,bo1}.log`, `rl/runs/behaviour_complaints_v18.log`.

## `v19-warmup` — the LR warm-up works mechanically; no strength gain

**DONE 2026-10-09, 100/110 iterations (stopped on `--max-hours 2.2`), trunc 0%.**
v18's recipe warm-started FROM v18, with `--lr-warmup 10` (BACKLOG W1) and
v17-targets added to the pinned pool. **The jump is gone:** iteration-1 kl
**0.0012** against v18's 0.31, rising smoothly to 0.025 at iteration 10 and
annealing to 0.0016. Final `vs_greedy` 77.0%, `vs_opp` ewma ~54%.

| duel vs v18-fairdmg (sampled) | result | Elo | |
|---|---|---|---|
| Bo3, 600 matches | 306-293 (1 undecided) = 51.1% ±4.0% | +8, n.s. | p ≈ 0.6 |
| games inside those matches | 745-731 = 50.5% ±2.6% | | |
| **Bo1, 2,400 games** | **1157-1243 = 48.2% ±2.0%** | **−12, CI [−26, +2]** | **p ≈ 0.08** |

Complaint rates (100 games each, same deals): enemy-buff 8.6 → 8.2%; idle buff
spell 50.0 → **37.7%**; must-win attack leaving units home 20.6 → **28.2%**
(worse, intervals overlap); uncontested hold 23.8 → 23.3%.

**Read.** A null, leaning slightly worse in Bo1. It is the v16 pattern again:
100 more iterations of the same recipe from a converged checkpoint do not move
strength. This run does NOT test whether the warm-up itself helps; that would
need the same run without it. The warm-up stays (it is cheap and removes a
known destructive first step), but it is not a strength lever. `current.pt`
stays v18. Logs `rl/runs/v19_vs_v18_{bo1,bo3}.log`,
`behaviour_complaints_v19.{log,json}`.
