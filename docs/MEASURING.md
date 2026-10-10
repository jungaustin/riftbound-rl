# Measuring the agent

How to tell whether a change helped. This is a separate document because getting
it wrong cost two runs and a week of wrong conclusions.

---

## The lesson that produced this page

For most of this project, strength meant one number: **win rate against
`agents/greedy.py`**. Then v10-exploiter beat its own predecessor **56.3%** in
head-to-head play while `vs_greedy` sat still — 70.5% before, 68.3% after, inside
noise.

The policy got better and the metric could not see it. So "71% is this
architecture's ceiling", stated twice on the strength of that number, was a
statement about the ruler.

**Why it is structural, not a tuning problem.** Greedy attacks only when it can
wipe every defender. It can therefore never:

- swing one unit at a time,
- make a deliberate losing trade,
- kill a specific blocker to open a line next turn.

A policy can learn all three and beat greedy by *exactly* as much as before.
`vs_greedy` saturates, and the saturation point is a property of greedy.

**Keep it anyway**, for two things it is still good at: it is a cheap regression
alarm (a policy that falls below ~60% has broken, not plateaued), and it is a
fixed point that never drifts, which makes it a usable anchor — see below.

---

## The ladder — `rl/ladder.py`

Rate policies against **each other**.

```bash
# Round-robin over a handful of checkpoints
PYTHONPATH="$PWD" python3 rl/ladder.py --deals 25

# Look INSIDE a run: was it still improving when it stopped?
PYTHONPATH="$PWD" python3 rl/ladder.py --snapshots v11-pool --ref v8-postfix
```

**Greedy is pinned at 0.** Elo fitted inside a moving population can rise while
every member gets weaker — everyone beating everyone slightly better proves
nothing. That warning is already at the top of `pool.py`, and it is why greedy and
random stay in the ladder as rungs that never train. A rating here means the same
thing across runs and across months.

### Things that are easy to get wrong here

**Fit all pairings at once, not game by game.** `pool.elo_update` applied
sequentially gives an answer that depends on the order results arrive, which for a
set of already-finished matches is an artefact. `ladder.fit_elo` does a
Bradley-Terry **MM** fit instead: monotone, no step size.

**Do not use gradient ascent for it.** The residual is in units of *games*, so
any learning rate large enough to move a rating also diverges. This was a real bug
in the first version of the file.

**A 100% result has no finite rating.** A half-win prior against the anchor keeps
the fit defined.

**`--anchors-only` for many players.** O(n) pairings rather than O(n²); a
round-robin over 50 snapshots is 1,225 pairings. `--snapshots` implies it.

**`--ref` is worth setting.** Greedy is a weak anchor, and a strong rung pins the
top of the scale far better than extrapolating from the bottom.

### Snapshots are what make a run legible from the inside

`ppo.py` takes a pool snapshot every `--pool-every` iterations and, since
2026-10-01, **keeps it on disk** (`<out>/snapshots/iterNNNNN.pt`, ~4MB each).

Before that the pool was memory-only, so every run generated 20–50 intermediate
policies and discarded all of them, leaving `best.pt` and `last.pt`. That is the
whole reason "did v7 plateau?" was unanswerable: the evidence was created and
thrown away.

A resume also **rebuilds the pool from those snapshots**. Without that, a resumed
run restarts the pool empty and quietly becomes pure self-play until `pool_every`
refills it — the only visible symptom is `vs_opp` disappearing from the log.

### The baseline scale (2026-10-01, 50 games per pairing)

| player | Elo | vs greedy |
|---|---|---|
| **v10-exploiter** | **+133** | 68.0% |
| v7-victory8 | +91 | 60.0% |
| v8-postfix | +87 | 64.0% |
| greedy | 0 | — |
| random | −779 | 0.0% |

Three things worth reading off it:

1. **v7 and v8 are tied** (+91 vs +87, 4 Elo apart). That is the same verdict the
   three separate measurements reached — the engine fixes cost nothing and bought
   nothing — now on one scale instead of three incomparable ones.
2. **v10 leads by ~45 Elo.** Corroborated independently: the 300-game duel put it
   at 56.3% ±5.6 over v8, which is +44 Elo. Two methods, same answer.
3. **random is 779 Elo below greedy.** Greedy is a weak *ceiling* but not a weak
   player, which is why `vs_random ≥ 90%` was never the hard part of Phase 4.

**Take these as ±50 Elo.** 50 games per pairing is ±13–14pp, which is enough to
order the lineage but not to split v7 from v8 — and in fact they should not be
split. Raise `--deals` before quoting any single number.

**The lineage spans ~46 Elo** (v8 +87 to v10 +133) while greedy-to-v8 is +87. So
most of the distance this project has travelled was covered by v7, and everything
since has been incremental.

---

## Head-to-head duels — `rl/tests/ab_checkpoints.py`

```bash
PYTHONPATH="$PWD" python3 rl/tests/ab_checkpoints.py 150 A/best.pt B/best.pt
```

Seats swap every deal, because the first-player edge would otherwise decide it.

**Always run the control.** A checkpoint against *itself* must come out at 50%; if
it does not, the harness is biased and every number from it is suspect. When this
mattered — validating v10 — the control came back 50.3% ±5.7 and the result
56.3% ±5.6, so the 6-point gap was real.

---

## Verify the config before trusting any of it

v9-exploiter burned four hours measuring nothing because it was launched without
`--spells` while its frozen champion had been trained with them. `ck["shapes"]`
could not catch it: the observation layout is byte-identical either way, so **a
rules difference is invisible to a tensor shape.**

What exists now:

- `--real-decks` **implies** spells; `--units-only` is the explicit escape hatch
  and the two contradict with an error.
- Every run's header prints **which game it is playing**.
- Checkpoints carry the full `Config`, and `--exploit` refuses a mismatch while
  `--init` refuses a `units_only` mismatch.

**Two independent smell tests, for logs that predate the fingerprint:**

| | |
|---|---|
| **decisions per turn** = `len / turns` | ~16–20 with spells, ~5 without |
| **the first eval** | a warm start from a champion must reproduce the champion |

That second one is the cheap one and it is the one I missed. v9's first eval read
`vs_greedy 55.0%` at iteration 20 with kl ~0.005 — a policy that has not moved
cannot lose 15 points by drifting, so an instant drop is an environment change.
Three hours of probing for a subtle credit-assignment bug followed.

---

## Sizing a ladder run — learned by getting it wrong

The v11 snapshot ladder was run at `--deals 15` (30 games per pairing, 3 pairings
per player, so ~90 games per rating) and **produced nothing usable**. The tell was
in the output: different snapshots came back with *identical* Elo, because at that
sample size their win/loss records collide exactly and the fit quantizes.

| games per player | SE | resolves |
|---|---|---|
| ~90 | ±10pp | ±~90 Elo — nothing |
| ~300 | ±5.6pp | ±~40 Elo — a large change |
| ~600 | ±4pp | ±~28 Elo |
| ~1,200 | ±2.8pp | ±~20 Elo |

**The lineage's per-run increments are ~30 Elo**, so a curve needs ~600 games per
rung. Ten rungs is then ~6,000 games, which is hours.

**So prefer few rungs with many games.** Three well-powered points show where a
run flattened; ten underpowered ones show nothing, and look like a result.

Better still for a yes/no question: **one direct duel.** `ab_checkpoints.py` at 150
deals is 300 games against a single opponent, which beats spreading the same games
across three anchors.

## `best.pt` was selecting on noise

Found 2026-10-01. `best.pt` was the max of `vs_random + vs_greedy` over a run's
evals. But `vs_greedy` is saturated, so across a run it is a flat noisy series --
v11's 14 evals had mean 71.2% and **sd 2.7**, and a single eval at n=60 is ±6pp.
Taking the max of 14 such draws is taking the largest noise excursion: it picked
**iteration 20 of 280.**

That propagates. `best.pt` is what feeds the next run's `--init`, `play.py` and the
ladder, so every downstream use inherits one lucky eval.

Now selected on an **ewma** of the eval series, carried across resumes, with the
chosen value stored in the checkpoint as `selected_on`. For a run whose metric is
saturated and which did not diverge, **`last.pt` is the more honest artefact** --
the old advice to always prefer `best.pt` dates from v5, when drift was real and
the metric was not yet saturated.

## Validate a filter against the thing it filters for

2026-10-02, building the #28 solver. Its cheap gate (`could_win_now`) exists to
skip decisions that cannot hold a forced win, and skipping a real one would
silently defeat the whole solver. First check: 12 games, **"0 wins skipped"** --
except there were **0 wins in those 1,582 decisions**, so the check had nothing to
be wrong about.

The probe's own data said why, and it is worth knowing in its own right: **forced
wins cluster.** 339 of 400 games hold none; 61 games hold all 184; one game holds
10. A uniform sample of a clustered rare event mostly samples the empty part.

Re-run against the games the probe had already found wins in, the gate let through
2 of 2 and skipped none -- a real check, on a sample chosen to be able to fail.

**So: a filter is validated on inputs it is supposed to pass, not on a sample
that happens to contain none of them.**

## Co-occurrence is not causation, and a paired test separates them

2026-10-02. The lethal probe found that in 31 of 400 games the policy threw away a
forced win, and that it **lost 15 of those 31**. I sized BACKLOG #28 at "a miss
coincides with losing in **3.8% of games**" and recommended building the solver on
that basis — twice, and once as the highest-value item in the project.

The word doing the work there is *coincides*. A paired test — the same 31 games
with and without the solver, identical seeds — moved **zero** games. The solver
proved 34 wins, overrode 20 moves, and every proof held: in the 4 games where it
fired, seat 0 won, and it would have won them anyway.

**Throwing away a forced win on turn N does not lose the game if you win on turn
N+2.** The misses were real and the losses were real; they were not the same games
in any causal sense.

**So: a co-occurrence estimate is an upper bound on an effect, not a measurement of
it.** The cheap way to tell the difference is a paired A/B on the subset where the
mechanism can act — which is also far more sensitive than a fleet-wide duel. 62
games settled what 700 would have been needed to see at fleet level.

**Two confounds worth naming**, because both inflated my estimate: the probe
searched for *either* seat while the solver only runs on seat 0, so roughly half
the misses were never addressable; and the probe's trajectories differ from the
A/B's, so the same game index does not always reach the same position.

## A zero is a sample too

2026-10-02, while scoping #28: a forced-win probe returned **0 hits in 747
decisions**, and I wrote that down as "a forced win that never cedes priority does
not exist", with a tidy structural argument for why. The same probe over **4,209**
decisions found **3**.

Nothing was wrong with the reasoning; it was stated at a confidence the sample
could not carry. A rate of 0.07% needs thousands of trials before zero means
anything — at 747 decisions, seeing none was the *expected* outcome of a real
effect.

**So: an absence is a measurement with an interval, and it gets the same
treatment as a presence.** 0/747 bounds the rate at under ~0.4%; it does not
establish it is zero.

## Behavioural metrics -- `rl/tools/behaviour.py`

BACKLOG 15b, built 2026-10-04. Elo says a policy got stronger; it cannot say
*what changed*. This drives real games and reports what a policy **does**.

```bash
PYTHONPATH="$PWD" python3 rl/tools/behaviour.py --games 200 \
    random greedy rl/runs/v7-victory8/last.pt rl/runs/v10-exploiter/last.pt \
    rl/runs/v11-pool/last.pt rl/runs/v12-entropy/last.pt
PYTHONPATH="$PWD" python3 rl/tools/behaviour.py --render rl/runs/behaviour.json
```

Shape copied from `lethal_probe.py`: real games through `RiftboundEnv`, every
decision of the agent's own seats interrogated, per-game rows cached to
`rl/runs/behaviour.json` every 10 games, progress with an ETA. ~0.8s/game, so a
200-game column is ~3 minutes and the whole table above is ~20.

**All agents on the same footing**: same deck pool, same deals, same seeds, and
**self-play by default**, so a number is a property of the agent rather than of
whoever it was matched against. `random` and `greedy` are columns because a
policy compared only with itself says nothing.

### The baseline table (2026-10-04, 200 games per agent, self-play, victory 8 with spells)

| | random | greedy | v7 | v10 | v11 | v12 |
|---|---|---|---|---|---|---|
| **2+ committed combats / combat-turn** | 8.7% | 7.4% | **7.3%** | 11.1% | **12.7%** | 11.1% |
| ... at 2+ distinct battlefields | 2.2% | 3.3% | 4.4% | 7.9% | 8.2% | 8.0% |
| ... same from the engine's combat log | 5.2% | 3.6% | 12.6% | 16.0% | 16.9% | 17.2% |
| combat-turns / own turn | 13.7% | 21.7% | 33.4% | 33.2% | 33.5% | 34.1% |
| **committed combat it does not wipe** | 55.2% | **0.0%** | 46.1% | 38.6% | 39.3% | **30.4%** |
| committed combat, Might deficit | 54.3% | 1.2% | 45.7% | 38.0% | 39.0% | 31.0% |
| declined an immediate score | 16.8% | 8.4% | 2.0% | 2.3% | 3.9% | 0.0% |
| ended a turn with a Move available | 15.7% | 1.8% | 2.7% | 2.6% | 2.8% | 2.3% |

Denominators are 500-1,000 combat-turns and 950-1,000 committed combats per
column, so a column is worth +-2-3pp; the full numerator/denominator pairs and a
95% cluster bootstrap over games are in the tool's own output.

**Two things fall out of it, and both are new information:**

1. **Multi-combat sequencing appeared between v7 and v10, and nothing else in
   this project could see it.** v7 sits at the random/greedy level (7.3% vs 8.7
   and 7.4) -- the same verdict as v5's 10% -- while v10/v11/v12 run 11-13%, and
   the distinct-battlefield variant nearly doubles (4.4% -> 8%). Those intervals
   do not overlap. The exploiter run is where it arrived. **v10 -> v11 -> v12 is
   not resolved at this sample** (11.1 / 12.7 / 11.1, intervals overlapping).
2. **v12-entropy changed its combat selection even though its win rate did
   not.** Non-wiping attacks 39.3% -> 30.4%, intervals disjoint, on a run whose
   `vs_greedy` was flat and whose kl/ent decayed to v11's. That is exactly the
   situation this instrument was built for: *a null in Elo with a change in
   behaviour is a different situation from neither moving.* It says nothing about
   whether the change is good -- 30% is not obviously better than 39% -- only
   that the policy moved.

**The confound to quote with it:** self-play means each column faced a different
opponent, so the *rates per turn* are not strictly comparable across columns
(a policy that fights more also offers its mirror more to fight). Within a
column -- and for a yes/no "did the behaviour change" between two runs -- it is
sound. `--opp greedy` pins the opponent instead, at the cost of measuring
behaviour against a weak player.

### The definitions, and three ways they were wrong first

Full text in the module docstring. What matters here is that **the examples, not
the numbers, found the errors** -- the tool prints the first few events with
`viewer.describe`, which is the only reason they were caught:

- *committed combat* = chose `A_COMMIT` into a battlefield where the opponent had
  a live Unit present (461 then stages the Combat). Cross-checked against the
  engine's own `combat_at` log, which counts ~50% more: those come from Chain
  resolution (effect moves, Ambush, triggers that drag a unit in), not from a
  declared Move. Both counters are reported, with a breakdown of which action
  kind the engine first logged the extra ones under.
- *does not wipe* = the attacking pool does not kill every defender by
  `solve_kills`. **Greedy must read 0% and reads 0/644** -- the validation of the
  detector against a case whose answer is known in advance. Note that greedy's
  *Might deficit* is 8/644, not zero: the two metrics really are different.
- *declined a score*: the naive "a point was one action away and it played
  something else" was **mostly false positives** -- `ADD` before `COMMIT`, and
  Conquer points that land after the Chain resolves rather than inside the apply.
  Fixed by re-searching from the position the chosen action reaches (opponent
  windows driven with PASS, depth 3) and counting a still-reachable point as a
  *deferral*, outside the denominator. That took random from 53% to 17%.

Per-ply counting also has a trap worth stating: a ply key is only created when
the agent is the **active** player. Counting the plies where it merely held
priority as a defender inflates the denominator ~25%, since it can only commit a
Move on its own turn.

**Uncertainty is cluster-bootstrapped over games**, because plies inside a game
are correlated. Measured here: 2+-combat turns are only mildly clustered (38-82
of 200 games hold one, at most 4 in any game), so the bootstrap and a naive
binomial agree -- which is worth knowing, given that forced wins were 61-of-400.

## What is still missing

| | what | why it matters |
|---|---|---|
| **per-deck `n`** | BACKLOG 15. n=24 gives ±8–20% per deck: the spread is real, the ordering is not | Any single deck's number is currently a story, not a measurement |
| **paired per-deck lift** | `per_deck` uses `seed0=500_000`, `per_deck_baseline` uses `700_000`, so the lift is not paired | ~10 minutes to fix and it makes the comparison honest |
