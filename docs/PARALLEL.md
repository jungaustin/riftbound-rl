# Faster training: where the time goes, and the design to cut it

BACKLOG **P1/P2**. This is a design, not code yet. Measured 2026-10-09 on the M1 Max
(10 cores, 64 GB), v18 weights, 64 envs. Two duel processes and a behaviour run
were using the CPU at the same time, so the absolute times are pessimistic. The
ratios are the point.

## Where an iteration's time goes

One `collect()` of 4,170 learner steps (pure self-play) plus its `update()`:

| part | time | share | runs on |
|---|---|---|---|
| engine (`env.step`, minus encoding) | 4.3s | 26% | one CPU core, Python |
| observation encoding (`Encoder.encode`) | 2.8s | 17% | one CPU core, Python |
| learner forward (`net.act`) | 1.3s | 8% | CPU, 8 torch threads |
| batching and bookkeeping | 0.2s | 2% | one core |
| **PPO update** (4 epochs × 4 minibatches) | **8.0s** | **48%** | CPU, 8 torch threads |

A real run also has frozen-opponent forwards, which add a few percent. Applied to
a real iteration (rollout 12,288, about 19k env steps, about 80s), the shares
come out roughly like this:

| part | share of a real iteration |
|---|---|
| engine and encoding (single core) | ~50% |
| PPO update | ~30-35% |
| network forwards | ~15% |

**This corrects what I first told Austin.** I said the games are the bottleneck
and a GPU barely matters. Measured, the update is a third to a half of the time,
and the Mac's GPU does help:

| update on | first call | after warm-up |
|---|---|---|
| CPU | 8.1s | 7.3s |
| MPS (Apple GPU) | 13.8s (one-time compile) | **1.7s** |

That is 4.3× on the update.

## P1 — the update on the Apple GPU — BUILT 2026-10-09, modest gain

**Result:** `--update-device mps` works but gains less than the profile
promised. **MPS (torch 2.6) computes wrong gradients on large minibatches.** At
~800+ real transitions, the card encoder's WEIGHT gradients came out with norm
~1e7 and cosine 0.003 to the CPU's; biases and every other layer were exact. The
first smoke run therefore took random steps and diverged (kl 1.3 at iteration 9,
then 2,639 and 99,192; entropy 0.25 → 0.07). The small-batch unit test could not
see it. The fix feeds each minibatch to MPS in chunks of 256 (`MPS_CHUNK`) and
sums the gradients, which is the same update. That matches the CPU at full scale
(kl 0.1117 vs 0.1107), but chunking costs speed: **8.2s vs the CPU's 14.1s per
update, 1.7x, so ~1.15x per iteration.** A newer torch may fix MPS and restore
the 4.3x; test it in a separate venv before trusting it. Not yet run in a real
training run.

### Original plan

- Keep a master copy of the net and its Adam state on `mps`, and a mirror on
  `cpu` for `collect()`. Small batches of single-step forwards are faster on the
  CPU than on MPS once transfer is counted.
- After each `update()`, copy the weights to the CPU mirror. That is about 4 MB,
  so it is negligible.
- Flag: `--update-device mps`, default `cpu` until it is validated. The existing
  `--device` keeps meaning "everything".

**Validate before trusting it:**

1. From the same weights and the same rollout, one update on CPU and on MPS
   gives close losses and KL, and close weights (float32 tolerance).
2. A 20-iteration smoke run shows no NaN guard trips, with kl and ev on the
   normal curve.
3. `resume.pt` round-trips. The Adam state is saved on CPU and moved on load.

**Expected:** iteration time ×~0.75, about **1.3×** faster on its own.

## P2 — the games in worker processes — BUILT 2026-10-09, ~2x per iteration

**Result:** `ppo.py --workers N` runs the games in `rl/parallel.py`'s
`ParallelVec` (default 0 = the serial `VecRiftbound`, unchanged). Same seed,
same games: `tests/test_parallel.py [1]` checks every observation array,
reward, done flag, episode summary and `continuing` against the serial env for
1,500 steps x 8 envs, Bo1 and Bo3 (39 later match games), and a 3-iteration
`--bo3` smoke run printed identical training metrics at `--workers 8` and 0.
Measured on the M1 Max with a UI server and other light load running (load
average 3-6), 64 envs, Bo3, victory 8:

| | workers 0 | 4 | 6 | 8 |
|---|---|---|---|---|
| env steps/s, random actions | 389 | 833 | 1018 | 1124 |
| `Trainer.collect()`, rollout 4096, untrained net | 25.9s | | | 7.8s (3.3x) |
| `ppo.py 3 ... --bo3`, avg st/s incl. CPU update | 324 | | | 664 (2.05x) |

**Why 8 workers give ~3x, not ~8x: one slow game holds the lockstep.** Per env
step the cost in a worker equals the serial cost (2.5 ms; no contention), but
the busiest worker did 2.4x the mean worker's work. Measured serially: env.step
p50 1.2 ms, p99 21 ms, max 497 ms, and ONE slot's game took 11 of 47 engine
seconds. Profiled, 10.3 of that game's 13.4s is `GameState.attachments`'s
Python list comprehension (515k calls, ~20 us each) under `row_statics_sourced`
-- Equipment on a ~17-permanent board makes every board walk O(rows^2) in numpy
scalar reads. Vectorizing `attachments` (or caching it per state version) is the
next speed lever and helps serial too; pipelined groups (P3) only hide it.

**Deviations from the design, measured:**

- **Torch threads are not capped during collect.** Lockstep never overlaps the
  forwards with the workers, so a cap only slows the forwards: collect at
  workers 8 took 8.6s / 7.8s / 7.4s / 7.3s at 1 / 2 / 4 / 8 threads.
- **`Bo3Dealer.matches` / `.games` counters** live in the workers and are not
  reported back (nothing reads them).
- The `BatchObs` returned by `ParallelVec.step` IS the shared memory and is
  overwritten by the next step. `slice_obs` copies, so `collect` is safe;
  anything new that keeps an observation across a step must copy it.
- `ParallelVec.busy` / `.wall` accumulate per-worker step seconds and main's
  wall time in `step`, for the straggler diagnosis above.

### Original design

### Shape

```
main process                                 worker k  (k = 0..W-1, spawn)
-------------------------------              ---------------------------------
learner net (CPU mirror)                     full_table(), Encoder, deal_fn
frozen opponent nets, pool, PFSP             envs for slots [k*S, (k+1)*S)
Rollout, GAE, update (MPS)                   step(actions) -> encode -> write
                                               obs into SHARED MEMORY, reply
   actions[64] --pipe--> each worker           rewards, done, episode summaries
   <--pipe-- "done" + small per-slot info      over the pipe (small)
   reads obs straight from shared memory
```

`ParallelVec` has the same interface as `VecRiftbound` (`reset`, `step`,
`shapes`, `truncation_rate`, `episodes`), so `Trainer.collect()` does not change.
`Trainer` picks the class by `--workers N`, and `--workers 0` is the current
serial path.

### Decisions, and why

- **Shared memory for observations, pipes for everything else.** One env's
  observation is about 140 KB (legal-action rows 78 KB, privileged critic input
  63 KB). 64 envs is about 9 MB per step, and there are hundreds of steps per
  iteration. Pickling that through a pipe would eat the gain. Each worker writes
  its slots' rows into preallocated `multiprocessing.shared_memory` arrays
  shaped from `enc.shapes()`. Main wraps the same buffers as the `BatchObs`
  arrays, with no copy. Actions, rewards, done flags and episode summaries are
  tiny and go over the pipe.
- **Lockstep first, pipelined later.** Every step, main sends all actions, waits
  for all workers, then runs the forwards. That is simple and deterministic. The
  next refinement is two groups of envs, A and B: the net thinks for A while the
  workers step B. That hides the forward time, but it changes how often each
  slot moves, so it waits until lockstep is proven.
- **Dealing happens in the worker.** `Bo3Dealer.for_slot` needs the slot's
  finished `env.state` to continue a match, and that state lives in the worker.
  `continuing(i)` is reported back with the done flag.
- **`--matchup-pfsp` is refused with `--workers`.** Its dealer takes per-deal
  decisions from main's pool (`take()`). It is off by default and was a measured
  null (#21), so it is not worth plumbing until something needs it. The default
  pool sampling (`_assign`) stays entirely in main, untouched.
- **Seeding becomes per slot:** `seed0 + slot + n_envs × (episodes in that
  slot)`. The current single global counter depends on the order games finish
  in, which differs between serial and parallel. Per-slot seeds make the two
  produce identical games. That identity is the test.
- **Torch threads.** During `collect`, main keeps 1-2 torch threads, so workers
  get the cores. The update runs on MPS anyway.
- **Workers.** Start with 8 for 64 envs (8 slots each). The M1 Max has 8
  performance cores and 2 efficiency cores, so main plus 7-8 workers is the
  ceiling. Measure 4, 6 and 8.
- **Failure handling.** If a worker raises, main surfaces its traceback and the
  run stops cleanly. `resume.pt` already holds everything that matters, and
  env states are never in it. Workers are daemons, so a dead main never leaves
  orphans.

### What has to move or change

| piece | today | with workers |
|---|---|---|
| `VecRiftbound` | owns the envs | unchanged; `ParallelVec` is a sibling class |
| `deal_fn` / `Bo3Dealer` | built in main | built in each worker from picklable arguments (deck names, seed, mode) |
| `Trainer.__init__` deal-queue drain | pops `n_envs` deals | becomes a no-op when the dealer lives in the workers (matchup PFSP only) |
| episode summary (`env.summary()`) | read from the env | sent back with the done flag |
| eval (`rl/eval.py`, every 10 iterations) | serial | unchanged at first; the next candidate after P2 |
| `ab_checkpoints.py`, `behaviour.py` | serial | easy follow-on: shard seeds over processes (no nets to share) |

### Tests (`rl/tests/test_parallel.py`)

1. **Identity.** Under scripted random actions, `ParallelVec(workers=3)` and
   `VecRiftbound` with the same slot seeds produce identical observations,
   rewards and done flags for 2,000 steps, Bo1 and Bo3.
2. **Bo3 continuity.** A match's game 2 continues in its own slot with
   `seen_match` carried over.
3. **Clean shutdown.** A worker exception reaches main with its traceback.
   Shared memory is unlinked on exit, including on Ctrl-C.
4. **Speed.** env steps per second at workers 0, 4, 6 and 8, printed and not
   asserted (the machine varies).

### Expected gain

Engine and encoding are about 50% of an iteration. Spread over about 7
effective workers, that part becomes about 7%, plus some overhead. Together with
P1:

| | relative time | speed |
|---|---|---|
| today | 100% | 1× |
| P1 only | ~75% | ~1.3× |
| P2 only | ~55% | ~1.8× |
| **P1 + P2** | **~35%** | **~2.5-3×** |
| + pipelined groups (hide forwards) | ~25-30% | ~3.5× |

These are estimates from the profile above. Each step is measured as it lands.
Then the network forwards and the per-step overhead become the next limit, not
the engine.

## Order

1. **P1** (MPS update), then validate.
2. ~~**Seeding per slot** in `VecRiftbound`.~~ Done (`vec.slot_seed`).
3. ~~**`ParallelVec` with lockstep**, plus `test_parallel.py`.~~ Done, ~2x.
3b. **Vectorize `GameState.attachments`** -- the straggler that caps step 3.
4. Run one short real training run with `--workers 8 --update-device mps`, and
   compare its curve to v19's first 20 iterations.
5. Later: pipelined groups, then parallel eval, duels and behaviour.

## Step 3b — `GameState.attachments` vectorized (2026-10-09)

The straggler that held every `--workers` lockstep was a Python list
comprehension called ~3M times per few hundred games. Now it is one
`np.flatnonzero` pass. A fixed random rollout (32 envs, 1,500 steps) took
**169.3s → 105.7s (1.6x), with identical observations (same sha256)**, and the
engine suites pass. That helps the serial path as much as the parallel one.
After it, the profile is flat: no single function is above ~7% of the time
(`_board_cost_mods`, `GameState.clone`, `row_statics_sourced`, the encoder's
`_board`/`_decks`).
