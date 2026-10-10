"""The vector env in worker processes is the same vector env (BACKLOG P2).

    PYTHONPATH=. python3 -m rl.tests.test_parallel [--no-speed]

  [1] identity: `ParallelVec(workers=3)` and `VecRiftbound`, same seed0, real
      decks, driven by the same random legal actions, produce identical
      observations (every BatchObs array), rewards, done flags, episode
      summaries and `continuing` -- Bo1, then Bo3 (where game 2+ of a match
      must actually occur, so match continuity is covered by the identity)
  [2] a worker exception reaches main as WorkerError carrying the worker's
      traceback; the shared-memory block is unlinked and the workers are gone,
      and the same holds when main is stopped by Ctrl-C
  [3] speed, printed and not asserted: env steps/sec at workers 0/4/6/8 for
      64 envs under random actions, and one Trainer.collect() of rollout 4096
      with an untrained net at workers 0 vs 8
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from rl.config import Config
from rl.engine.cardtable import full_table
from rl.parallel import DealSpec, ParallelVec, WorkerError
from rl.ppo import select_decks
from rl.vec import VecRiftbound

T = full_table()
CFG = replace(Config().at_victory_score(8), units_only=False,
              choose_payment=True)
POOL = select_decks(T, None, 0.6)
FAILED: list[str] = []


def ok(name):
    print(f"  \033[32mPASS\033[0m {name}")


def die(name, msg):
    print(f"  \033[31mFAIL\033[0m {name}: {msg}")
    FAILED.append(name)


def gone(name: str) -> bool:
    """Is the shared-memory block unlinked? (macOS has no /dev/shm to list.)"""
    try:
        s = SharedMemory(name=name)
    except FileNotFoundError:
        return True
    s.close()
    return False


def random_actions(rng, b) -> np.ndarray:
    return np.array([int(rng.integers(int(n))) for n in b.n_legal], np.int64)


def compare(where: str, a, b) -> str | None:
    """First difference between two BatchObs, or None."""
    pairs = [("globals", a.globals, b.globals),
             ("legal_actions", a.legal_actions, b.legal_actions),
             ("action_mask", a.action_mask, b.action_mask),
             ("to_move", a.to_move, b.to_move),
             ("n_legal", a.n_legal, b.n_legal),
             ("privileged", a.privileged, b.privileged)]
    if a.zones.keys() != b.zones.keys():
        return f"{where}: zone keys differ"
    for k in a.zones:
        pairs.append((f"zones[{k}]", a.zones[k], b.zones[k]))
        pairs.append((f"zone_mask[{k}]", a.zone_mask[k], b.zone_mask[k]))
    for name, x, y in pairs:
        if x is None or y is None:
            if (x is None) != (y is None):
                return f"{where}: {name} is None on one side only"
            continue
        if x.dtype != y.dtype or x.shape != y.shape:
            return f"{where}: {name} {x.dtype}{x.shape} vs {y.dtype}{y.shape}"
        if not np.array_equal(x, y):
            rows = sorted({int(r) for r in np.argwhere(x != y)[:, 0]})
            return f"{where}: {name} differs in slots {rows}"
    return None


def identity(label: str, spec: DealSpec, steps: int, n: int = 8,
             workers: int = 3, seed0: int = 17) -> None:
    name = f"identity-{label}"
    ser = VecRiftbound(T, CFG, n, spec.build(T), seed0=seed0)
    par = ParallelVec(T, CFG, n, spec, seed0=seed0, workers=workers)
    rng = np.random.default_rng(seed0)
    try:
        bs, bp = ser.reset(), par.reset()
        err = compare("reset", bs, bp)
        episodes = continuing = 0
        for t in range(steps):
            if err:
                break
            act = random_actions(rng, bs)
            bs, rs, ds, is_ = ser.step(act)
            bp, rp, dp, ip = par.step(act)
            err = compare(f"step {t}", bs, bp)
            if err is None and not (np.array_equal(rs, rp)
                                    and np.array_equal(ds, dp)):
                err = f"step {t}: rewards/done differ"
            if err is None and is_ != ip:
                err = f"step {t}: infos differ"
            for i in range(n):
                if err is None and ser.continuing(i) != par.continuing(i):
                    err = f"step {t}: continuing({i}) differs"
            episodes += int(ds.sum())
            continuing += sum(par.continuing(int(i))
                              for i in np.flatnonzero(dp))
        if err is None and (ser.episodes, ser.truncations) != (
                par.episodes, par.truncations):
            err = (f"counters: serial {ser.episodes}/{ser.truncations} vs "
                   f"parallel {par.episodes}/{par.truncations}")
        if err:
            die(name, err)
        elif episodes < 5:
            die(name, f"only {episodes} episodes finished; the test proved "
                      f"nothing about auto-reset")
        elif spec.kind == "bo3" and continuing == 0:
            die(name, "no slot ever reached game 2 of a match")
        else:
            ok(f"{label}: {steps} steps x {n} envs over {workers} workers, "
               f"{episodes} episodes ({continuing} later games of a match): "
               f"every array, reward, done flag and summary identical")
    finally:
        par.close()


CHILD = r"""
import sys, time
from dataclasses import replace
import numpy as np
from rl.config import Config
from rl.engine.cardtable import full_table
from rl.parallel import DealSpec, ParallelVec
from rl.ppo import select_decks
T = full_table()
cfg = replace(Config().at_victory_score(8), units_only=False)
v = ParallelVec(T, cfg, 4, DealSpec("pool", select_decks(T, None, 0.6), 0.6),
                workers=2)
v.reset()
print(v.shm.name, *[p.pid for p in v._procs], flush=True)
while True:
    b = v.step(np.zeros(4, np.int64))[0]
"""


def main() -> None:
    # ---------------------------------------------------------------------------
    print("\n[1] identity with the serial vector env")
    identity("bo1", DealSpec("pool", POOL, 0.6), 1500)
    identity("bo3", DealSpec("bo3", POOL, 0.6), 1500)


    # ---------------------------------------------------------------------------
    print("\n[2] a worker exception surfaces in main, and nothing leaks")
    par = ParallelVec(T, CFG, 6, DealSpec("pool", POOL, 0.6), seed0=0, workers=3)
    shm_name, procs = par.shm.name, list(par._procs)
    b = par.reset()
    bad = np.zeros(6, np.int64)
    bad[4] = 10_000                     # slot 4 lives in the last worker
    try:
        par.step(bad)
        die("worker-error", "an out-of-range action did not raise")
    except WorkerError as e:
        msg = str(e)
        if "Traceback" not in msg or "outside the" not in msg or "env.py" not in msg:
            die("worker-error", f"no worker traceback in the message:\n{msg}")
        elif "worker 2" not in msg:
            die("worker-error", f"the failing worker is not named:\n{msg[:300]}")
        else:
            ok("an out-of-range action in worker 2 raised WorkerError with the "
               "worker's own traceback (env.py's assertion)")
    time.sleep(0.2)
    if not gone(shm_name):
        die("worker-error", f"shared memory {shm_name} still exists after the error")
    elif any(p.is_alive() for p in procs):
        die("worker-error", "a worker is still alive after the error")
    else:
        ok("after the error the shared block is unlinked and every worker exited")
    try:
        par.step(np.zeros(6, np.int64))
        die("worker-error", "stepping a closed ParallelVec did not raise")
    except WorkerError:
        ok("a closed ParallelVec refuses to step")

    # Ctrl-C: a child Python makes a ParallelVec, reports its block, and is
    # interrupted mid-run. Its exit must unlink the block and leave no workers.
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    child = subprocess.Popen([sys.executable, "-c", CHILD], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    line = child.stdout.readline().split()
    if not line:
        die("ctrl-c", "child never started:\n" + child.stderr.read()[-2000:])
    else:
        c_shm, pids = line[0], [int(x) for x in line[1:]]
        time.sleep(1.0)
        child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            die("ctrl-c", "the child did not exit on SIGINT")
        time.sleep(0.5)
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        err = child.stderr.read()
        if not gone(c_shm):
            die("ctrl-c", f"{c_shm} leaked after Ctrl-C")
            try:                              # do not leave it behind ourselves
                s = SharedMemory(name=c_shm); s.close(); s.unlink()
            except FileNotFoundError:
                pass
        elif alive:
            die("ctrl-c", f"workers {alive} outlived main")
        elif "KeyboardInterrupt" not in err:
            die("ctrl-c", f"child did not stop by KeyboardInterrupt:\n{err[-1500:]}")
        elif "leaked shared_memory" in err:
            die("ctrl-c", f"resource tracker reports a leak:\n{err[-1500:]}")
        else:
            ok("Ctrl-C in main: block unlinked, workers gone, no leak warning")


    # ---------------------------------------------------------------------------
    if "--no-speed" not in sys.argv:
        print("\n[3] speed (printed, not asserted -- the machine is shared)")
        spec = DealSpec("bo3", POOL, 0.6)
        N, STEPS = 64, 300
        for w in (0, 4, 6, 8):
            vec = (VecRiftbound(T, CFG, N, spec.build(T), seed0=0) if w == 0
                   else ParallelVec(T, CFG, N, spec, seed0=0, workers=w))
            rng = np.random.default_rng(0)
            b = vec.reset()
            for _ in range(20):                       # warm-up
                b = vec.step(random_actions(rng, b))[0]
            t0 = time.perf_counter()
            for _ in range(STEPS):
                b = vec.step(random_actions(rng, b))[0]
            dt = time.perf_counter() - t0
            print(f"    workers {w}: {N * STEPS / dt:7.0f} env steps/s "
                  f"({1000 * dt / STEPS:.1f} ms per lockstep)")
            if w:
                vec.close()

        import torch
        from rl.ppo import HP, Trainer
        for w in (0, 8):
            hp = HP(n_envs=64, rollout=4096, bo3=True)
            torch.manual_seed(0)
            tr = Trainer(T, CFG, spec.build(T), hp, seed=0, workers=w,
                         deal_spec=spec)
            t0 = time.perf_counter()
            roll = tr.collect()
            dt = time.perf_counter() - t0
            print(f"    Trainer.collect rollout 4096, workers {w}: {dt:.1f}s "
                  f"({len(roll.steps) / dt:.0f} st/s, {roll.episodes} episodes)")
            tr.close()

    if FAILED:
        print(f"\n\033[31m{len(FAILED)} FAILED\033[0m: {FAILED}")
        sys.exit(1)
    print("\n\033[32mall parallel tests passed\033[0m")


if __name__ == "__main__":
    main()
