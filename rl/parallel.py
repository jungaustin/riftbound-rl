"""The vector env in worker processes -- BACKLOG P2, `docs/PARALLEL.md`.

`ParallelVec` is a drop-in sibling of `vec.VecRiftbound`: same `reset`,
`step`, `shapes`, `continuing`, `episodes`, `truncations`, `truncation_rate`.
The `n_envs` games are split into `workers` contiguous shards, one per spawned
process, and stepped in lockstep: main sends every shard its actions, every
worker steps and encodes its slots, main waits for all of them.

**Observations travel through shared memory, everything else over pipes.** One
env's observation is ~140 KB, so 64 of them pickled through a pipe every step
would eat the gain. Each field of `BatchObs` is a preallocated array in one
`multiprocessing.shared_memory` block; a worker writes its own rows, and main
wraps the same memory as the `BatchObs` it returns, with no copy.

**Consequence: the `BatchObs` returned by `step` is overwritten by the next
`step`.** It is the same object every time. Anything that keeps an observation
across steps must copy it -- `ppo.slice_obs` does, which is what
`Trainer.collect` stores, and `_subset`/`to_torch` are consumed before the next
step. Code that holds `prev = obs` across a step will silently see the new data.

**Dealing happens in the worker**, from a picklable `DealSpec`: `Bo3Dealer`
continues a match from the slot's finished `env.state`, which lives there.
`continuing(i)` is reported back with the done flag. `MatchupDealer` cannot be
moved (its choices are made from main's opponent pool), so `--matchup-pfsp`
with `--workers` is refused.

**Seeds are per slot** (`vec.slot_seed`), in both classes, so the serial and
parallel envs deal identical games; `tests/test_parallel.py [1]` checks every
observation array, reward and done flag against `VecRiftbound`.
"""

from __future__ import annotations

import multiprocessing as mp
import signal
import time
import traceback
import weakref
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from rl.config import Config
from rl.vec import BatchObs, slot_seed


class WorkerError(RuntimeError):
    """A worker process raised (its traceback is the message) or died."""


@dataclass
class DealSpec:
    """Everything a worker needs to build the deal function itself.

    `kind`: "v0" | "v1" (procedural), "pool" (`deck_pool_deal`), "bo3"
    (`Bo3Dealer`). `pool` is the list of `decks.DeckLoad` (plain dataclasses,
    picklable) already narrowed by `--decks` / `--holdout`, so the worker never
    re-selects. `ppo.main` builds its own in-process deal from the same spec,
    so eval and the envs cannot drift apart.
    """
    kind: str
    pool: list | None = None
    min_coverage: float = 0.0

    def build(self, table):
        from rl import ppo   # lazy: the deal code lives with the trainer
        if self.kind == "v0":
            return ppo.v0_deal(table)
        if self.kind == "v1":
            return ppo.v1_deal(table)
        if self.kind == "pool":
            return ppo.deck_pool_deal(table, self.min_coverage,
                                      seed_decks=self.pool)
        if self.kind == "bo3":
            return ppo.Bo3Dealer(table, self.pool, self.min_coverage)
        raise ValueError(f"unknown deal kind {self.kind!r}")


# ---------------------------------------------------------------------------
# Shared-memory layout
# ---------------------------------------------------------------------------

def _layout(shapes: dict, n: int, privileged: bool) -> list[tuple]:
    """[(key, shape, dtype str, offset)], and the total size as the last offset.

    Keys: "z:<zone>", "m:<zone>", "globals", "legal_actions", "action_mask",
    "to_move", "n_legal", "privileged". Dtypes are `obs.Obs`'s; a worker checks
    them on its first write, so a changed encoder fails loudly, not silently.
    """
    fields = []
    for z in shapes["zones"]:
        fields.append((f"z:{z}", (n,) + tuple(shapes[z]), "float32"))
        fields.append((f"m:{z}", (n, shapes[z][0]), "bool"))
    fields += [("globals", (n,) + tuple(shapes["globals"]), "float32"),
               ("legal_actions", (n,) + tuple(shapes["actions"]), "float32"),
               ("action_mask", (n, shapes["actions"][0]), "bool"),
               ("to_move", (n,), "int8"),
               ("n_legal", (n,), "int16")]
    if privileged:
        fields.append(("privileged", (n,) + tuple(shapes["privileged"]),
                       "float32"))
    out, off = [], 0
    for key, shape, dt in fields:
        out.append((key, shape, dt, off))
        nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
        off += (nbytes + 63) // 64 * 64
    return out, max(off, 64)


def _views(buf, layout) -> dict[str, np.ndarray]:
    return {key: np.ndarray(shape, dtype=dt, buffer=buf, offset=off)
            for key, shape, dt, off in layout}


def _write(v: dict, i: int, o, check: bool) -> None:
    """One `Obs` into row i of the shared arrays."""
    if check:
        for z in o.zones:
            assert o.zones[z].dtype == v[f"z:{z}"].dtype, f"zone {z} dtype"
            assert o.zone_mask[z].dtype == v[f"m:{z}"].dtype, f"mask {z} dtype"
        assert o.globals.dtype == v["globals"].dtype
        assert o.legal_actions.dtype == v["legal_actions"].dtype
        assert o.action_mask.dtype == v["action_mask"].dtype
        assert (o.privileged is None) == ("privileged" not in v)
    for z, a in o.zones.items():
        v[f"z:{z}"][i] = a
        v[f"m:{z}"][i] = o.zone_mask[z]
    v["globals"][i] = o.globals
    v["legal_actions"][i] = o.legal_actions
    v["action_mask"][i] = o.action_mask
    v["to_move"][i] = o.to_move
    v["n_legal"][i] = o.n_legal
    if o.privileged is not None:
        v["privileged"][i] = o.privileged


def _batch_view(v: dict, zones) -> BatchObs:
    return BatchObs(
        zones={z: v[f"z:{z}"] for z in zones},
        zone_mask={z: v[f"m:{z}"] for z in zones},
        globals=v["globals"], legal_actions=v["legal_actions"],
        action_mask=v["action_mask"], to_move=v["to_move"],
        n_legal=v["n_legal"], privileged=v.get("privileged"))


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

def _worker(lo: int, hi: int, n_envs: int, table, cfg: Config, spec: DealSpec,
            seed0: int, auto_pass: bool, privileged: bool, check: bool,
            shm_name: str, layout, conn) -> None:
    # Ctrl-C reaches the whole process group. Main owns shutdown; a worker that
    # died on its own KeyboardInterrupt would turn a clean stop into a
    # WorkerError mid-step.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    shm = None
    try:
        from rl.env import RiftboundEnv
        from rl.obs import Encoder
        shm = SharedMemory(name=shm_name)
        v = _views(shm.buf, layout)
        deal_fn = spec.build(table)
        for_slot = getattr(deal_fn, "for_slot", None)
        cont = getattr(deal_fn, "continuing", None)
        enc = Encoder(table, cfg, privileged=privileged)
        envs = [RiftboundEnv(table, cfg, encoder=enc, auto_pass=auto_pass,
                             check=check) for _ in range(hi - lo)]
        dealt = [0] * (hi - lo)
        first = [True]

        def fresh(j: int):
            slot = lo + j
            seed = slot_seed(seed0, slot, n_envs, dealt[j])
            dealt[j] += 1
            env = envs[j]
            if for_slot is not None:
                args, ctx = for_slot(slot, env.state, seed)
                return env.reset(seed, *args, match=ctx)
            return env.reset(seed, *deal_fn(seed))

        def put(j: int, o) -> None:
            _write(v, lo + j, o, first[0])
            first[0] = False

        def conts() -> np.ndarray:
            return np.array([bool(cont(lo + j)) if cont is not None else False
                             for j in range(hi - lo)], bool)

        while True:
            msg = conn.recv()
            if msg[0] == "step":
                t0 = time.perf_counter()
                acts = msg[1]
                s = hi - lo
                rewards = np.zeros((s, 2), np.float32)
                done = np.zeros(s, bool)
                infos: list[dict] = []
                for j, env in enumerate(envs):
                    r = env.step(int(acts[j]))
                    info = {"action": r.info["action"]}
                    if r.done:
                        rewards[j] = r.rewards
                        done[j] = True
                        info["episode"] = env.summary()
                        info["truncated"] = bool(r.truncated)
                        put(j, fresh(j))
                    else:
                        put(j, r.obs)
                    infos.append(info)
                c = conts()
                conn.send(("ok", (rewards, done, infos, c,
                                  time.perf_counter() - t0)))
            elif msg[0] == "reset":
                for j in range(hi - lo):
                    put(j, fresh(j))
                conn.send(("ok", conts()))
            elif msg[0] == "raise":          # test hook: tests/test_parallel [2]
                raise RuntimeError(msg[1])
            elif msg[0] == "close":
                break
    except BaseException:                    # noqa: BLE001 -- report, then die
        try:
            conn.send(("err", traceback.format_exc()))
        except Exception:                    # main already gone
            pass
    finally:
        if shm is not None:
            v = None
            try:
                shm.close()
            except BufferError:
                pass
        conn.close()


# ---------------------------------------------------------------------------
# Main side
# ---------------------------------------------------------------------------

def _shutdown(shm: SharedMemory, procs: list, conns: list) -> None:
    """Stop the workers and unlink the block. Idempotent; runs from `close`,
    from garbage collection and at interpreter exit (`weakref.finalize`)."""
    for c in conns:
        try:
            c.send(("close",))
        except Exception:
            pass
    for p in procs:
        p.join(timeout=2.0)
        if p.is_alive():
            p.terminate()
            p.join(timeout=2.0)
    for c in conns:
        try:
            c.close()
        except Exception:
            pass
    # Unlink FIRST: it only removes the name, and it is the part that leaks.
    # `close` fails with BufferError while a caller still holds a view of the
    # last BatchObs; the mapping then goes away with the process.
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    try:
        shm.close()
    except BufferError:
        pass


class ParallelVec:
    """`VecRiftbound`, with the games in `workers` spawned processes.

    `deal` is a `DealSpec` (it has to cross a process boundary), not a deal
    function. See the module docstring for the shared-memory contract.
    """

    def __init__(self, table, cfg: Config, n_envs: int, deal: DealSpec,
                 seed0: int = 0, workers: int = 4, auto_pass: bool = True,
                 privileged: bool = True, check: bool = False) -> None:
        from rl.obs import Encoder
        if not isinstance(deal, DealSpec):
            raise TypeError("ParallelVec needs a DealSpec: the deal is built "
                            "inside each worker, so it must be picklable")
        workers = max(1, min(int(workers), n_envs))
        self.cfg, self.n_envs, self.workers = cfg, n_envs, workers
        self.enc = Encoder(table, cfg, privileged=privileged)
        self._shapes = self.enc.shapes()
        layout, size = _layout(self._shapes, n_envs, privileged)
        self.shm = SharedMemory(create=True, size=size)
        self._procs: list = []
        self._conns: list = []
        # Registered before any process starts, so a failure part-way through
        # construction still unlinks the block.
        self._final = weakref.finalize(self, _shutdown, self.shm,
                                       self._procs, self._conns)
        self._views = _views(self.shm.buf, layout)
        self._obs = _batch_view(self._views, self._shapes["zones"])

        bounds = np.linspace(0, n_envs, workers + 1).round().astype(int)
        self.shards = [(int(bounds[k]), int(bounds[k + 1]))
                       for k in range(workers)]
        ctx = mp.get_context("spawn")
        for lo, hi in self.shards:
            here, there = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_worker, daemon=True, name=f"riftbound-env-{lo}-{hi}",
                args=(lo, hi, n_envs, table, cfg, deal, seed0, auto_pass,
                      privileged, check, self.shm.name, layout, there))
            p.start()
            # Main must drop its copy of the child's end, or a worker that
            # dies without a word never produces an EOF and main waits forever.
            there.close()
            self._procs.append(p)
            self._conns.append(here)
        self._cont = np.zeros(n_envs, bool)
        # Diagnostics: seconds each worker spent stepping, and main's wall
        # time inside `step`. busy.max() / wall near 1 = workers saturated;
        # busy.mean() / busy.max() well below 1 = lockstep waits on stragglers.
        self.busy = np.zeros(workers)
        self.wall = 0.0
        self.episodes = 0
        self.truncations = 0

    # -- plumbing ----------------------------------------------------------

    def _gather(self) -> list:
        out, errs = [], []
        for k, c in enumerate(self._conns):
            try:
                tag, payload = c.recv()
            except (EOFError, OSError):
                p = self._procs[k]
                p.join(timeout=1.0)
                errs.append(f"worker {k} {self.shards[k]} died without a "
                            f"traceback (exitcode {p.exitcode})")
                continue
            if tag == "err":
                errs.append(f"worker {k} slots {self.shards[k]}:\n{payload}")
            else:
                out.append(payload)
        if errs:
            self.close()
            raise WorkerError("\n".join(errs))
        return out

    def _send(self, msgs) -> None:
        if not self._final.alive:
            raise WorkerError("ParallelVec is closed")
        for c, m in zip(self._conns, msgs):
            try:
                c.send(m)
            except (BrokenPipeError, OSError):
                pass        # the reason arrives (or is synthesized) in _gather

    # -- the VecRiftbound interface ------------------------------------------

    def reset(self) -> BatchObs:
        self._send([("reset",)] * self.workers)
        for (lo, hi), cont in zip(self.shards, self._gather()):
            self._cont[lo:hi] = cont
        return self._obs

    def step(self, action_indices) -> tuple[BatchObs, np.ndarray, np.ndarray,
                                            list[dict]]:
        """`VecRiftbound.step`. The returned `BatchObs` is shared memory and
        is overwritten by the next call -- copy what you keep."""
        t0 = time.perf_counter()
        a = np.asarray(action_indices, np.int64)
        self._send([("step", a[lo:hi]) for lo, hi in self.shards])
        rewards = np.zeros((self.n_envs, 2), np.float32)
        done = np.zeros(self.n_envs, bool)
        infos: list[dict] = []
        for k, ((lo, hi), (r, d, inf, cont, busy)) in enumerate(
                zip(self.shards, self._gather())):
            self.busy[k] += busy
            rewards[lo:hi] = r
            done[lo:hi] = d
            self._cont[lo:hi] = cont
            for info in inf:
                if "truncated" in info:
                    self.episodes += 1
                    self.truncations += int(info.pop("truncated"))
                infos.append(info)
        self.wall += time.perf_counter() - t0
        return self._obs, rewards, done, infos

    def continuing(self, slot: int) -> bool:
        return bool(self._cont[slot])

    def shapes(self) -> dict:
        return self._shapes

    @property
    def truncation_rate(self) -> float:
        return self.truncations / max(1, self.episodes)

    def close(self) -> None:
        self._final()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
