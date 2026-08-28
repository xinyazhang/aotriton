#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Crash-isolated CLI runner handles ("exaid").

One `ExaidProxy` owns a pipe to a subprocess and speaks a line protocol to
it; one `ExaidWorker` subclass per DAG class turns that into typed commands.
The proxy is DAG-neutral -- it never knows what it is talking to, because the
argv comes from the worker via `_spawn_argv()` (perfmon rev2 R04).

    ExaidProxy              pipe, line protocol, OVERHEATING tolerance,
                            crash relaunch, exit/EOF shutdown.

    ExaidWorker (base)      module_name, gpu_id, proxy ownership, exit(),
                            registration in _cache. Declares the abstract
                            _spawn_argv() and the process-lifetime policy.
      |
      +-- ExaidTunerWorker    prepare_data, probe, benchmark
      +-- ExaidPerfmonWorker  enumerate, measure, platform

Both subclasses speak the SAME wire DSL, which is what lets one proxy serve
both: the C++ perfmon runner implements it too (perfmon/core/protocol.cc).

--- Two caches, at two levels -------------------------------------------

These are independent, and conflating them is a real hazard:

  1. worker object -- `exaid_create` / `ExaidWorker._cache`, keyed
     `(klass, module_name, gpu_id)`, living for the worker process.
  2. OS process -- held INSIDE a worker. ExaidTunerWorker keeps one
     persistent process. ExaidPerfmonWorker keeps at most one, tagged with
     the profile it was launched for, and replaces it when the profile
     changes.

`preset` must NOT appear in the level-1 key. Keying worker objects by preset
would give one worker -- and so one live process -- per preset, which is
exactly the N-process problem the profile tag exists to avoid.

--- Why perfmon replaces rather than pools, or one-shots ------------------

Measured on gfx942: the runner starts in 390 ms, of which HIP context
creation is 285 ms (83%) and is irreducible -- unchanged by the visible
device mask, HIP_MODULE_LAZY_LOADING or GPU_MAX_HW_QUEUES. A strictly
one-shot process would pay that per measurement: 535 ms for 145 ms of work,
+269%. A tag of exactly one, rather than an LRU, bounds resident state to a
single runner and makes the wrong-preset hazard one explicit comparison
instead of an invariant spread across a dict.
"""

import sys
import os
from pathlib import Path
from .utils import safe_readline
import subprocess
import errno
import json
import logging

logger = logging.getLogger(__name__)

CURRENT_FILE_PATH = Path(__file__).resolve()
AOTRITON_ROOT = CURRENT_FILE_PATH.parent.parent.parent.absolute()


def perfmon_launcher() -> Path:
    """R05's shim: it resolves the preset, sets HIP_VISIBLE_DEVICES/ROCM_PATH
    and exec()s the C++ runner. Spawning the shim rather than the binary keeps
    GPU selection and environment setup out of both the runner and this file.

    `AOTRITON_PERFMON_ROOT` is REQUIRED -- no fallback, by design. perfmon/ is
    source beside the package and is never installed with it, so nothing this
    file can compute from its own location is an answer: the previous fallback,
    `Path(__file__).parent.parent.parent / 'perfmon'`, silently produced
    <site-packages>/perfmon on any non-editable install and turned a missing
    configuration into a confusing ENOENT at spawn time. Whoever starts the
    worker knows where the checkout is; .tune/remote/worker_service.sh exports
    it.

    Read per call, not at import, so a variable set after this module is
    imported still takes effect.
    """
    root = os.environ.get('AOTRITON_PERFMON_ROOT')
    if not root:
        raise RuntimeError(
            'AOTRITON_PERFMON_ROOT is not set. It must point at the perfmon/ '
            'directory of an aotriton checkout (the launcher shim lives there '
            'and is never installed with the package). '
            '.tune/remote/worker_service.sh exports it for workers it starts.')
    return Path(root) / 'launch_runner.sh'


def first(line, sep=" "):
    seps = line.split(sep, maxsplit=1)
    if len(seps) > 1:
        return seps
    return seps[0], None


class ExaidSubprocessNotOK(RuntimeError):
    def __init__(self, stdout: str|None, stderr: str|None):
        self.stdout = stdout
        self.stderr = stderr


class ExaidProfileMismatch(RuntimeError):
    """A runner did not identify as the profile it was launched for.

    Backs the profile comparison with the runner's own self-identification,
    which catches what the comparison alone cannot: a mis-provisioned worker,
    or a stale binary at the resolved path.
    """


class ExaidProxy(object):
    """Pipe and line protocol. Knows nothing about what it launched.

    `spawn_argv` is a callable returning the argv to exec. It is called on
    every (re)launch rather than once, so a worker that changes what it wants
    to run -- ExaidPerfmonWorker swapping presets -- needs no cooperation
    from this class beyond `shutdown()`.
    """

    def __init__(self, spawn_argv, base_dir: str | None = None):
        self._spawn_argv = spawn_argv
        self._base_dir = base_dir
        self._process = None
        self._last_error = None

    def get_base_dir(self):
        return self._base_dir if self._base_dir is not None else AOTRITON_ROOT.as_posix()

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _reap(self):
        """Forget a process that has already exited, so the next use relaunches."""
        if self._process is None:
            return
        pid = self._process.pid
        try:
            self._process.wait(timeout=0)
        except Exception:
            pass
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            try:
                stream.close()
            except Exception:
                pass
        self._process = None
        logger.warning(f"Exaid worker process {pid} is gone; will relaunch on next use")

    @property
    def process(self):
        # Detect a process that died since we last spoke to it. Without this,
        # only readinfo() noticed a crash (it clears _process on failure) while
        # write() did not -- so after a runner was killed, every subsequent
        # write hit EPIPE against the dead handle forever and the proxy never
        # relaunched. Crash relaunch is this class's job, so the check belongs
        # here, on the one path both write() and readinfo() go through.
        if self._process is not None and self._process.poll() is not None:
            self._reap()
        if self._process is None:
            args = self._spawn_argv()
            logger.info(f"Starting exaid worker process: {' '.join(str(a) for a in args)}")
            self._process = subprocess.Popen(args,
                                             stdin=subprocess.PIPE,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             cwd=self.get_base_dir())
            logger.info(f"Exaid worker process started: pid={self._process.pid}")
        return self._process

    def write(self, *objects, sep=' '):
        cmd = sep.join(str(o) for o in objects)
        proc = self.process
        logger.info(f"Sending command to worker (pid={proc.pid}): {cmd}")
        line = cmd + '\n'
        try:
            proc.stdin.write(line.encode('utf-8'))
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            # It died between the liveness check above and this write. Clear
            # the handle so the caller's next attempt starts a fresh process
            # rather than retrying against a corpse.
            self._reap()
            raise OSError(errno.EPIPE,
                          f"exaid worker died while sending {cmd!r}: {e}") from e

    def readinfo(self, *, timeout: int | float = 10):
        logger.info(f"Waiting for response from worker (pid={self.process.pid}, timeout={timeout}s)")
        while True:
            (line, eno, error_msg) = safe_readline(self.process, timeout=timeout)
            if eno != 0 or line is None:
                if eno == errno.ETIMEDOUT:
                    logger.error(f"Worker timeout after {timeout}s, killing process (pid={self._process.pid})")
                    self._process.kill()
                elif line is None:
                    logger.error(f"Worker closed stdout unexpectedly (pid={self._process.pid})")
                else:
                    logger.error(f"Worker error (pid={self._process.pid}, errno={eno}): {error_msg}")
                self._process.wait()
                stdout = self._process.stdout.read().decode('utf-8', errors='replace')
                stderr = self._process.stderr.read().decode('utf-8', errors='replace')
                logger.error(f"Worker stdout: {stdout}")
                logger.error(f"Worker stderr: {stderr}")
                del self._process
                self._process = None
                error_desc = error_msg if error_msg else "stdout closed unexpectedly"
                raise OSError(eno if eno != 0 else errno.EPIPE, error_desc + "\nSTDOUT:\n" + stdout + "\nSTDERR:\n" + stderr)
            ret, info = first(line)
            if ret == "OVERHEATING:":
                logger.warning(f"Worker overheating warning: {line}")
                continue
            if ret != "OK":
                logger.error(f"Worker returned non-OK status: {line}")
                raise ExaidSubprocessNotOK(line, error_msg)
            logger.info(f"Received response from worker (pid={self.process.pid}): {ret} {info}")
            break
        return info

    def join(self):
        if self._process is None:
            return
        pid = self._process.pid
        try:
            self._process.wait(0.2)
            logger.info(f"Worker process exited cleanly (pid={pid})")
        except subprocess.TimeoutExpired:
            logger.warning(f"Worker process did not exit in 0.2s, killing (pid={pid})")
            self._process.kill()
            self._process.wait()
            logger.info(f"Worker process killed (pid={pid})")
            del self._process
            self._process = None

    def shutdown(self):
        """Terminate the current process, if any, and forget it.

        The next `process` access relaunches, calling `spawn_argv` afresh --
        which is how a worker swaps what it is running underneath itself.

        Both termination paths are verified against the C++ runner: the
        `exit` command, and a bare EOF on stdin (protocol.cc's
        `while (std::getline(in, line))` ends naturally), each rc=0. `exit`
        is tried first; EOF is the fallback for a process already too wedged
        to parse a command.
        """
        if self._process is None:
            return
        pid = self._process.pid
        logger.info(f"Shutting down exaid worker process (pid={pid})")
        try:
            self.write('exit')
        except (OSError, ValueError, BrokenPipeError):
            try:
                self._process.stdin.close()   # EOF
            except Exception:
                pass
        self.join()
        if self._process is not None:
            self._process.kill()
            self._process.wait()
        self._process = None


class ExaidWorker(object):
    """Base: a handle on an exaid-isolated runner. Abstract.

    Kept named `ExaidWorker` rather than renamed, because it is the concept
    and every call site meaning "an exaid-isolated runner handle" still means
    that. Subclasses are named for the subsystem they belong to.
    """

    TMPFS_LOCATION = Path('/dev/shm/aotriton-tuner')
    _cache = {}

    #: task_queue.class this worker serves. Set by each subclass.
    KLASS: str | None = None

    def __init__(self, module_name: str, gpu_id: int):
        self._module_name = module_name
        self._module = None
        self._gpu_id = gpu_id
        self._proxy = None

    def _spawn_argv(self) -> list:
        """argv for the runner process. The proxy calls this on every launch."""
        raise NotImplementedError

    @property
    def tmpfs(self) -> Path:
        return self.TMPFS_LOCATION

    @property
    def proxy(self):
        if self._proxy is None:
            self._proxy = ExaidProxy(self._spawn_argv)
        return self._proxy

    def exit(self):
        self.proxy.write("exit")
        self.proxy.join()


class ExaidTunerWorker(ExaidWorker):
    """The `tune_kernel` DAG's runner: `aotriton.tune.testrun`.

    Process lifetime: ONE persistent process, reused across tasks. Startup is
    2.131 s (torch plus a CUDA context), so respawning per task is not on the
    table.

    Callers here (localq's tune_kernel handlers) DO carry a tuning_level --
    but it lives in `task_config['tuning_level']`, not on this class, and is
    threaded into `probe()` as a per-call filter argument, never stored as
    worker state.
    """

    KLASS = 'tune_kernel'

    def _spawn_argv(self):
        # Byte-identical to the pre-R04 hardcoded argv.
        return ['python', '-m', 'aotriton.tune.testrun',
                self._module_name, '--gpu', str(self._gpu_id)]

    @property
    def module(self):
        if self._module is None:
            from .registry import load_tune_module
            self._module = load_tune_module(self._module_name)
        return self._module

    def entry_from_dict(self, entry_dict: dict):
        tune = self.module.TuneDesc()
        return tune.ENTRY_CLASS.from_dict(entry_dict)

    def get_tmpfs_for(self, entry_dict):
        return self.TMPFS_LOCATION / self.entry_from_dict(entry_dict).as_posix()

    def prepare_data(self, entry_dict: dict, workdir: Path, extra_im_texts: list[str] = []):
        logger.info(f"prepare_data: entry={entry_dict}, workdir={workdir}, extra_ims={len(extra_im_texts)}")
        workdir.mkdir(parents=True, exist_ok=True)
        entry = self.entry_from_dict(entry_dict)
        self.proxy.write('prepare_data', entry.as_text(), workdir.as_posix(), *extra_im_texts)
        result = self.proxy.readinfo(timeout=600)
        logger.info(f"prepare_data completed: {result}")
        return result

    def probe(self, workdir: Path, arch: str | None = None, tuning_level: str | None = None):
        """
        Args:
            tuning_level: optional per-call filter (e.g. 'kernel' or 'op'),
                NOT worker state -- passed straight through to `testrun`'s
                `probe` command as a third wire token. Callers driving a task
                (e.g. localq's ProbeHandler) pass `task_config['tuning_level']`
                here so a worker only probes what its container's library can
                serve; filtering happens on this call, before `testrun`
                attempts any import, not by post-hoc filtering this method's
                return value.
        """
        logger.info(f"probe: workdir={workdir} arch={arch} tuning_level={tuning_level}")
        self.proxy.write('probe', workdir.as_posix(), arch or '', tuning_level or '')
        result = json.loads(self.proxy.readinfo())
        logger.info(f"probe completed: found {len(result)} kernels")
        return result

    def benchmark(self, workdir: Path, impl_selector):
        """
        Args:
            impl_selector: an `aotriton.tune.tdesc.ImplSelector` instance
                identifying the impl variant to benchmark. Its `as_text()`
                form (e.g. 'attn_fwd=3' or 'op.attn_fwd=1') is the DSL the
                `testrun.py` worker process's `benchmark` command parses back
                into an `ImplSelector` via `ImplSelector.parse_text()`.
        """
        logger.info(f"benchmark: workdir={workdir}, impl_selector={impl_selector.as_text()}")
        self.proxy.write('benchmark', workdir.as_posix(), impl_selector.as_text())
        result = json.loads(self.proxy.readinfo(timeout=30))
        logger.info(f"benchmark completed: {impl_selector.as_text()} "
                   f"result={result.get('result', 'unknown')}")
        return result


class ExaidPerfmonWorker(ExaidWorker):
    """The `perf_measure` DAG's runner: a subject's own `bin/runner`, via
    R05's launcher shim.

    Process lifetime: AT MOST ONE process, tagged with the profile it was
    launched for. `use_profile()` reuses it when the profile matches and
    replaces it when it does not -- see the module docstring for why
    replacement beats both pooling and one-shotting.

    A profile is (preset, module_name): which AOTriton build to measure, and
    which family's runner to measure it with. The gpu_id is not part of it --
    that is fixed for the life of the worker and is in the level-1 cache key.
    """

    KLASS = 'perf_measure'

    def __init__(self, module_name: str, gpu_id: int):
        super().__init__(module_name, gpu_id)
        self._profile = None          # the preset the live process serves
        self._pending_profile = None  # what the next launch should serve
        self._vram_total_gb = None    # cached from platform(), see _assert_identity

    def _spawn_argv(self):
        if self._pending_profile is None:
            raise RuntimeError('ExaidPerfmonWorker: use_profile() must be called '
                               'before the runner can be launched -- the shim '
                               'needs a preset to resolve a subject.')
        return [perfmon_launcher().as_posix(),
                '--preset', str(self._pending_profile),
                '--module', self._module_name,
                '--gpu', str(self._gpu_id)]

    def use_profile(self, preset: str):
        """Make the live process serve `preset`, replacing it if it does not.

        Returns True if a process was replaced (or first launched), False if
        the existing one was reused. Idempotent.
        """
        if self._profile == preset and self.proxy.alive:
            return False
        if self.proxy.alive:
            logger.info(f"perfmon profile change {self._profile!r} -> {preset!r}: "
                        "replacing runner process")
            self.proxy.shutdown()
        self._pending_profile = preset
        # Force the launch here rather than lazily, so the identity assertion
        # below runs exactly once per process rather than once per command.
        self.proxy.process
        self._profile = preset
        self._assert_identity(preset)
        return True

    def _assert_identity(self, preset: str):
        """Back the profile comparison with the runner's own self-report.

        `platform` costs ~0 ms and runs once per process, not per
        measurement. It catches a mis-provisioned worker or a stale binary at
        the resolved path -- failures the profile comparison cannot see,
        because both sides of that comparison are this process's own belief.
        """
        info = self.platform()
        subject_id = info.get('subject_id')

        # An ABSENT id is a different failure from a WRONG one, and saying so
        # is the whole value of this check. `subject_id=''` used to be reported
        # as "not the subject that was asked for", which sent the reader
        # looking for a mis-provisioned subject when the runner had simply
        # never been told who it is: main.cc takes the id from argv[1] and
        # defaults it to empty, so a launcher that does not pass one produces
        # this for every subject alike.
        if not subject_id:
            self.proxy.shutdown()
            self._profile = None
            raise ExaidProfileMismatch(
                f"runner launched for preset {preset!r} reported no subject_id. "
                f"launch_runner.sh passes it as argv[1] from the subject's "
                f"subject_id file; an empty one means this node is running a "
                f"launcher that predates that, so aotriton.src needs syncing.")

        # Exact equality. This was `preset not in subject_id`, a substring test
        # from when the two were spelled differently -- which also silently
        # accepted any id that merely CONTAINED the preset, so a subject named
        # after a superstring of it would have passed. They are one string now
        # (build_subject.sh writes the preset as both the directory name and
        # the id), so there is nothing left to be loose about.
        if subject_id != preset:
            self.proxy.shutdown()
            self._profile = None
            raise ExaidProfileMismatch(
                f"runner launched for preset {preset!r} identifies as "
                f"subject_id={subject_id!r}; the resolved binary is not the "
                f"subject that was asked for")
        logger.info(f"perfmon runner identity confirmed: subject_id={subject_id}")
        # D05a: the runner is the only process guaranteed to have the GPU
        # (masked to exactly one device via HIP_VISIBLE_DEVICES), so it is
        # the source of VRAM for D05's resolve_entry(). platform() already
        # runs once per process here; cache its answer rather than asking
        # again per measurement.
        self._vram_total_gb = info.get('vram_total_gb')

    @property
    def vram_total_gb(self) -> float | None:
        """This worker's GPU's total VRAM in GiB, from the runner's own
        `platform` self-report (D05a). None until a profile has been
        established (`use_profile()` -> `_assert_identity()`)."""
        return self._vram_total_gb

    def platform(self) -> dict:
        self.proxy.write('platform')
        return json.loads(self.proxy.readinfo())

    def enumerate(self, entry_text: str, iface: int) -> dict:
        """Wire shape `enumerate <entry_pon> <iface>` (perfmon/core/main.cc's
        `enumerate_cmd`, entry_codec.h item 1): `iface` is a separate,
        already-resolved integer index (`PerfDesc.list_ifaces()` order), not
        read from `entry_text`'s own `iface=` key, which is Python-side
        bookkeeping only. `dispatch-perfmon-exec.md` D12 is this method's
        first caller; prior to that this method sent only `entry_text`,
        silently missing the `iface` token the runner requires -- fixed here
        rather than worked around in the handler.
        """
        self.proxy.write('enumerate', entry_text, iface)
        return json.loads(self.proxy.readinfo())

    def measure(self, entry_text: str, iface: int, backend: int) -> dict:
        self.proxy.write('measure', entry_text, iface, backend)
        return json.loads(self.proxy.readinfo(timeout=600))


#: task_queue.class -> the worker that serves it.
_WORKER_CLASSES: dict[str, type] = {
    ExaidTunerWorker.KLASS: ExaidTunerWorker,
    ExaidPerfmonWorker.KLASS: ExaidPerfmonWorker,
}


def exaid_create(klass: str, module_name: str, gpu_id: int):
    """Get (or make) the worker handle for one (class, module, gpu).

    `klass` is the task_queue.class the caller is serving. It is part of the
    key because the two DAGs run different binaries with different process
    lifetimes; a `preset` is deliberately NOT, since that would give one live
    process per preset.
    """
    try:
        worker_cls = _WORKER_CLASSES[klass]
    except KeyError:
        raise ValueError(f"exaid_create: unknown class {klass!r}; "
                         f"expected one of {sorted(_WORKER_CLASSES)}") from None
    key = (klass, module_name, gpu_id)
    if key not in ExaidWorker._cache:
        ExaidWorker._cache[key] = worker_cls(module_name, gpu_id)
    return ExaidWorker._cache[key]


def exaid_exitall():
    for _, exaid in ExaidWorker._cache.items():
        exaid.exit()
    ExaidWorker._cache = {}
