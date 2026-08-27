# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Handlers for the `perf_measure` DAG (dispatch-perfmon-exec.md D12; rev2
P01):

    perf_measure -> probe -> N x measure -> perf_result
                          -> finalize (depends: perf_result)

Mirrors `tune_kernel.py`'s structure and its two rules -- message classes
namespaced `perf_measure/...`; the DAG-START class stays the bare
`perf_measure`, matching the `task_queue.class` value the PG reader forwards
verbatim -- with the differences P01 spells out explicitly:

* No `preprocess` node: inputs are filled on-device by the runner (rev0
  §5.3), so there is no tmpdir and no `.pt` file for this DAG.
* `ProbeHandler` fans out over `(iface, backend)`, discovered by asking the
  runner itself (`exaid.enumerate()`) rather than guessing counts in Python
  -- `enumerate_backends` calls `hipGetDeviceProperties`, so it needs a GPU
  of the target arch, which the dispatching host need not have (P01).
* The `len(variants) <= 1` skip that `tune_kernel.ProbeHandler` applies at
  op level must NOT apply here: a single-backend interface still needs its
  *number* to measure it with.
* `impl_index` here is not a position in a list (as it is for `tune_kernel`)
  -- it is the actual backend id `enumerate()` reported, because that is
  exactly the value `measure`'s wire protocol requires as `<backend>`
  (perfmon/core/main.cc's `measure_cmd`).
* Both `ProbeHandler` and `MeasureHandler` independently call `desc.
  resolve_entry(entry, exaid.vram_total_gb)` -- deliberately not computed
  once and threaded through the fanned-out messages, because `perf_measure/
  measure` messages are pulled off the shared `gpu_queue` and may land on a
  *different* GPU worker (different VRAM) than the one that probed the
  entry (D05's `resolve_entry` is deterministic in (entry, vram_total_gb),
  not in entry alone).

`PerfDescription.resolve_entry` (added to the ABC in `pq/perfmon/pdesc.py`
by this same task, D05 having only committed flash's implementation) is
what keeps this module family-neutral: it calls `desc.resolve_entry(...)`,
never `modules.flash.perfmon.resolve` directly.
"""

import dataclasses
import logging

from ...exaid import exaid_create, ExaidSubprocessNotOK
from ...registry import load_family_perfmon
from ...pq.queue import TaskQueue
from ...pq.perf import save_perf_result
from .base import MessageHandler

logger = logging.getLogger(__name__)


def _entry_pon(desc, resolved, iface_name: str) -> str:
    """The full wire `entry_pon` text for one (resolved metadata, iface)
    pair: `functional_pon` (the feature half) and `shape_pon` (the shape
    half), joined with the same `;` separator each half already uses
    internally (`aotriton.utils.pon.render_pon`'s default `sep`)."""
    return desc.functional_pon(resolved, iface_name) + ';' + desc.shape_pon(resolved)


class PerfMeasureHandler(MessageHandler):
    """
    Starts the DAG by creating the initial probe message.

    Input: perf_measure message from PG reader
    Output: probe message
    """

    @classmethod
    def get_class_name(cls) -> str:
        return "perf_measure"

    def handle(self, message: dict) -> dict:
        return {
            'class': 'perf_measure/probe',
            'target_queue': 'gpu_queue',
            'task_id': message['task_id'],
            'task_config': message['task_config'],
        }


class ProbeHandler(MessageHandler):
    """
    Discovers, per interface, which backends the runner can serve this
    entry with (`exaid.enumerate()`), and fans out one `measure` message per
    `(iface, backend)` pair plus one `finalize` message that waits on all of
    them.

    Input: probe message
    Output: N x measure messages + one finalize message (with dependencies),
        or a mark_task_failed message.
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "perf_measure/probe"

    def handle(self, message: dict) -> list[dict] | dict:
        task_config = message['task_config']
        task_id = message['task_id']
        module = task_config['module']
        preset = task_config['preset']
        arch = task_config.get('arch')

        exaid = exaid_create('perf_measure', module, self.gpu_id)
        # load_family_perfmon() returns the module (see registry.py); the
        # instance is its PerfDesc attribute -- same pattern dispatch_tasks.
        # py's resolve_module() uses for the CLI's own --workload=perfmon path.
        desc = load_family_perfmon(module).PerfDesc()

        try:
            exaid.use_profile(preset)
            entry = desc.ENTRY_CLASS.from_dict(task_config['entry'])
            resolved = desc.resolve_entry(entry, exaid.vram_total_gb)

            measure_tasks = []
            for iface_name in desc.list_ifaces():
                iface_idx = desc.list_ifaces().index(iface_name)
                entry_pon = _entry_pon(desc, resolved, iface_name)
                enumerated = exaid.enumerate(entry_pon, iface_idx)
                for backend in enumerated['backends']:
                    measure_tasks.append((iface_name, backend))
        except (OSError, ExaidSubprocessNotOK, RuntimeError) as e:
            logger.error(f"Probe failed for task_id={task_id}: {e}")
            return {
                'class': 'mark_task_failed',
                'target_queue': 'cpu_queue',
                'task_id': task_id,
                'arch': arch,
                'error': f"Probe failed: {type(e).__name__}: {e}",
            }

        return self._build_fanout(measure_tasks, task_id, task_config)

    def _build_fanout(self, measure_tasks: list[tuple[str, int]], task_id: int,
                       task_config: dict) -> list[dict]:
        results = []
        for iface_name, backend in measure_tasks:
            results.append({
                'class': 'perf_measure/measure',
                'target_queue': 'gpu_queue',
                'task_id': task_id,
                'task_config': task_config,
                'iface_name': iface_name,
                'impl_index': backend,
            })

        # expected_impls/received_impls bookkeeping mirrors tune_kernel's
        # ProbeHandler exactly: keyed by iface_name, values are the list of
        # backend ids expected for it. An entry with zero measured backends
        # anywhere yields expected_impls={} -- `depends` is then an empty
        # list, which `LocalBroker.forward()` treats as "not blocked" (an
        # empty list is falsy), so finalize is sent immediately rather than
        # waiting on results that will never arrive.
        expected_impls: dict[str, list[int]] = {}
        for name, backend in measure_tasks:
            expected_impls.setdefault(name, []).append(backend)

        results.append({
            'class': 'perf_measure/finalize',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'task_config': task_config,
            'depends': ['perf_measure/perf_result'],
            'expected_impls': expected_impls,
            'received_impls': {},
        })
        logger.info(f"Probed {len(measure_tasks)} (iface, backend) pair(s) "
                   f"for task_id={task_id}")
        return results


class MeasureHandler(MessageHandler):
    """
    Measures one (iface, backend) pair. The only node in this DAG that
    drives an actual measurement on the device.

    Input: measure message
    Output: perf_result message
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "perf_measure/measure"

    def handle(self, message: dict) -> dict:
        task_config = message['task_config']
        task_id = message['task_id']
        iface_name = message['iface_name']
        impl_index = message['impl_index']  # a backend id, not a position

        module = task_config['module']
        preset = task_config['preset']
        arch = task_config['arch']

        exaid = exaid_create('perf_measure', module, self.gpu_id)
        desc = load_family_perfmon(module).PerfDesc()

        report = {'arch': arch, 'class': 'perf_measure', 'subclass': '',
                  'iface_name': iface_name, 'impl_index': impl_index}
        try:
            # Independently resolved here (not threaded through from
            # ProbeHandler's own resolution) -- this message may have been
            # pulled by a different GPU worker than the one that probed it;
            # see the module docstring.
            exaid.use_profile(preset)
            entry = desc.ENTRY_CLASS.from_dict(task_config['entry'])
            resolved = desc.resolve_entry(entry, exaid.vram_total_gb)
            iface_idx = desc.list_ifaces().index(iface_name)
            entry_pon = _entry_pon(desc, resolved, iface_name)

            result_data = exaid.measure(entry_pon, iface_idx, impl_index)
            # D11's pq/perf.py gates an OK report on this key: the resolved
            # working-set size (BATCH, N_HEADS) is otherwise unrecoverable.
            result_data['resolved_metadata'] = dataclasses.asdict(resolved)
            report['result'] = 'OK'
            report['result_data'] = result_data
            report['error'] = None
        except OSError as e:
            logger.error(f"Measure crashed for {iface_name}[{impl_index}]: {e}")
            report['result'] = 'crash'
            report['result_data'] = None
            report['error'] = {'errno': e.errno, 'stderr': e.strerror}
        except ExaidSubprocessNotOK as e:
            logger.error(f"Measure NotOK for {iface_name}[{impl_index}]: {e}")
            report['result'] = 'NotOK'
            report['result_data'] = None
            report['error'] = {'stdout': e.stdout, 'stderr': e.stderr}
        except RuntimeError as e:
            # Includes ExaidProfileMismatch (use_profile) and resolve_entry's
            # "does not fit in VRAM even at the floor" -- neither is a
            # process crash, but neither has a resolved_metadata to report.
            logger.error(f"Measure failed for {iface_name}[{impl_index}]: {e}")
            report['result'] = 'NotOK'
            report['result_data'] = None
            report['error'] = {'message': str(e)}

        return {
            'class': 'perf_measure/perf_result',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'iface_name': iface_name,
            'impl_index': impl_index,
            'report': report,
        }


class WritePerfResultHandler(MessageHandler):
    """
    Writes one measurement to task_reports via pq/perf.py.

    Input: perf_result message
    Output: None (triggers dependency resolution for finalize)
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "perf_measure/perf_result"

    def handle(self, message: dict) -> None:
        task_id = message['task_id']
        report = message['report']

        save_perf_result(task_id, report['arch'], report, self.db_conn)

        logger.debug(f"Wrote perf result for task_id={task_id} "
                    f"{report['iface_name']}[{report['impl_index']}]")
        return None


class FinalizeHandler(MessageHandler):
    """
    Aggregates all measurement results and marks the task completed.

    Input: finalize message (after dependencies resolved)
    Output: dag_ack message (triggers PG reader to continue)

    Same dual-context design as `tune_kernel.PostprocessHandler`:
    `resolve_dependency()` runs in the BROKER context (`db_conn=None`);
    `handle()` runs in the CPU-worker context (a real `db_conn`). No tmpdir
    to clean up here -- P01: "no preprocess node".
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "perf_measure/finalize"

    def resolve_dependency(self, blocked_msg: dict, incoming_msg: dict) -> bool:
        """Called in the BROKER context. Do NOT access self.db_conn here --
        it will be None."""
        if blocked_msg['class'] != 'perf_measure/finalize':
            return False

        if incoming_msg['class'] not in blocked_msg['depends']:
            return False

        if blocked_msg['task_id'] != incoming_msg['task_id']:
            return False

        impl_name = incoming_msg['iface_name']
        impl_index = incoming_msg['impl_index']
        blocked_msg['received_impls'].setdefault(impl_name, {})[impl_index] = incoming_msg['report']

        expected = blocked_msg['expected_impls']
        received = blocked_msg['received_impls']
        for name, indices in expected.items():
            if name not in received:
                return False
            for idx in indices:
                if idx not in received[name]:
                    return False

        logger.info(f"All measurements received for task_id={blocked_msg['task_id']}, "
                   f"unblocking finalize")
        return True

    def handle(self, message: dict) -> dict:
        task_id = message['task_id']
        task_config = message['task_config']
        arch = task_config.get('arch')

        logger.info(f"FinalizeHandler: Marking task_id={task_id} as completed (arch={arch})")
        TaskQueue(self.db_conn).mark_completed(task_id, arch)

        logger.info(f"Finalize completed for task_id={task_id}")

        return {
            'class': 'dag_ack',
            'task_id': task_id,
        }

    def teardown_with_unmet_dependency(self, message: dict) -> dict:
        """Called during graceful shutdown when a finalize message has
        unmet dependencies (GPU workers stopped before all measurements
        completed) -- move the task back to pending. No tmpdir to clean up
        (unlike tune_kernel's PostprocessHandler)."""
        task_id = message['task_id']
        task_config = message.get('task_config', {})
        arch = task_config.get('arch')

        logger.info(f"FinalizeHandler teardown: task_id={task_id} has unmet dependencies, "
                   f"creating cancel message")

        return {
            'class': 'graceful_cancel_running_task',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'arch': arch
        }
