# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
`perf_measure` result storage (dispatch-perfmon-exec.md D11).

The perfmon analogue of `results.py`'s `save_task_report`/`get_task_results`,
scoped to the one (class, subclass) pair `perf_measure` ever uses
(`'perf_measure'`, `''`) so callers never have to spell those two literals
themselves. Every function here takes a live `conn`, never connection
params -- same contract as the rest of `pq/` (see `pq/README.md`).

`result_data` shape (dispatch-perfmon.md §2.3, §12 item 1; rev2 P02):
this module does not compute any of it -- that is the GPU-worker handler's
job (dispatch-perfmon-exec.md D12) -- it only enforces the one field whose
absence makes a row unusable rather than merely incomplete:

  * `resolved_metadata` -- the worker-resolved `FlashInputMetadata` (D05's
    `resolve_entry()`), as a plain dict (`dataclasses.asdict()`). `BATCH`
    and `N_HEADS` are chosen on the worker from its own VRAM (dispatch-
    perfmon.md §2.2), so a row missing this is not a smaller measurement --
    it is one nobody can interpret, because the working-set size it ran at
    is unknown and unrecoverable after the fact. `save_perf_result` raises
    rather than storing such a report.

`result_data` may also carry `timing_method`, `variant_id`, `platform_id`,
and the tri-state `thermal`/`throttled` (rev0 §5.2) -- all supplied by the
caller (the runner's own `measure`/`platform` self-reports, D05a), passed
through unvalidated here: only `resolved_metadata` is load-bearing enough to
gate the write.
"""

from __future__ import annotations

from collections.abc import Iterator

from psycopg.rows import dict_row

from .results import save_task_report

#: The only (class, subclass) pair perf_measure ever writes or reads
#: (schema.sql's CHECK: `class = 'perf_measure' AND subclass = ''`).
_KLASS = 'perf_measure'
_SUBKLASS = ''


class MissingResolvedMetadataError(ValueError):
    """Raised by `save_perf_result` when `report['result_data']` does not
    carry `resolved_metadata`. Never caught and silently downgraded to a
    lesser row -- an unresolved measurement is not partial data, it is
    uninterpretable data (module docstring)."""


def save_perf_result(task_id: int, arch: str, report: dict, conn) -> None:
    """Write one `perf_measure` measurement to `task_reports`.

    Args:
        task_id: Task ID from task_queue.
        arch: GPU architecture the measurement ran on, e.g. 'gfx942'.
        report: dict with keys:
            - iface_name: Interface name (e.g. 'attn_fwd')
            - impl_index: Backend index
            - result: Result status (OK/NotOK/crash/ERROR)
            - result_data: dict, REQUIRED to contain 'resolved_metadata'
              (see module docstring). May be absent/None only when `result`
              is not 'OK' -- a failed/crashed measurement has no resolved
              metadata to report and is not held to this rule.
            - error: Optional error information (JSONB)
            - complete_on_gpu: Optional GPU ID used for the measurement
        conn: PostgreSQL connection (from psycopg.connect)

    Raises:
        MissingResolvedMetadataError: `result` is 'OK' but `result_data` is
            missing or does not carry `resolved_metadata`.
        psycopg.Error: Database errors
    """
    result = report['result']
    result_data = report.get('result_data')
    if result == 'OK' and (not isinstance(result_data, dict)
                            or 'resolved_metadata' not in result_data):
        raise MissingResolvedMetadataError(
            f"save_perf_result: task_id={task_id} arch={arch!r} "
            f"iface_name={report.get('iface_name')!r} "
            f"impl_index={report.get('impl_index')!r}: result_data is "
            "missing the resolved FlashInputMetadata (D05's resolve_entry) "
            "-- refusing to store an OK measurement whose working-set size "
            "(BATCH, N_HEADS) cannot be recovered later.")

    save_task_report(task_id, {
        'arch': arch,
        'class': _KLASS,
        'subclass': _SUBKLASS,
        'iface_name': report['iface_name'],
        'impl_index': report['impl_index'],
        'result': result,
        'result_data': result_data,
        'error': report.get('error'),
        'complete_on_gpu': report.get('complete_on_gpu'),
    }, conn)


def get_perf_results(conn, *, arch: str | None = None,
                      preset: str | None = None,
                      iface_name: str | None = None) -> list[dict]:
    """`perf_measure` rows, optionally filtered by arch/preset/iface_name.

    `preset` is matched against `task_queue.task_config->>'preset'` (D10),
    which is why this is a join rather than a plain `task_reports` scan --
    the preset a measurement ran under is dispatch-time information, not
    something `task_reports` denormalizes. Matched exactly, never split
    into halves (dispatch-perfmon.md §3.2: do not route on a preset's
    rocm/tag halves outside `perfmon/presets.py`).
    """
    query = """
        SELECT
            r.id,
            r.task_id,
            r.arch,
            r.iface_name,
            r.impl_index,
            r.result,
            r.result_data,
            r.error,
            r.gpu_id,
            r.created_at,
            q.task_config ->> 'preset' AS preset
        FROM task_reports r
        JOIN task_queue q
          ON q.id = r.task_id AND q.arch = r.arch AND q.class = r.class
        WHERE r.class = %s AND r.subclass = %s
    """
    params: list = [_KLASS, _SUBKLASS]
    if arch is not None:
        query += " AND r.arch = %s"
        params.append(arch)
    if preset is not None:
        query += " AND q.task_config ->> 'preset' = %s"
        params.append(preset)
    if iface_name is not None:
        query += " AND r.iface_name = %s"
        params.append(iface_name)
    query += " ORDER BY r.arch, preset, r.iface_name, r.impl_index"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [{
        'id': row['id'],
        'task_id': row['task_id'],
        'arch': row['arch'],
        'preset': row['preset'],
        'iface_name': row['iface_name'],
        'impl_index': row['impl_index'],
        'result': row['result'],
        'result_data': row['result_data'],
        'error': row['error'],
        'gpu_id': row['gpu_id'],
        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
    } for row in rows]


def iter_perf_rows_for_export(conn, tag: str) -> Iterator[dict]:
    """Stream every `perf_measure` row whose preset was built against
    AOTriton version `tag` (e.g. '0.13b') -- the tag half of
    `perfmon.presets.parse_preset`'s `(rocm, tag)` pair.

    Matched as a `+aotriton<tag>` suffix on `task_config->>'preset'` rather
    than by parsing a preset apart into halves and comparing the tag half in
    Python: `rocm<ver>` is a single fleet-wide constant today (dispatch-
    perfmon.md §3.2), so a suffix match and a parsed-half match select the
    same rows, and the suffix form needs no `perfmon.presets` import here.
    Do not generalize this into per-half routing elsewhere -- multi-ROCm is
    explicitly out of scope.

    A generator so a caller building a large export (Stage F, not this
    module's concern) can process rows without holding the full result set;
    this function itself still executes one query and fetches once -- there
    is no live cursor kept open across yields.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT
                r.id,
                r.task_id,
                r.arch,
                r.iface_name,
                r.impl_index,
                r.result,
                r.result_data,
                r.error,
                r.gpu_id,
                r.created_at,
                q.task_config ->> 'preset' AS preset
            FROM task_reports r
            JOIN task_queue q
              ON q.id = r.task_id AND q.arch = r.arch AND q.class = r.class
            WHERE r.class = %s AND r.subclass = %s
              AND (q.task_config ->> 'preset') LIKE %s
            ORDER BY r.arch, preset, r.iface_name, r.impl_index
        """, [_KLASS, _SUBKLASS, f'%+aotriton{tag}'])
        rows = cur.fetchall()

    for row in rows:
        yield {
            'id': row['id'],
            'task_id': row['task_id'],
            'arch': row['arch'],
            'preset': row['preset'],
            'iface_name': row['iface_name'],
            'impl_index': row['impl_index'],
            'result': row['result'],
            'result_data': row['result_data'],
            'error': row['error'],
            'gpu_id': row['gpu_id'],
            'created_at': row['created_at'].isoformat() if row['created_at'] else None,
        }
