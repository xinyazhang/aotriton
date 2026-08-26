# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Core queue operations for Tuner v3.5

Provides atomic task fetching, status updates, and queue management
using PostgreSQL SELECT FOR UPDATE SKIP LOCKED.
"""

import psycopg
import json
from dataclasses import asdict, is_dataclass

from psycopg.rows import dict_row
from dataclasses import dataclass
from datetime import datetime
import socket
import os
import logging

logger = logging.getLogger(__name__)


def entry_filter(entry, *, arch=None, klass=None, tuning_level=None, module=None):
    """WHERE fragment selecting task_queue rows for one tuning entry.

    Returns (sql, params). `entry` is a mapping of entry fields, or any
    dataclass instance (a family's Entry class) -- the field names are the
    family's, but the shape of the query is not, so it lives here with the
    rest of the task_queue schema knowledge.

    Field types drive the cast, because task_config is JSONB and ->> yields
    text. bool is tested before int deliberately: bool is a subclass of int
    in Python, so the order is load-bearing. Tuples are compared as JSON
    arrays via -> rather than ->>, which is how a composite value such as an
    asymmetric hdim is stored.
    """
    if is_dataclass(entry) and not isinstance(entry, type):
        entry = asdict(entry)
    clauses, params = [], []
    if arch is not None:
        clauses.append("task_config->>'arch' = %s")
        params.append(arch)
    if klass is not None:
        clauses.append('class = %s')
        params.append(klass)
    if tuning_level is not None:
        clauses.append('subclass = %s')
        params.append(tuning_level)
    if module is not None:
        clauses.append('module = %s')
        params.append(module)
    for field_name, value in entry.items():
        col = f"task_config->'entry'->>'{field_name}'"
        if isinstance(value, (tuple, list)):
            clauses.append(f"task_config->'entry'->'{field_name}' = %s::jsonb")
            params.append(json.dumps(list(value)))
            continue
        if isinstance(value, bool):
            clauses.append(f'({col})::boolean = %s')
        elif isinstance(value, int):
            clauses.append(f'({col})::integer = %s')
        elif isinstance(value, float):
            clauses.append(f'({col})::float = %s')
        else:
            clauses.append(f'{col} = %s')
        params.append(value)
    return ' AND '.join(clauses), params


# workload -> (task_queue.class, task_queue.subclass) for fetch_tasks().
#
# The workload a node serves determines both, so it is the only thing callers
# pass.
#
# The second element is matched literally against the denormalized
# task_queue.subclass column (`AND subclass = %s`), so it is not free-form:
# it must be a value the schema permits for that class. schema.sql enforces
# the vocabulary:
#
#     CHECK ((class = 'tune_kernel'  AND subclass IN ('kernel', 'op')) OR
#            (class = 'perf_measure' AND subclass = ''))
#
# which is why perfmon maps to the EMPTY string and not to 'kernel'. Pairing
# perf_measure with 'kernel' produces a predicate the CHECK guarantees can
# never match, so the worker claims nothing and does so silently -- no error,
# no warning, just an idle reader. An earlier revision of this table did
# exactly that on the theory that subclass was inert for perfmon.
#
# This mapping is where the tuning-domain vocabulary stops. `workload` is a
# node-kind concept the .tune/ scripts own; below this line only the generic
# queue terms class/subclass travel, because the queue also carries
# perf_measure work that has no "tuning mode" at all.
WORKLOAD_TASK_SELECTOR: dict[str, tuple[str, str]] = {
    'kernel':  ('tune_kernel', 'kernel'),
    'op':      ('tune_kernel', 'op'),
    'perfmon': ('perf_measure', ''),
}


class TaskSubclassMismatch(RuntimeError):
    """fetch_tasks() claimed a task of a class/subclass it did not ask for.

    Only reachable if the UPDATE's `subclass` predicate is broken, since
    PostgreSQL would otherwise not return the row. Systemic rather than
    transient: the filter is the sole thing keeping a worker off tasks it
    cannot execute -- for a tuning worker, tasks its pyaotriton build does
    not implement -- so every later claim is suspect too.
    """


@dataclass
class Task:
    """Task representation.

    `klass` names the DAG (perfmon rev2 R01; 'tune_kernel' | 'perf_measure').
    Spelled `klass`, not `class`, because `class` is a Python keyword --
    the SQL column is still `class` (see schema.sql / queue SQL below,
    which alias it `AS klass` when needed). `subclass` replaces the old
    `tuning_level` field name, mirroring the task_queue column rename;
    the concept is unchanged ('kernel' | 'op' for tune_kernel rows).
    """
    id: int
    arch: str
    module: str
    klass: str
    subclass: str
    task_config: dict
    status: str
    priority: int = 5
    worker_id: str | None = None
    node_hostname: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0


class TaskQueue:
    """PostgreSQL-based task queue with architecture partitioning"""

    def __init__(self, conn):
        """
        Initialize task queue.

        Args:
            conn: PostgreSQL connection (from psycopg.connect)
        """
        self.conn = conn
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}"
        self.node_hostname = socket.gethostname()

    def fetch_tasks(self, arch: str, batch_size: int = 10, *, klass: str,
                    subklass: str) -> list[Task]:
        """
        Fetch pending tasks for a specific architecture.

        Uses SELECT FOR UPDATE SKIP LOCKED for atomic task claiming.
        Queries the (arch, class) leaf partition directly for performance
        (perfmon rev2 R01: task_queue is sharded rank-2 on (arch, class),
        with leaves living in the `shards` schema).

        Args:
            arch: GPU architecture (e.g., 'gfx942', 'gfx90a')
            batch_size: Number of tasks to fetch
            klass: which DAG to fetch for: 'tune_kernel' | 'perf_measure'.
                Filters on task_queue.class and selects the (arch, class)
                leaf partition.
            subklass: the task_queue.subclass value to claim. Matched
                literally, so it must be one schema.sql's CHECK permits for
                this class: 'kernel' | 'op' for tune_kernel, '' for
                perf_measure. Pairing a class with a subclass the CHECK
                forbids yields a predicate that can never match, and the
                worker then idles silently.

                Both are REQUIRED and keyword-only, with no default: a
                worker must never claim a task of a class or subclass it
                cannot execute (see modular-tune.md F16 for the tuning case),
                so a caller that forgets one fails at the call site rather
                than silently defaulting.

                These are deliberately GENERIC queue terms. `tuning_mode` is
                a tuning-domain name and belongs to tuning-specific code --
                the scripts under .tune/ and the tuning DAG -- not to the
                task queue, which also carries perf_measure work.

        Returns:
            List of claimed Task objects

        TODO: targeted message delivery. `arch` + `klass` + `subklass` is
        the whole routing vocabulary today, so any worker of the right arch,
        class, and level can claim any pending task. It cannot express a
        per-task requirement on what the *host* has provisioned -- which is
        needed once several builds of the library under test coexist on one
        fleet.

        Intended shape: a `tags` column on task_queue (see schema.sql) and a
        tag set per worker, with `AND <task tags are satisfied by worker
        tags>` added to the claim UPDATE below, plus the same post-claim
        assertion this method already makes for `subklass` -- claiming a
        task whose tags do not match is a bug, not a warning.
        """
        partition_table = f"shards.task_queue_{arch}_{klass}"

        with self.conn.cursor(row_factory=dict_row) as cur:
            # Atomic task claiming using UPDATE ... RETURNING
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'running',
                    worker_id = %s,
                    node_hostname = %s,
                    started_at = NOW()
                WHERE id IN (
                    SELECT id FROM {partition_table}
                    WHERE status = 'pending'
                      AND class = %s
                      AND subclass = %s
                    ORDER BY priority DESC, id ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING id, arch, module, class AS klass, subclass, task_config, status, priority,
                          worker_id, node_hostname, created_at, started_at,
                          completed_at, error, retry_count
            """, (self.worker_id, self.node_hostname, klass, subklass, batch_size))

            try:
                rows = cur.fetchall()
            except psycopg.errors.QueryCanceled:
                logger.warning(f"TaskQueue.fetch_tasks: statement_timeout hit for {partition_table}")
                return []

            tasks = [Task(**row) for row in rows]

            # The UPDATE above filters on class and subclass, so a row of the
            # wrong class/level means that predicate is broken. Release the
            # whole batch -- connections here are autocommit, so the claim is
            # already durable and raising without this would strand every row
            # in 'running' -- then fail. The batch is released entirely, not
            # just the offending rows: this raises out of the worker, so
            # correctly-claimed tasks would be stranded too.
            wrong = [t for t in tasks if t.klass != klass or t.subclass != subklass]
            if wrong:
                cur.execute(f"""
                    UPDATE {partition_table}
                       SET status = 'pending', worker_id = NULL,
                           node_hostname = NULL, started_at = NULL
                     WHERE id = ANY(%s)
                """, ([t.id for t in tasks],))
                raise TaskSubclassMismatch(
                    f"fetch_tasks({arch!r}, klass={klass!r}, subklass={subklass!r}) "
                    f"claimed task_ids={[t.id for t in wrong]} with (class, subclass)="
                    f"{sorted({(t.klass, t.subclass) for t in wrong})}; the class/subclass "
                    f"filter is not doing its job. Released "
                    f"{len(tasks)} claim(s) back to pending.")

            if tasks:
                task_ids = [t.id for t in tasks]
                logger.info(f"TaskQueue.fetch_tasks: Claimed {len(tasks)} task(s) from {partition_table}: "
                           f"task_ids={task_ids}, status=pending→running, worker_id={self.worker_id}")

            return tasks

    def mark_completed(self, task_id: int, arch: str) -> None:
        """
        Mark task as completed.

        Targets the per-arch partition rather than the (arch, class) leaf, and
        so does every other by-id mutator below. That is deliberate, not an
        oversight: these are addressed by primary key, PostgreSQL narrows to
        at most one leaf per arch on its own, and none of the call sites (the
        localq handlers) carry the task's class -- threading it through every
        one of them would be real churn to save a scan of a second index.
        Only fetch_tasks(), which selects rather than mutates and always knows
        its class, targets the leaf directly.

        Args:
            task_id: Task ID
            arch: GPU architecture (for partition routing)
        """
        partition_table = f"shards.task_queue_{arch}"

        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'completed',
                    completed_at = NOW()
                WHERE id = %s
            """, (task_id,))

            logger.info(f"TaskQueue.mark_completed: task_id={task_id}, arch={arch}, "
                       f"status=→completed, partition={partition_table}")

    def mark_failed(self, task_id: int, *, arch: str | None = None, error_message: str) -> None:
        """
        Mark task as failed with error message.

        Args:
            task_id: Task ID
            arch: GPU architecture (for partition routing, optional, keyword-only)
            error_message: Error message (keyword-only)
        """
        with self.conn.cursor() as cur:
            if arch:
                partition_table = f"shards.task_queue_{arch}"
                cur.execute(f"""
                    UPDATE {partition_table}
                    SET status = 'failed',
                        completed_at = NOW(),
                        error = %s
                    WHERE id = %s
                """, (error_message, task_id))
                logger.error(f"TaskQueue.mark_failed: task_id={task_id}, arch={arch}, "
                            f"status=→failed, partition={partition_table}, error={error_message}")
            else:
                # Update parent table when arch unknown
                cur.execute("""
                    UPDATE task_queue
                    SET status = 'failed',
                        completed_at = NOW(),
                        error = %s
                    WHERE id = %s
                """, (error_message, task_id))
                logger.error(f"TaskQueue.mark_failed: task_id={task_id}, arch=unknown, "
                            f"status=→failed, partition=task_queue (parent), error={error_message}")

    def mark_pending(self, task_id: int, arch: str) -> None:
        """
        Mark task as pending (used during graceful shutdown to cancel running tasks).

        Status only -- existing results are left alone. See reset_to_pending()
        for the bulk re-run path, which can also discard them.

        Args:
            task_id: Task ID
            arch: GPU architecture (for partition routing)
        """
        partition_table = f"shards.task_queue_{arch}"

        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'pending',
                    worker_id = NULL,
                    node_hostname = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    error = NULL
                WHERE id = %s
            """, (task_id,))

            logger.info(f"TaskQueue.mark_pending: task_id={task_id}, arch={arch}, "
                       f"status=→pending, partition={partition_table}")

    def retry_task(self, task_id: int, arch: str, max_retries: int = 3) -> bool:
        """
        Retry a failed task if under retry limit.

        Args:
            task_id: Task ID
            arch: GPU architecture
            max_retries: Maximum retry attempts

        Returns:
            True if task was retried, False if max retries exceeded
        """
        partition_table = f"shards.task_queue_{arch}"

        with self.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {partition_table}
                SET status = 'pending',
                    retry_count = retry_count + 1,
                    worker_id = NULL,
                    node_hostname = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    error = NULL
                WHERE id = %s
                  AND retry_count < %s
                RETURNING id
            """, (task_id, max_retries))

            result = cur.fetchone()
            return result is not None

    def get_queue_stats(self, arch: str | None = None) -> dict[str, int]:
        """
        Get queue statistics.

        Args:
            arch: Optional architecture filter (None = all architectures)

        Returns:
            Dictionary with pending, running, completed, failed, cancelled counts
        """
        with self.conn.cursor() as cur:
            if arch:
                partition_table = f"shards.task_queue_{arch}"
                cur.execute(f"""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'running') as running,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed
                    FROM {partition_table}
                """)
            else:
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'pending') as pending,
                        COUNT(*) FILTER (WHERE status = 'running') as running,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed
                    FROM task_queue
                """)

            row = cur.fetchone()
            return dict(row) if row else {'pending': 0, 'running': 0, 'completed': 0, 'failed': 0}

    def detect_stale_tasks(self, timeout_seconds: int = 7200) -> list[Task]:
        """
        Detect tasks running longer than timeout.

        Args:
            timeout_seconds: Task timeout in seconds (default: 2 hours)

        Returns:
            List of stale tasks
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""
                SELECT id, arch, module, class AS klass, subclass, task_config, status, priority,
                       worker_id, node_hostname, created_at, started_at,
                       completed_at, error, retry_count
                FROM task_queue
                WHERE status = 'running'
                  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > %s
                ORDER BY started_at ASC
            """, (timeout_seconds,))

            rows = cur.fetchall()
            return [Task(**row) for row in rows]

    def reset_stale_tasks(self, timeout_seconds: int = 7200) -> int:
        """
        Reset stale tasks back to pending status.

        Args:
            timeout_seconds: Task timeout in seconds

        Returns:
            Number of tasks reset
        """
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE task_queue
                SET status = 'pending',
                    worker_id = NULL,
                    node_hostname = NULL,
                    started_at = NULL,
                    retry_count = retry_count + 1
                WHERE status = 'running'
                  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > %s
                RETURNING id
            """, (timeout_seconds,))

            count = len(cur.fetchall())
            return count

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Entry / id lookups
    # ------------------------------------------------------------------

    _LOOKUP_COLUMNS = 'id, arch, module, class AS klass, subclass, status'

    def find_by_entry(self, entry, *, arch=None, klass=None, tuning_level=None,
                      module=None, columns: str | None = None,
                      limit: int | None = None) -> list[dict]:
        """task_queue rows matching one tuning entry.

        Filters are optional so a caller can be as specific as it can be: a
        pytest node ID carries no arch, while a TUNE_V3BIS line does. Supply
        tuning_level whenever it is known -- both levels share arch and every
        task_config field, so omitting it matches an entry's rows at both.

        `klass` narrows to one DAG. Supplying it matters for a caller that
        omits tuning_level: entry fields are shared across classes, so an
        unfiltered lookup can match a row belonging to a different workflow
        entirely. A caller that does pass tuning_level is already narrowed --
        'kernel'/'op' cannot collide with perf_measure's empty subclass -- but
        relying on that is relying on a coincidence of vocabularies.
        """
        where, params = entry_filter(entry, arch=arch, klass=klass,
                                     tuning_level=tuning_level, module=module)
        sql = f'SELECT {columns or self._LOOKUP_COLUMNS} FROM task_queue'
        if where:
            sql += f' WHERE {where}'
        sql += ' ORDER BY arch, id'
        if limit is not None:
            sql += f' LIMIT {int(limit)}'
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def find_by_ids(self, task_ids: list[int],
                    columns: str | None = None) -> list[dict]:
        """task_queue rows for the given ids, in (arch, id) order."""
        if not task_ids:
            return []
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f'SELECT {columns or self._LOOKUP_COLUMNS} FROM task_queue '
                f'WHERE id = ANY(%s) ORDER BY arch, id', (task_ids,))
            return cur.fetchall()

    def get_progress(self, klass: str, subklass: str, *,
                     recent_window: str = '5 minutes',
                     stale_seconds: int = 7200) -> dict:
        """Per-arch queue progress for one (class, subclass).

        Returns {'progress': [...], 'speed': [...], 'stale': [...]}: the
        slice's queue-progress rows, recent completion counts, and
        long-running task counts. Callers merge and format these; the SQL and
        the predicates live here so a queue-schema change does not have to be
        mirrored into the web UI (see pq/README.md).

        There used to be one view per tuning level and a dict mapping the
        level to a view name. R07 collapsed those into a single
        queue_progress grouped by (arch, class, subclass), so selecting a
        slice is now a WHERE clause -- and a new class needs no view, no dict
        entry and no schema change.
        """
        with self.conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                'SELECT * FROM queue_progress '
                'WHERE class = %s AND subclass = %s ORDER BY arch',
                (klass, subklass))
            progress = cur.fetchall()

            cur.execute("""
                SELECT arch, COUNT(*) AS recent_completions
                FROM task_queue
                WHERE status = 'completed'
                  AND completed_at > NOW() - %s::interval
                  AND class = %s AND subclass = %s
                GROUP BY arch
            """, (recent_window, klass, subklass))
            speed = cur.fetchall()

            cur.execute("""
                SELECT arch, COUNT(*) AS stale_count
                FROM task_queue
                WHERE status = 'running'
                  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > %s
                  AND class = %s AND subclass = %s
                GROUP BY arch
            """, (stale_seconds, klass, subklass))
            stale = cur.fetchall()
        return {'progress': progress, 'speed': speed, 'stale': stale}

    def reset_to_pending(self, row_ids: list[int], tuning_level: str, *,
                         delete_results: bool) -> int:
        """Reset the given task_queue rows to pending, for re-running.

        delete_results is keyword-only and REQUIRED because it is destructive
        and the destruction is not implied by the method name. With it set,
        the tasks' task_reports and most_accurate_tuning_results rows are
        dropped as well -- GPU-hours of measurements -- so that a re-run
        starts clean instead of mixing new results with stale ones. Pass False
        to requeue while keeping the existing rows.

        Compare mark_pending(), which only moves a single task back to pending
        and never touches results.

        Every statement is scoped by tuning_level as well as id: callers select
        ids by arch/entry, which both levels share, so an id list can span
        levels. Returns the number of task_queue rows actually reset.
        """
        if not row_ids:
            return 0
        with self.conn.cursor() as cur:
            if delete_results:
                cur.execute(
                    'DELETE FROM most_accurate_tuning_results '
                    'WHERE tuning_level = %s AND task_id = ANY(%s)',
                    (tuning_level, row_ids))
                cur.execute(
                    'DELETE FROM task_reports '
                    'WHERE subclass = %s AND task_id = ANY(%s)',
                    (tuning_level, row_ids))
            cur.execute("""
                UPDATE task_queue
                   SET status       = 'pending',
                       worker_id    = NULL,
                       node_hostname= NULL,
                       started_at   = NULL,
                       completed_at = NULL,
                       error        = NULL
                 WHERE subclass = %s AND id = ANY(%s)
            """, (tuning_level, row_ids))
            return cur.rowcount
