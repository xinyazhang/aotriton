-- Tuner v3.5 PostgreSQL Queue Schema
-- Copyright © 2026 Advanced Micro Devices, Inc.
-- SPDX-License-Identifier: MIT
--
-- Phase 2 (modularization unification, modular-tune.md §4.3/§4.7): the
-- Phase-1 `flash`/`flash_op` module-name split is gone. `module` now always
-- names a FAMILY ('flash', ...); the axis that used to be spelled as a
-- `_op`-suffixed module name is `tuning_level` ('kernel' | 'op'), stored as
-- its own NOT NULL column everywhere the old code matched on
-- `module (NOT) LIKE '%_op'`.
--
-- `iface_name`/`impl_index` replace the old `kernel_name`/`hsaco_index` and
-- `op_name`/`backend_index` column pairs (ImplSelector, modular-tune.md §4.2):
-- one interface-name/variant-index column pair shared by every tuning level,
-- not two level-specific ones.
--
-- Fresh schema, no migration (modular-tune.md, PR B directive): existing
-- `optune_results` / `best_optune_results` / `most_accurate_optune_results`
-- tables are gone outright, not ALTER'd. Re-initialize and re-tune.

-- NOTE: this file must stay pure SQL -- no psql backslash meta-commands.
-- It has two consumers: `.tune/bin/initdb` (psql -f) and
-- `pq/admin.py:QueueAdmin.init_schema()` (psycopg `cur.execute()`), and the
-- latter would reject a meta-command as a syntax error. Fail-fast belongs to
-- the caller: psycopg already runs this as one implicit transaction, and
-- initdb passes `-v ON_ERROR_STOP=1`.

-- perfmon rev2 (Stage R, R01): `class` names which DAG a queued row belongs
-- to -- today only 'tune_kernel'; a second DAG ('perf_measure') is added
-- later. It is promoted from the message-dict vocabulary
-- ({'class': 'tune_kernel', ...}, MessageHandler.get_class_name()) to a
-- column, so the queue row and the message it produces agree by
-- construction. NOT NULL, no DEFAULT: every dispatcher must name the DAG it
-- is queueing, so an insert that omits `class` fails loudly instead of
-- quietly becoming a tuning task.
--
-- `subclass` replaces the old `tuning_level` column name. The concept is
-- unchanged (still 'kernel'/'op' for tune_kernel rows) -- only the column
-- name moves, because its vocabulary is now defined per class (empty string
-- for 'perf_measure', see the composite CHECK below): `subclass` names the
-- *role*, which stays true for every class, where `tuning_level` would not.
-- The tuning workload keeps saying "tuning level" at its own surfaces --
-- --tuning_level on the CLI, task_config['tuning_level'], handlers.py,
-- tdesc.py -- a schema column rename does not oblige a CLI rename.
--
-- Sharded rank-2 on (arch, class): nested LIST partitioning, since
-- PostgreSQL rejects PARTITION BY LIST (arch, class) with "cannot use list
-- partition strategy with more than one column" (verified on 17.11). Every
-- partition -- both the per-arch level and its per-class leaves -- is
-- created by create_arch_partition() below, the single source of truth for
-- the class list, and lives in the `shards` schema so `\dt` in `public`
-- lists only this parent.
--
-- Parent table (partitioned by architecture)
--
-- TODO: targeted message delivery. Routing today is (arch, class, subclass):
-- a worker claims anything pending for its architecture, DAG class, and
-- subclass. That cannot express "only a host that has X provisioned may run
-- this", which is needed as soon as more than one build of the library under
-- test coexists on a fleet (e.g. measuring several AOTriton releases, where
-- the right binary must already be on the node).
--
-- Intended shape: a `tags` column on task_queue plus a matching set
-- advertised per worker, with the claim UPDATE gaining a containment
-- predicate so only a worker whose tags satisfy the task's may take it.
-- Keep the post-claim assertion pattern below (see fetch_tasks) when adding
-- it: a task claimed by a non-matching worker is a bug, not a warning.

-- Guard: refuse to apply onto a pre-rev2 database.
--
-- Every CREATE below is `IF NOT EXISTS`, which is right for re-applying onto a
-- database this same file already built, but silently WRONG against a stale
-- one: `task_queue` is skipped, keeps its old shape, and then every view,
-- index and function that references `class`/`subclass` fails in turn. The
-- operator sees a cascade of "column ... does not exist" and no statement of
-- the actual cause. Detect it once, here, and say so.
--
-- Deliberately checked by COLUMN, not by table existence: re-applying this
-- file onto its own output must stay a no-op.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'task_queue')
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'task_queue'
                  AND column_name  = 'class')
    THEN
        RAISE EXCEPTION
            'task_queue exists but has no `class` column: this database '
            'predates perfmon rev2 and there is no migration path'
        USING HINT =
            'Export anything you still need, then re-apply with a full drop: '
            '.tune/bin/initdb <workdir> --recreate';
    END IF;
END $$;

CREATE SCHEMA IF NOT EXISTS shards;

CREATE TABLE IF NOT EXISTS task_queue (
    id BIGSERIAL,
    arch TEXT NOT NULL,
    module TEXT NOT NULL,
    class TEXT NOT NULL CHECK (class IN ('tune_kernel', 'perf_measure')),
    subclass TEXT NOT NULL DEFAULT '',
    task_config JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending/running/completed/failed/cancelled
    priority INT DEFAULT 5,
    worker_id TEXT,
    node_hostname TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    retry_count INT DEFAULT 0,
    PRIMARY KEY (id, arch, class),
    -- subclass's vocabulary is defined per class -- enforced, not just
    -- conventional, so a stray value cannot slip in undetected.
    CHECK (
        (class = 'tune_kernel'  AND subclass IN ('kernel', 'op')) OR
        (class = 'perf_measure' AND subclass = '')
    )
) PARTITION BY LIST (arch);

-- Worker heartbeat table (for monitoring and health checks)
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    node_hostname TEXT NOT NULL,
    worker_name TEXT NOT NULL,
    arch TEXT NOT NULL,
    last_heartbeat TIMESTAMP NOT NULL DEFAULT NOW(),
    status TEXT NOT NULL DEFAULT 'active',  -- active/idle/dead
    tasks_completed INT DEFAULT 0,
    tasks_failed INT DEFAULT 0,
    PRIMARY KEY (node_hostname, worker_name)
);

CREATE INDEX IF NOT EXISTS idx_worker_heartbeat_alive
    ON worker_heartbeat (last_heartbeat DESC)
    WHERE status = 'active';

-- Per-task report rows: one row per (task_id, iface_name, impl_index).
--
-- Named `task_reports`, not `tuning_results`, because it now carries the
-- output of every DAG class, not just tuning (perfmon rev2 R06). The name is
-- the same correction `tuning_level` -> `subclass` makes: a framework table
-- must not be named after one of the workloads it serves.
--
-- `subclass` is denormalized here (not just on task_queue) because this
-- table is frequently queried by (iface_name, impl_index) alone, without a
-- task_queue join, and `iface_name` collides across subclasses -- e.g.
-- 'attn_fwd' is valid at both the kernel and op level (highest-risk area #2
-- of modular-tune.md). That reasoning is unchanged by the rename, and every
-- existing consumer already filters on it, so a second class cannot leak
-- into the best-results pipeline.
--
-- `arch` and `class` are new. Both are required by the shard key; `arch` is
-- additionally what makes pruning possible when reading one column of the
-- matrix.
CREATE TABLE IF NOT EXISTS task_reports (
    id BIGSERIAL,
    task_id BIGINT NOT NULL,
    arch TEXT NOT NULL,
    class TEXT NOT NULL CHECK (class IN ('tune_kernel', 'perf_measure')),
    subclass TEXT NOT NULL DEFAULT '',
    iface_name TEXT NOT NULL,
    impl_index INT NOT NULL,
    result TEXT NOT NULL,  -- OK/NotOK/crash/ERROR
    result_data JSONB,
    error JSONB,
    gpu_id INT,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, arch, class),
    -- Same per-class vocabulary as task_queue: the two must agree, since a
    -- report is written for a row that passed that CHECK.
    CHECK (
        (class = 'tune_kernel'  AND subclass IN ('kernel', 'op')) OR
        (class = 'perf_measure' AND subclass = '')
    )
) PARTITION BY LIST (arch);

-- Utility views for monitoring
CREATE OR REPLACE VIEW queue_progress AS
SELECT
    arch,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'running') as running,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) as pct_complete
FROM task_queue
GROUP BY arch
ORDER BY arch;

CREATE OR REPLACE VIEW kernel_queue_progress AS
SELECT
    arch,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'running') as running,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) as pct_complete
FROM task_queue
WHERE class = 'tune_kernel' AND subclass = 'kernel'
GROUP BY arch
ORDER BY arch;

CREATE OR REPLACE VIEW op_queue_progress AS
SELECT
    arch,
    COUNT(*) FILTER (WHERE status = 'pending') as pending,
    COUNT(*) FILTER (WHERE status = 'running') as running,
    COUNT(*) FILTER (WHERE status = 'completed') as completed,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE status = 'cancelled') as cancelled,
    COUNT(*) as total,
    ROUND(100.0 * COUNT(*) FILTER (WHERE status = 'completed') / NULLIF(COUNT(*), 0), 2) as pct_complete
FROM task_queue
WHERE class = 'tune_kernel' AND subclass = 'op'
GROUP BY arch
ORDER BY arch;

CREATE OR REPLACE VIEW worker_health AS
SELECT
    node_hostname,
    worker_name,
    arch,
    status,
    tasks_completed,
    tasks_failed,
    EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) as seconds_since_heartbeat,
    CASE
        WHEN EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) < 60 THEN 'healthy'
        WHEN EXTRACT(EPOCH FROM (NOW() - last_heartbeat)) < 300 THEN 'stale'
        ELSE 'dead'
    END as health_status
FROM worker_heartbeat
ORDER BY last_heartbeat DESC;

CREATE OR REPLACE VIEW task_timing_stats AS
SELECT
    arch,
    module,
    class,
    subclass,
    COUNT(*) as completed_tasks,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (completed_at - started_at))) as median_duration_sec,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (completed_at - started_at))) as p95_duration_sec,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_sec
FROM task_queue
WHERE status = 'completed' AND completed_at IS NOT NULL
GROUP BY arch, module, class, subclass;

CREATE OR REPLACE VIEW stale_tasks AS
SELECT
    id,
    arch,
    worker_id,
    node_hostname,
    EXTRACT(EPOCH FROM (NOW() - started_at)) / 3600 as hours_running
FROM task_queue
WHERE status = 'running'
  AND EXTRACT(EPOCH FROM (NOW() - started_at)) > 7200  -- Running > 2 hours
ORDER BY started_at ASC;

CREATE OR REPLACE VIEW completion_eta AS
WITH stats AS (
    SELECT
        arch,
        COUNT(*) FILTER (WHERE status = 'pending') as remaining,
        COUNT(*) FILTER (WHERE status = 'running') as active_workers,
        AVG(EXTRACT(EPOCH FROM (completed_at - started_at)))
            FILTER (WHERE status = 'completed' AND completed_at > NOW() - INTERVAL '1 hour')
            as avg_task_duration_sec
    FROM task_queue
    GROUP BY arch
)
SELECT
    arch,
    remaining,
    active_workers,
    avg_task_duration_sec,
    CASE
        WHEN active_workers > 0 AND avg_task_duration_sec IS NOT NULL THEN
            ROUND((remaining * avg_task_duration_sec / active_workers) / 3600, 2)
        ELSE NULL
    END as eta_hours
FROM stats
WHERE remaining > 0;

-- Function to create the (arch, class) partition tree for an architecture.
--
-- Rank-2 sharding on (arch, class): PostgreSQL rejects
-- `PARTITION BY LIST (arch, class)` with "cannot use list partition strategy
-- with more than one column" (verified on 17.11), so the arch-level
-- partition is itself declared `PARTITION BY LIST (class)` and this function
-- creates both levels. The ARRAY literal below is the single source of
-- truth for the set of DAG classes -- adding a class here is the only place
-- that needs to change to grow the class list; there is no DEFAULT
-- partition at either level, so an insert naming an unregistered class (or
-- arch) fails loudly with "no partition of relation ... found for row"
-- instead of being silently absorbed.
--
-- All partitions -- arch-level and class-level leaves -- live in the
-- `shards` schema, not `public`, so `\dt` (default search_path) lists only
-- the `task_queue` parent.
CREATE OR REPLACE FUNCTION create_arch_partition(arch_name TEXT)
RETURNS VOID AS $$
DECLARE
    arch_partition TEXT;
    class_name TEXT;
    class_partition TEXT;
    rpt_arch_partition TEXT;
    rpt_class_partition TEXT;
BEGIN
    arch_partition := 'task_queue_' || arch_name;
    rpt_arch_partition := 'task_reports_' || arch_name;

    -- Top level: one LIST partition per arch, itself sub-partitioned by class.
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS shards.%I PARTITION OF task_queue '
        'FOR VALUES IN (%L) PARTITION BY LIST (class)',
        arch_partition, arch_name
    );

    FOREACH class_name IN ARRAY ARRAY['tune_kernel', 'perf_measure']
    LOOP
        class_partition := arch_partition || '_' || class_name;

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS shards.%I PARTITION OF shards.%I '
            'FOR VALUES IN (%L)',
            class_partition, arch_partition, class_name
        );

        -- Indexes on the leaf (arch, class) partition. subclass leads
        -- (alongside status) since fetch_tasks() always filters on class,
        -- subclass, and status together -- a kernel worker must never claim
        -- an op task and vice versa (F16).
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON shards.%I '
            '(class, subclass, status, priority DESC, id ASC) WHERE status = %L',
            class_partition || '_fetch', class_partition, 'pending'
        );

        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON shards.%I (worker_id, status)',
            class_partition || '_worker', class_partition
        );

        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON shards.%I (created_at DESC)',
            class_partition || '_created', class_partition
        );
    END LOOP;

    -- task_reports gets the same (arch, class) tree, for the same reason and
    -- with the same no-DEFAULT-partition property: a report naming an
    -- unregistered class fails loudly rather than landing in a catch-all.
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS shards.%I PARTITION OF task_reports '
        'FOR VALUES IN (%L) PARTITION BY LIST (class)',
        rpt_arch_partition, arch_name
    );

    FOREACH class_name IN ARRAY ARRAY['tune_kernel', 'perf_measure']
    LOOP
        rpt_class_partition := rpt_arch_partition || '_' || class_name;

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS shards.%I PARTITION OF shards.%I '
            'FOR VALUES IN (%L)',
            rpt_class_partition, rpt_arch_partition, class_name
        );

        -- The two read patterns this table actually has: by task, and by
        -- (subclass, iface_name) without a task_queue join -- which is the
        -- whole reason subclass is denormalized onto it.
        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON shards.%I (task_id, iface_name, impl_index)',
            rpt_class_partition || '_task', rpt_class_partition
        );

        EXECUTE format(
            'CREATE INDEX IF NOT EXISTS %I ON shards.%I (subclass, iface_name, result)',
            rpt_class_partition || '_iface', rpt_class_partition
        );
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- best_tuning_results: plain table populated by compute_best_results.py.
-- Unified: replaces the former best_tuning_results/best_optune_results pair.
-- For each (task_id, iface_name): fastest impl_index meeting the accuracy
-- threshold relative to most_accurate_tuning_results.
CREATE TABLE IF NOT EXISTS best_tuning_results (
    task_id      BIGINT    NOT NULL,
    arch         TEXT      NOT NULL,
    tuning_level TEXT      NOT NULL CHECK (tuning_level IN ('kernel', 'op')),
    task_config  JSONB     NOT NULL,
    iface_name   TEXT      NOT NULL,
    impl_index   INT       NOT NULL,
    median_time  FLOAT     NOT NULL,
    impl_desc    JSONB,
    computed_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (task_id, iface_name)
);

CREATE INDEX IF NOT EXISTS idx_best_tuning_results_lookup
    ON best_tuning_results (arch, tuning_level, iface_name, task_id);

-- Extra unit tests associated with a task, populated by reset_broken_to_pending
-- when re-queuing entries that failed pytest correctness checks.
-- Rows accumulate across passes and are never deleted by reset_to_pending.
CREATE TABLE IF NOT EXISTS task_extra_uts (
    id          BIGSERIAL PRIMARY KEY,
    task_id     BIGINT  NOT NULL,
    im_text     TEXT    NOT NULL CHECK (im_text NOT LIKE '% %'),
    active      BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (task_id, im_text)
);

CREATE INDEX IF NOT EXISTS idx_task_extra_uts_task_id
    ON task_extra_uts (task_id);
