#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Dispatch tuning tasks to PostgreSQL queue (Tuner v3.5).

This script:
1. Loads a tuning module (e.g., 'flash')
2. Queries the module's parameter choices
3. Allows filtering via command-line arguments
4. Dispatches tasks to PostgreSQL queue using bulk INSERT
"""

import sys
import os
import argparse
import sqlite3
from pathlib import Path
from dataclasses import fields, asdict
import json

def load_config(workdir: Path):
    """Load config.rc from workdir and set environment variables."""
    config_rc = workdir / 'config.rc'
    if not config_rc.exists():
        sys.exit(f"Error: config.rc not found at {config_rc}")

    # Parse config.rc and set environment variables
    with open(config_rc) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')].strip()
            if '=' in line:
                # Simple parsing: KEY=VALUE
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

def get_db_connection_params():
    """Get PostgreSQL connection parameters from environment."""
    postgres_user = os.environ.get('POSTGRES_USER')
    postgres_password = os.environ.get('POSTGRES_PASSWORD')
    celery_service_host = os.environ.get('CELERY_SERVICE_HOST')
    postgres_port = os.environ.get('POSTGRES_PORT')

    if not all([postgres_user, postgres_password, celery_service_host, postgres_port]):
        sys.exit("Error: Missing PostgreSQL credentials in config.rc. "
                 "Required: POSTGRES_USER, POSTGRES_PASSWORD, CELERY_SERVICE_HOST, POSTGRES_PORT")

    return {
        'host': celery_service_host,
        'port': int(postgres_port),
        'user': postgres_user,
        'password': postgres_password,
    }

def get_parameter_choices(module_instance):
    """
    Get parameter choices from module.

    Returns the ENTRY_CLASS instance where each field is a list of choices.
    """
    return module_instance.get_entry_choices()

def generate_filtered_entries(module_instance, args):
    """
    Generate entries filtered by command-line arguments.

    args should have attributes matching entry field names,
    each containing a list of allowed values (or None for no filter).
    """
    # Get all choices
    all_choices = module_instance.get_entry_choices()

    # Filter choices based on command-line arguments
    filtered_choices = {}
    for field in fields(all_choices):
        all_values = getattr(all_choices, field.name)
        filter_values = getattr(args, field.name, None)

        if filter_values is None:
            # No filter specified, use all values
            filtered_choices[field.name] = all_values
        else:
            # Filter the choices
            # Convert bool to int for comparison if needed
            def normalize(v):
                return int(v) if isinstance(v, bool) else v

            normalized_filter = [normalize(v) for v in filter_values]
            filtered_values = [v for v in all_values if normalize(v) in normalized_filter]

            if not filtered_values:
                # No values match the filter - empty result
                return

            filtered_choices[field.name] = filtered_values

    # Create filtered choices instance
    ENTRY_CLASS = type(all_choices)
    filtered_choices_obj = ENTRY_CLASS(**filtered_choices)

    # Generate entries from filtered choices
    yield from module_instance.generate_entries_from_choices(filtered_choices_obj)

def get_registered_archs(workdir: Path) -> list[str]:
    """Get list of registered architectures from workers.db."""
    db_path = workdir / 'workers.db'
    if not db_path.exists():
        sys.exit(f"Error: workers.db not found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT DISTINCT arch FROM workers ORDER BY arch")
        archs = [row[0] for row in cursor.fetchall()]
        return archs
    finally:
        conn.close()

def get_completed_tasks(module_name: str, module_instance, subklass: str, verbose: bool = False):
    """
    Query PostgreSQL for completed tasks from task_queue.

    Returns a set of task_config tuples (hashable form) that have
    successfully completed (status = 'completed').

    Args:
        module_name: Name of the tuning module (e.g., 'flash')
        module_instance: Module instance with ENTRY_CLASS defining field structure
        subklass: 'kernel' or 'op' -- filters on the denormalized
            task_queue.subclass column so kernel-level and op-level
            completed tasks (which may share the same module name and entry
            fields) are never conflated. D09: this is `args.workload`
            (formerly the removed tuning-mode flag -- see
            `build_base_parser()`); named `subklass` here, not that old
            flag's name, since it is really `TaskQueue.completed_task_
            configs`'s `subklass` parameter passed straight through.
        verbose: Print debug info

    Raises exception if connection fails - caller should handle errors.

    D08: the SELECT itself now lives in pq/ (TaskQueue.completed_task_configs)
    -- CLAUDE.md forbids raw SQL outside pq/, and the query used to hardcode
    class='tune_kernel', which this function is the only caller of, so
    it is passed explicitly here rather than assumed by the query.
    """
    import psycopg
    from psycopg.rows import dict_row
    from .pq.queue import TaskQueue
    from .dispatch.driver import make_hashable_key

    # Get PostgreSQL connection parameters
    conn_params = get_db_connection_params()

    # Extract field names once for reuse (avoid repeated metadata access)
    entry_class = module_instance.ENTRY_CLASS
    entry_field_names = tuple(f.name for f in fields(entry_class))

    # Connect to PostgreSQL - let exceptions propagate
    conn = psycopg.connect(**conn_params, row_factory=dict_row)

    try:
        task_queue = TaskQueue(conn)
        task_configs = task_queue.completed_task_configs(
            module_name, klass='tune_kernel', subklass=subklass)

        completed_configs = {make_hashable_key(entry_field_names, tc)
                              for tc in task_configs}

        if verbose:
            print(f"Found {len(completed_configs)} completed tasks for module '{module_name}'")

        return completed_configs

    finally:
        conn.close()

class TuneEntrySource:
    """`dispatch.driver.EntrySource` for tuning (D07): cross product via
    `generate_filtered_entries`, a single `None` batch (tuning has no
    insert-order dependency, dispatch-perfmon.md §5.7), `class`/`subclass`
    from D01's `WORKLOAD_TASK_SELECTOR` table.

    `entries()` regenerates the filtered cross product once per `arch`
    (`generate_filtered_entries` is deterministic and side-effect-free), so
    the driver's batch/arch/entry loop nesting visits the same (arch,
    entry) pairs as before, just in a different order -- §5.7 again: order
    does not matter here."""

    def __init__(self, module_name: str, module_instance, args):
        self.module_name = module_name
        self.module_instance = module_instance
        self.args = args
        self.ENTRY_CLASS = module_instance.ENTRY_CLASS

    def batches(self):
        yield None

    def entries(self, batch, arch):
        yield from generate_filtered_entries(self.module_instance, self.args)

    def validate_hw_feature(self, arch, entry):
        return self.module_instance.validate_hw_feature(arch, entry)

    def task_config(self, batch, arch, entry) -> dict:
        task_config = {
            "arch": arch,
            "module": self.module_name,
            # D09: the old tuning-mode flag is gone; `--workload` (Phase 1)
            # already takes exactly the values 'kernel'/'op' for a tuning
            # workload, identical to WORKLOAD_TASK_SELECTOR's subclass for
            # those two.
            "tuning_level": self.args.workload,
            "entry": asdict(entry),
        }
        # Add max_hsaco if specified
        if self.args.max_hsaco is not None:
            task_config["max_hsaco"] = {"*": self.args.max_hsaco}
        return task_config

    def queue_row(self, task_config: dict) -> dict:
        # 'class' names the DAG this task starts (perfmon rev2 R01/R02) --
        # dispatch_tasks.py only ever dispatches the tune_kernel DAG; a
        # --class flag to dispatch perf_measure tasks is future work
        # (rev2 Stage P), out of scope here.
        return {
            'arch': task_config['arch'],
            'module': task_config['module'],
            'class': 'tune_kernel',
            'subclass': task_config['tuning_level'],
            'task_config': task_config,
            'priority': 5  # Default priority
        }

def dispatch_tasks(workdir: Path, module_name: str, module_instance, args):
    """Dispatch tuning tasks to PostgreSQL queue."""
    from .dispatch import driver as _driver

    # Get database connection parameters
    conn_params = get_db_connection_params()

    print(f"Dispatching tasks to architecture(s): {', '.join(args.arch)}")

    # Get completed tasks if --skip_completed is enabled
    completed_configs = set()
    if args.skip_completed:
        print("Querying PostgreSQL for completed tasks...")
        # D09: the old tuning-mode flag is gone; --workload ('kernel'/'op')
        # already IS the subclass value for a tuning workload.
        completed_configs = get_completed_tasks(module_name, module_instance,
                                                  subklass=args.workload,
                                                  verbose=args.verbose)
        if args.dry_run:
            print(f"{len(completed_configs)=}")
            if completed_configs:
                print(f"Example: {next(iter(completed_configs))}")

    source = TuneEntrySource(module_name, module_instance, args)
    return _driver.dispatch(source=source, arch_list=args.arch,
                             conn_params=conn_params, args=args,
                             completed_configs=completed_configs)

def dispatch_perf_tasks(workdir: Path, module_name: str, module_instance, args):
    """Dispatch perf_measure tasks to PostgreSQL queue (D10).

    No `--skip_completed` support yet: there is no perf_measure analogue of
    `get_completed_tasks()` (D11's `pq/perf.py` only adds `save_perf_result`/
    `get_perf_results`/`iter_perf_rows_for_export` -- no completed-task
    query), so `--skip_completed` on a perfmon invocation is accepted
    (`add_common_arguments`, shared) but has no effect: `completed_configs`
    stays empty, same as tuning's own default before any completion query
    ran. Future work, not this task's.
    """
    from .dispatch import driver as _driver
    from .perfmon.source import PerfEntrySource
    from .perfmon.presets import available_presets

    conn_params = get_db_connection_params()

    # --preset SELECTS a subset of the configured presets; it does not
    # validate one (dispatch-perfmon.md §3.4) -- given values are used
    # as-is, never checked against available_presets().
    presets = args.preset if args.preset else available_presets(workdir)

    print(f"Dispatching tasks to architecture(s): {', '.join(args.arch)}")
    print(f"Presets: {', '.join(presets)}")

    source = PerfEntrySource(module_name, module_instance, args, presets=presets)

    if args.dry_run:
        print("perf_measure --dry_run: counts below are QUEUE ROWS (entries)"
              " per (arch, preset), not measurements. The number of "
              "MEASUREMENTS per entry is a per-arch, per-preset multiple "
              "that is NOT known at dispatch time -- it depends on the GPU "
              "and even differs per AOTriton version (dispatch-perfmon.md "
              "§6).")
        for preset in source.batches():
            for arch in args.arch:
                count = sum(1 for entry in source.entries(preset, arch)
                            if source.validate_hw_feature(arch, entry)[0])
                print(f"  {arch} / {preset}: {count} entries")

    return _driver.dispatch(source=source, arch_list=args.arch,
                             conn_params=conn_params, args=args,
                             completed_configs=None)

def str_to_bool(s):
    """Convert '0' or '1' string to boolean for argparse."""
    if s == '0':
        return False
    elif s == '1':
        return True
    else:
        raise argparse.ArgumentTypeError(f"Boolean value must be 0 or 1, got '{s}'")

def add_common_arguments(parser):
    """Add common arguments (arch, max_hsaco, etc.) to a parser.

    D09: exactly two removals from the pre-D09 version --
      * `workdir` (was a positional here) is now `--workdir` on dispatch's
        own Phase-1 parser (`build_base_parser()`);
      * the old tuning-mode flag is now `--workload` on that same Phase-1
        parser.
    Nothing else changes. `--max_hsaco` stays here even though it means
    nothing for a `perfmon` workload -- splitting common vs. tuning-only vs.
    perfmon-only flags is real work, deliberately deferred
    (dispatch-perfmon.md §13); leaving the wart visible is intentional, not
    an oversight.
    """
    parser.add_argument('--arch', type=str, nargs='+',
                        help='Target architecture(s). If not specified, uses all registered workers.')
    parser.add_argument('--max_hsaco', type=int, metavar='N',
                        help='Maximum number of hsaco kernels to tune per entry (default: all)')
    parser.add_argument('--skip_completed', action='store_true',
                        help='Query PostgreSQL and skip tasks that have already completed successfully')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print parsed options and exit without dispatching tasks')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt and proceed with dispatch')

def build_module_parser(module_name, module_instance):
    """Phase 2 (D09): the module's own parser, for everything after `--`.

    A standalone `ArgumentParser`, not a subparser -- Phase 1 no longer
    instantiates every registered module up front to build one subparser
    each (that eager-instantiation helper is deleted by this task); only
    the ONE module named by `--module`/`--workload` is ever loaded, so
    there is nothing left to attach subparsers to.

    The per-field dynamic-choice loop below is the old `add_module_
    subparser()`'s loop, unchanged in behavior -- only its host changed.
    It only runs when `module_instance` exposes `get_entry_choices()`
    (`TuningDescription` does; `PerfDescription` does not -- perfmon's
    entry space is a curated prime/coverage set, not a per-field cross
    product). A perfmon module instead gets D10's own three flags:
    `--preset`, `--entry_set`, `--max_seqlen` -- tuning has no use for any
    of them, mirroring how `--max_hsaco` (added by `add_common_arguments`,
    shared) has no use for perfmon; splitting the shared/tuning-only/
    perfmon-only flags apart is deliberately deferred (dispatch-perfmon.md
    §13).
    """
    module_parser = argparse.ArgumentParser(
        prog=f'dispatch_tasks --module .../{module_name}',
        usage='%(prog)s -- [options...]',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_common_arguments(module_parser)

    if not hasattr(module_instance, 'get_entry_choices'):
        # PerfDescription (D10): its own dispatch flags, not shared with
        # tuning's per-field cross product below.
        module_parser.add_argument(
            '--preset', type=str, nargs='+', default=None,
            help="Preset(s) ('rocm<ver>+aotriton<tag>') to dispatch, "
                 "repeatable. NARROWS the configured set for a partial "
                 "run -- selects, does not validate that a preset is real "
                 "or servable (dispatch-perfmon.md §3.4). Default: every "
                 "preset this fleet is configured to measure "
                 "(perfmon.presets.available_presets).")
        module_parser.add_argument(
            '--entry_set', choices=['prime', 'coverage'], default='prime',
            help="Which curated entry set to dispatch (default: %(default)s).")
        module_parser.add_argument(
            '--max_seqlen', type=int, default=None, metavar='N',
            help="Override the seqlen ceiling validate_hw_feature enforces "
                 "(default: no CLI override -- only each arch's own "
                 "measured ceiling, if any, applies).")
        return module_parser

    all_choices = get_parameter_choices(module_instance)

    # Add dynamic arguments based on module parameters
    for field in fields(all_choices):
        param_name = field.name
        param_choices = getattr(all_choices, param_name)

        # Determine type from first choice
        metavar = None
        display_choices = param_choices
        actual_choices = param_choices

        if param_choices:
            first_choice = param_choices[0]
            if isinstance(first_choice, bool):
                arg_type = str_to_bool
                actual_choices = [False, True]
                display_choices = [0, 1]
                metavar = '0/1'
            elif isinstance(first_choice, int):
                arg_type = int
            elif isinstance(first_choice, float):
                arg_type = float
            else:
                arg_type = str
        else:
            arg_type = str

        if metavar is None:
            metavar = arg_type.__name__.upper()

        module_parser.add_argument(f'--{param_name}',
                            type=arg_type,
                            nargs='*',
                            default=actual_choices,
                            choices=actual_choices,
                            metavar=metavar,
                            help=f'Choices: {display_choices}')

    return module_parser

def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split `argv` on the first literal `--`, positionally.

    Deliberately not `parse_known_args`: that would let an unrecognized
    flag typed BEFORE `--` silently fall through and get attributed to the
    module parser instead of erroring where the actual mistake is
    (dispatch-perfmon-exec.md D09)."""
    if '--' in argv:
        i = argv.index('--')
        return argv[:i], argv[i + 1:]
    return argv, []

def build_base_parser():
    """Phase 1 (D09): dispatch's own options -- `--workdir`, `--module`,
    `--workload`. Strict: anything before `--` that this parser does not
    recognize is an error here, never silently passed through."""
    from .pq.queue import WORKLOAD_TASK_SELECTOR

    base = argparse.ArgumentParser(
        prog='dispatch_tasks',
        description='Dispatch tasks to PostgreSQL queue (Tuner v3.5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Two-phase CLI: options BEFORE `--` are dispatch_tasks's own
(--workdir/--module/--workload, all required); options AFTER `--` belong to
the module named by --module (its own per-field choices for a kernel/op
workload; --arch/--max_hsaco/--skip_completed/--verbose/--dry_run/--yes for
any workload). `--help` (before `--`) prints this; `-- --help` prints the
module's own options.

Examples:
  # Dispatch all flash kernel-level tasks for gfx942
  %(prog)s --workdir /path/to/workdir --module modules/flash --workload kernel \\
      -- --arch gfx942

  # Dispatch only float16 tasks with specific sequence lengths
  %(prog)s --workdir /path/to/workdir --module modules/flash --workload kernel \\
      -- --arch gfx942 --dtype float16 --seqlen_q 128 256 --seqlen_k 128 256

  # Dispatch to multiple architectures
  %(prog)s --workdir /path/to/workdir --module modules/flash --workload kernel \\
      -- --arch gfx942 gfx90a

  # Limit number of hsaco kernels to tune per entry
  %(prog)s --workdir /path/to/workdir --module modules/flash --workload kernel \\
      -- --max_hsaco 5 --dtype float16

  # Skip tasks that have already completed (queries PostgreSQL)
  %(prog)s --workdir /path/to/workdir --module modules/flash --workload kernel \\
      -- --skip_completed --arch gfx942
''')
    base.add_argument('--workdir', type=Path, required=True,
                       help='Project working directory')
    base.add_argument('--module', type=Path, required=True,
                       help='PATH to the module directory, e.g. modules/flash '
                            '(a directory containing a tune/ or perfmon/ subdirectory)')
    base.add_argument('--workload', required=True, choices=list(WORKLOAD_TASK_SELECTOR),
                       help="'kernel'/'op' dispatch the tune_kernel DAG; "
                            "'perfmon' dispatches the perf_measure DAG")
    return base

def resolve_module(base: argparse.ArgumentParser, args):
    """Resolve `--module`/`--workload` to `(family, module_instance)` --
    from a PATH, never a name looked up in a registry (dispatch-perfmon.md
    §4): the path-driven `load_family_tune`/`load_family_perfmon` loaders,
    NOT the name-and-registry-driven alias `registry.py` also exports for
    the static `_FAMILIES` list -- a path is its own identity and needs no
    pre-registration there."""
    from .registry import load_family_tune, load_family_perfmon

    module_path = args.module.resolve()
    if not (module_path / 'tune').is_dir() and not (module_path / 'perfmon').is_dir():
        base.error(f"{module_path} is not a module directory "
                   f"(no tune/ or perfmon/ under it)")

    family = module_path.name          # e.g. 'flash'
    modules_dir = module_path.parent   # e.g. .../modules

    if args.workload == 'perfmon':
        mod = load_family_perfmon(family, modules_dir=modules_dir)
        attr = 'PerfDesc'
    else:
        mod = load_family_tune(family, modules_dir=modules_dir)
        attr = 'TuneDesc'

    if not hasattr(mod, attr):
        base.error(f"Module '{family}' has no class '{attr}'")
    return family, getattr(mod, attr)()

def parse_cli(argv: list[str]):
    """The full two-phase parse (D09), factored out of `main()` so it can
    be exercised without a live database -- everything after this point in
    `main()` needs PostgreSQL (config.rc, workers.db, the dispatch itself),
    but the parse does not.

    Returns `(args, family, module_instance)`. `args` carries the Phase-1
    fields (`workdir`/`module`/`workload`) and the Phase-2 fields
    (`arch`/`max_hsaco`/... and, for a tuning module, its per-field entry
    filters) merged onto one `argparse.Namespace`, exactly as `dispatch_
    tasks()`/`generate_filtered_entries()` expect (unchanged from before
    D09 in this respect).
    """
    head, tail = _split_argv(argv)

    base = build_base_parser()
    args = base.parse_args(head)

    family, module_instance = resolve_module(base, args)

    module_parser = build_module_parser(family, module_instance)
    mod_args = module_parser.parse_args(tail)
    for key, value in vars(mod_args).items():
        setattr(args, key, value)

    return args, family, module_instance

def main():
    args, family, module_instance = parse_cli(sys.argv[1:])

    # Validate workdir
    if not args.workdir.is_dir():
        sys.exit(f"Error: Working directory does not exist: {args.workdir}")

    # Load config (required for task dispatch)
    load_config(args.workdir)

    # If arch not specified, use all registered architectures
    if args.arch is None:
        args.arch = get_registered_archs(args.workdir)
        if not args.arch:
            sys.exit(f"Error: No workers registered in {args.workdir}/workers.db")
        print(f"No --arch specified, using all registered: {', '.join(args.arch)}")

    if args.workload == 'perfmon':
        # D10: PerfEntrySource wires up perf_measure dispatch.
        dispatch_perf_tasks(args.workdir, family, module_instance, args)
    else:
        dispatch_tasks(args.workdir, family, module_instance, args)

if __name__ == '__main__':
    main()
