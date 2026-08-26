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

def load_module(module_name: str):
    """Load tuning module and return the module class instance."""
    from .registry import load_tune_module
    try:
        mod = load_tune_module(module_name)
    except ImportError as e:
        sys.exit(f"Error: Failed to import module '{module_name}': {e}")
    # Module __init__.py should export TuneDesc as the main class
    if not hasattr(mod, 'TuneDesc'):
        sys.exit(f"Error: Module '{module_name}' has no class 'TuneDesc'")
    module_class = getattr(mod, 'TuneDesc')
    return module_class()

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

def get_completed_tasks(module_name: str, module_instance, tuning_mode: str, verbose: bool = False):
    """
    Query PostgreSQL for completed tasks from task_queue.

    Returns a set of task_config tuples (hashable form) that have
    successfully completed (status = 'completed').

    Args:
        module_name: Name of the tuning module (e.g., 'flash')
        module_instance: Module instance with ENTRY_CLASS defining field structure
        tuning_mode: 'kernel' or 'op' -- filters on the denormalized
            task_queue.subclass column so kernel-level and op-level
            completed tasks (which may share the same module name and entry
            fields) are never conflated.
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
            module_name, klass='tune_kernel', subklass=tuning_mode)

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
            "tuning_level": self.args.tuning_mode,
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
        completed_configs = get_completed_tasks(module_name, module_instance,
                                                  tuning_mode=args.tuning_mode,
                                                  verbose=args.verbose)
        if args.dry_run:
            print(f"{len(completed_configs)=}")
            if completed_configs:
                print(f"Example: {next(iter(completed_configs))}")

    source = TuneEntrySource(module_name, module_instance, args)
    return _driver.dispatch(source=source, arch_list=args.arch,
                             conn_params=conn_params, args=args,
                             completed_configs=completed_configs)

def str_to_bool(s):
    """Convert '0' or '1' string to boolean for argparse."""
    if s == '0':
        return False
    elif s == '1':
        return True
    else:
        raise argparse.ArgumentTypeError(f"Boolean value must be 0 or 1, got '{s}'")

def get_available_modules():
    """
    Return a dict mapping each registered tuning module name (see
    .registry._MODULE_TO_FAMILY) to an instantiated TuneDesc object. The
    module list is now a static registry (F8), not a directory glob rooted
    here -- 'flash'/'flash_op' moved to modules/flash/tune/, outside this
    package, so a glob rooted at this file can no longer discover them.
    """
    from .registry import available_module_names, load_tune_module
    modules = {}
    for name in available_module_names():
        mod = load_tune_module(name)
        if hasattr(mod, 'TuneDesc'):
            modules[name] = mod.TuneDesc()
    return modules

def add_common_arguments(parser):
    """Add common arguments (workdir, arch, etc.) to a parser."""
    parser.add_argument('workdir', type=Path,
                        help='Project working directory')
    parser.add_argument('--tuning_mode', type=str, default='kernel', choices=['kernel', 'op'],
                        help='Tuning level to dispatch tasks for: kernel (default) or op')
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

def add_module_subparser(subparsers, module_name, module_instance):
    """
    Add a subparser for a specific tuning module with its parameter choices.

    Args:
        subparsers: The subparsers object from ArgumentParser.add_subparsers()
        module_name: Name of the tuning module (e.g., 'flash')
        module_instance: Already-instantiated TuneDesc object for this module

    Returns:
        The created subparser
    """
    module_parser = subparsers.add_parser(
        module_name,
        help=f'{module_name.capitalize()} tuning module',
        usage=f'%(prog)s <workdir> [options...]',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_common_arguments(module_parser)

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

def main():
    parser = argparse.ArgumentParser(
        description='Dispatch tuning tasks to PostgreSQL queue (Tuner v3.5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dispatch all flash tasks for gfx942
  %(prog)s flash /path/to/workdir --arch gfx942

  # Dispatch only float16 tasks with specific sequence lengths
  %(prog)s flash /path/to/workdir --arch gfx942 --dtype float16 --seqlen_q 128 256 --seqlen_k 128 256

  # Dispatch to multiple architectures
  %(prog)s flash /path/to/workdir --arch gfx942 gfx90a

  # Limit number of hsaco kernels to tune per entry
  %(prog)s flash /path/to/workdir --max_hsaco 5 --dtype float16

  # Skip tasks that have already completed (queries PostgreSQL)
  %(prog)s flash /path/to/workdir --skip_completed --arch gfx942
''')

    # Create subparsers for each module BEFORE parsing
    # Module comes first as a positional argument
    subparsers = parser.add_subparsers(dest='module', required=True,
                                       help='Tuning module')

    available_modules = get_available_modules()
    for module_name, module_instance in available_modules.items():
        add_module_subparser(subparsers, module_name, module_instance)

    # Parse all arguments
    args = parser.parse_args()

    # Validate workdir
    if not args.workdir.is_dir():
        parser.error(f"Working directory does not exist: {args.workdir}")

    # Load config (required for task dispatch)
    load_config(args.workdir)

    # If arch not specified, use all registered architectures
    if args.arch is None:
        args.arch = get_registered_archs(args.workdir)
        if not args.arch:
            parser.error(f"No workers registered in {args.workdir}/workers.db")
        print(f"No --arch specified, using all registered: {', '.join(args.arch)}")

    # Dispatch tasks
    dispatch_tasks(args.workdir, args.module, available_modules[args.module], args)

if __name__ == '__main__':
    main()
