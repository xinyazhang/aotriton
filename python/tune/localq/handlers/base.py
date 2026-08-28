# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
DAG-neutral message handling: the handler base class, and the handlers that
apply to any DAG.

`pq` + `localq` + `exaid` is a general distributed execution framework -- a
queue, a DAG runner, and crash-isolated CLI runners. Nothing in this module
knows which DAG it is serving: graceful cancellation and failure marking are
the same operation whether the task was tuning a kernel or measuring
performance.

Everything that DOES know a specific DAG lives in a sibling module named for
that DAG's task_queue.class value (perfmon rev2 R03).
"""

import logging
import shutil
from pathlib import Path
from typing import List

from ...pq.queue import TaskQueue

logger = logging.getLogger(__name__)


class MessageHandler:
    """Base class for message handlers"""

    @classmethod
    def get_class_name(cls) -> str:
        """Get message class this handler processes"""
        raise NotImplementedError

    def handle(self, message: dict) -> dict | List[dict] | None:
        """
        Process message and return result message(s) (or None).

        Result message is automatically forwarded to its target_queue.

        Args:
            message: Input message

        Returns:
            Result message, list of result messages, or None
        """
        raise NotImplementedError

    def resolve_dependency(self, blocked_msg: dict, incoming_msg: dict) -> bool:
        """
        Called when incoming_msg arrives that might resolve blocked_msg's dependency.

        Args:
            blocked_msg: Message waiting for dependencies
            incoming_msg: Newly arrived message

        Returns:
            True if dependency is resolved (unblock message)
        """
        return False

    def teardown_with_unmet_dependency(self, message: dict) -> dict | None:
        """
        Called during graceful shutdown when message has unmet dependencies.

        Default implementation returns None (no action needed).
        Override in subclasses if teardown requires specific actions.

        Args:
            message: Blocked message being torn down

        Returns:
            Result message to enqueue (or None)
        """
        return None


class GracefulCancelRunningTaskHandler(MessageHandler):
    """
    Moves task state back to pending when gracefully cancelled.

    This handler is used during graceful shutdown to cancel running tasks
    that have unmet dependencies (incomplete tune_hsaco work).
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "graceful_cancel_running_task"

    def handle(self, message: dict) -> None:
        task_id = message['task_id']
        arch = message['arch']

        logger.info(f"Gracefully cancelling task_id={task_id}, moving back to pending")

        # Move task back to pending state
        task_queue = TaskQueue(self.db_conn)
        task_queue.mark_pending(task_id, arch)

        logger.info(f"Task {task_id} moved back to pending state")

        # No result message
        return None


class MarkTaskFailedHandler(MessageHandler):
    """
    Marks task as failed in database.

    This handler is used when GPU workers encounter exceptions during
    preprocess or probe stages. GPU workers don't have DB access, so they
    send this message to CPU workers to write the failure to the database.
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "mark_task_failed"

    def handle(self, message: dict) -> dict:
        task_id = message['task_id']
        arch = message['arch']
        error = message['error']

        logger.info(f"Marking task_id={task_id} as failed: {error}")

        # Mark task as failed in database
        task_queue = TaskQueue(self.db_conn)
        task_queue.mark_failed(task_id, arch=arch, error_message=error)

        logger.info(f"Task {task_id} marked as failed in database")

        # Remove prepared data from tmpfs to free space
        tmpdir = message.get('tmpdir')
        if tmpdir:
            tmpdir_path = Path(tmpdir)
            if tmpdir_path.exists():
                try:
                    shutil.rmtree(tmpdir_path)
                    logger.info(f"Removed tmpdir {tmpdir_path} for failed task {task_id}")
                except OSError as e:
                    logger.warning(f"Failed to remove tmpdir {tmpdir_path}: {e}")

        # Return nak (negative ack) message to unblock PG reader
        return {
            'class': 'dag_ack',
            'task_id': task_id,
            'negative': True
        }
