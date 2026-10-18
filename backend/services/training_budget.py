"""Bounded fitting capacity; optional PostgreSQL coordination across service workers."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from config import settings


class TrainingBusyError(RuntimeError):
    """A retryable capacity or coordination failure, never a prediction fallback."""


_lock = threading.Lock()
_budgets: dict[int, threading.BoundedSemaphore] = {}
# Session-level advisory lock: held only while fitting, no tables or migrations.
_PG_LOCK_ID = 73716720261002


@contextmanager
def training_slot():
    capacity = settings.forecast_training_max_concurrency
    with _lock:
        budget = _budgets.setdefault(capacity, threading.BoundedSemaphore(capacity))
    if not budget.acquire(blocking=False):
        raise TrainingBusyError("Forecast training is busy. Please retry shortly.")
    connection = None
    try:
        url = settings.forecast_training_coordination_database_url
        if url:
            import psycopg

            try:
                connection = psycopg.connect(url, connect_timeout=2, autocommit=True)
                with connection.cursor() as cursor:
                    cursor.execute("SET statement_timeout = '2000ms'")
                    cursor.execute("SELECT pg_try_advisory_lock(%s)", (_PG_LOCK_ID,))
                    acquired = cursor.fetchone()[0]
            except Exception as error:
                raise TrainingBusyError(
                    "Forecast coordination is unavailable. Please retry shortly."
                ) from error
            if not acquired:
                raise TrainingBusyError("Forecast training is busy. Please retry shortly.")
        yield
    finally:
        # Closing the session releases its advisory lock even after an exception.
        try:
            if connection is not None:
                connection.close()
        finally:
            budget.release()
