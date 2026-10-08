"""PostgreSQL helpers (psycopg 3)."""
from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import psycopg
from psycopg.rows import dict_row

log = logging.getLogger(__name__)


def connect(database_url: str, application_name: str = "crawler-rag") -> psycopg.Connection:
    """Autocommit connection; atomic units use explicit ``conn.transaction()`` blocks."""
    return psycopg.connect(database_url, autocommit=True, row_factory=dict_row,
                           application_name=application_name, connect_timeout=15)


def wait_for_db(database_url: str, timeout_s: float = 90.0, application_name: str = "crawler-rag") -> psycopg.Connection:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return connect(database_url, application_name)
        except psycopg.OperationalError as exc:
            if time.monotonic() > deadline:
                raise
            log.warning("database not ready, retrying", extra={"error": str(exc).strip()})
            time.sleep(2)


def scalar(conn: psycopg.Connection, sql: str, params: Sequence[Any] | dict | None = None) -> Any:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    return next(iter(row.values()))
