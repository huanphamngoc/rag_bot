"""Minimal, checksum-verified SQL migration runner (same runner as the crawler).

Files named ``V<NNN>__<description>.sql`` in ``crawlerrag/migrations`` are
applied in order exactly once; an edited, already-applied file is an error.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import psycopg

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class MigrationError(RuntimeError):
    pass


def migrate(conn: psycopg.Connection) -> list[str]:
    applied_now: list[str] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext('crawlerrag.migrate'))")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS public.schema_migration (
                version     text PRIMARY KEY,
                filename    text        NOT NULL,
                checksum    char(64)    NOT NULL,
                applied_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        applied = {r["version"]: r for r in conn.execute("SELECT version, filename, checksum FROM public.schema_migration")}
        for path in sorted(MIGRATIONS_DIR.glob("V*__*.sql")):
            version = path.name.split("__", 1)[0]
            sql = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            if version in applied:
                if applied[version]["checksum"] != checksum:
                    raise MigrationError(
                        f"Migration {path.name} was modified after being applied "
                        f"(db={applied[version]['checksum'][:12]}, file={checksum[:12]}). "
                        "Create a new migration instead of editing an applied one."
                    )
                continue
            log.info("applying migration", extra={"migration": path.name})
            conn.execute(sql)
            conn.execute(
                "INSERT INTO public.schema_migration (version, filename, checksum) VALUES (%s, %s, %s)",
                (version, path.name, checksum),
            )
            applied_now.append(path.name)
    if applied_now:
        log.info("migrations applied", extra={"count": len(applied_now)})
    return applied_now
