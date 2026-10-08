"""Storing the metadata catalog in the vector database.

Kept as history, one row per extraction: when a rule stops validating, the question is usually "what
changed in the source schema and when", and a digest per run answers it. Only the latest run is read
back.
"""
from __future__ import annotations

import logging

import psycopg
from psycopg.types.json import Jsonb

from crawlerrag.db import scalar
from crawlerrag.meta.models import Catalog, Column, Relationship, Table

log = logging.getLogger(__name__)


def save_catalog(conn: psycopg.Connection, catalog: Catalog, *, source: str = "cli") -> int:
    with conn.transaction():
        run_id = int(scalar(conn, """
            INSERT INTO meta.catalog_run (captured_at, source, digest, table_count, column_count,
                                          relationship_count)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING run_id
        """, (catalog.captured_at, source, catalog.digest, catalog.table_count, catalog.column_count,
              catalog.relationship_count)))
        with conn.cursor() as cur:
            with cur.copy("COPY meta.table_info (run_id, schema_name, table_name, kind, primary_key, "
                          "unique_keys, est_rows, comment) FROM STDIN") as copy:
                for table in catalog.tables:
                    copy.write_row((run_id, table.schema, table.name, table.kind,
                                    list(table.primary_key), Jsonb([list(k) for k in table.unique_keys]),
                                    table.est_rows, table.comment))
            with cur.copy("COPY meta.column_info (run_id, schema_name, table_name, column_name, ordinal, "
                          "data_type, is_nullable, has_default, max_length, precision, scale, comment) "
                          "FROM STDIN") as copy:
                for table in catalog.tables:
                    for column in table.columns:
                        copy.write_row((run_id, table.schema, table.name, column.name, column.ordinal,
                                        column.data_type, column.is_nullable, column.has_default,
                                        column.max_length, column.precision, column.scale,
                                        column.comment))
            with cur.copy("COPY meta.relationship (run_id, from_table, from_columns, to_table, to_columns, "
                          "kind, constraint_name) FROM STDIN") as copy:
                for rel in catalog.relationships:
                    copy.write_row((run_id, rel.from_table, list(rel.from_columns), rel.to_table,
                                    list(rel.to_columns), rel.kind, rel.constraint))
    log.info("metadata catalog stored", extra={"run_id": run_id, "digest": catalog.digest[:12],
                                               "tables": catalog.table_count})
    return run_id


def latest_run(conn: psycopg.Connection) -> dict | None:
    return conn.execute("SELECT * FROM meta.catalog_run ORDER BY run_id DESC LIMIT 1").fetchone()


def load_catalog(conn: psycopg.Connection) -> Catalog | None:
    """The most recent stored catalog, or None when nothing has been extracted yet."""
    run = latest_run(conn)
    if run is None:
        return None
    columns: dict[tuple[str, str], list[Column]] = {}
    for row in conn.execute(
            "SELECT * FROM meta.column_info WHERE run_id = %s ORDER BY schema_name, table_name, ordinal",
            (run["run_id"],)).fetchall():
        key = (row["schema_name"], row["table_name"])
        columns.setdefault(key, []).append(Column(
            schema=key[0], table=key[1], name=row["column_name"], ordinal=row["ordinal"],
            data_type=row["data_type"], is_nullable=row["is_nullable"], has_default=row["has_default"],
            max_length=row["max_length"], precision=row["precision"], scale=row["scale"],
            comment=row["comment"]))
    tables = []
    for row in conn.execute("SELECT * FROM meta.table_info WHERE run_id = %s ORDER BY schema_name, table_name",
                            (run["run_id"],)).fetchall():
        key = (row["schema_name"], row["table_name"])
        tables.append(Table(schema=key[0], name=key[1], kind=row["kind"],
                            columns=tuple(columns.get(key, ())),
                            primary_key=tuple(row["primary_key"] or ()),
                            unique_keys=tuple(tuple(k) for k in row["unique_keys"] or ()),
                            est_rows=row["est_rows"], comment=row["comment"]))
    relationships = [Relationship(from_table=row["from_table"], from_columns=tuple(row["from_columns"]),
                                  to_table=row["to_table"], to_columns=tuple(row["to_columns"]),
                                  kind=row["kind"], constraint=row["constraint_name"])
                     for row in conn.execute("SELECT * FROM meta.relationship WHERE run_id = %s "
                                             "ORDER BY from_table, to_table", (run["run_id"],)).fetchall()]
    return Catalog(captured_at=run["captured_at"], tables=tuple(tables),
                   relationships=tuple(relationships))
