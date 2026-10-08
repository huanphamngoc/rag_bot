"""Reading table, column, key and relationship information out of Postgres, read-only.

Everything comes from ``pg_catalog`` rather than ``information_schema``, for one concrete reason:
``format_type()`` prints what a person would write - ``text[]``, ``numeric(10,2)``, ``timestamptz`` -
while ``information_schema.columns.data_type`` says ``ARRAY`` and hides the element type in
``udt_name``. Numeric precision and length come from ``information_schema`` because they are plainer
there.

Relationships are read **and** derived. The crawler's own database declares them properly - 25 foreign
keys on 2026-10-05, including every recall child table - but a schema restored without constraints, or
a view, declares none, and the child's primary key still starts with the parent's. So both are
collected and each relationship records which it is: ``declared`` or ``inferred``. Validation accepts
either, and says so when it found neither.
"""
from __future__ import annotations

import datetime as dt
import fnmatch
import logging
from typing import Iterable, Sequence

import psycopg

from crawlerrag.meta.models import Catalog, Column, Relationship, Table

log = logging.getLogger(__name__)

TABLES_SQL = """
SELECT n.nspname AS schema_name, c.relname AS table_name,
       CASE c.relkind WHEN 'r' THEN 'table' WHEN 'p' THEN 'partitioned table' WHEN 'v' THEN 'view'
            WHEN 'm' THEN 'materialized view' WHEN 'f' THEN 'foreign table' END AS kind,
       nullif(c.reltuples, -1)::bigint AS est_rows,
       obj_description(c.oid, 'pg_class') AS comment
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE n.nspname = ANY(%(schemas)s) AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
 ORDER BY 1, 2
"""

COLUMNS_SQL = """
SELECT n.nspname AS schema_name, c.relname AS table_name, a.attname AS column_name,
       a.attnum AS ordinal, format_type(a.atttypid, a.atttypmod) AS data_type,
       NOT a.attnotnull AS is_nullable, a.atthasdef AS has_default,
       col_description(c.oid, a.attnum) AS comment
  FROM pg_attribute a
  JOIN pg_class c ON c.oid = a.attrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE n.nspname = ANY(%(schemas)s) AND a.attnum > 0 AND NOT a.attisdropped
   AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
 ORDER BY 1, 2, 4
"""

SIZES_SQL = """
SELECT table_schema AS schema_name, table_name, column_name,
       character_maximum_length AS max_length, numeric_precision AS precision, numeric_scale AS scale
  FROM information_schema.columns
 WHERE table_schema = ANY(%(schemas)s)
"""

KEYS_SQL = """
SELECT n.nspname AS schema_name, c.relname AS table_name, i.indisprimary,
       (SELECT array_agg(a.attname ORDER BY k.ord)
          FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = k.attnum) AS columns
  FROM pg_index i
  JOIN pg_class c ON c.oid = i.indrelid
  JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE n.nspname = ANY(%(schemas)s) AND i.indisunique AND i.indpred IS NULL AND i.indexprs IS NULL
 ORDER BY 1, 2, i.indisprimary DESC
"""

FOREIGN_KEYS_SQL = """
SELECT con.conname AS constraint_name,
       fn.nspname || '.' || f.relname AS from_table,
       tn.nspname || '.' || t.relname AS to_table,
       (SELECT array_agg(a.attname ORDER BY k.ord)
          FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum) AS from_columns,
       (SELECT array_agg(a.attname ORDER BY k.ord)
          FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum) AS to_columns
  FROM pg_constraint con
  JOIN pg_class f ON f.oid = con.conrelid
  JOIN pg_namespace fn ON fn.oid = f.relnamespace
  JOIN pg_class t ON t.oid = con.confrelid
  JOIN pg_namespace tn ON tn.oid = t.relnamespace
 WHERE con.contype = 'f' AND fn.nspname = ANY(%(schemas)s)
 ORDER BY 1
"""


def _excluded(qualified_name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(qualified_name, pattern) for pattern in patterns)


def read_catalog(conn: psycopg.Connection, schemas: Sequence[str],
                 exclude: Sequence[str] = ()) -> Catalog:
    """One snapshot of the schemas named, through whatever (read-only) role the connection holds."""
    params = {"schemas": list(schemas)}
    sizes = {(r["schema_name"], r["table_name"], r["column_name"]): r
             for r in conn.execute(SIZES_SQL, params).fetchall()}
    columns: dict[tuple[str, str], list[Column]] = {}
    for row in conn.execute(COLUMNS_SQL, params).fetchall():
        key = (row["schema_name"], row["table_name"])
        extra = sizes.get((*key, row["column_name"]), {})
        columns.setdefault(key, []).append(Column(
            schema=row["schema_name"], table=row["table_name"], name=row["column_name"],
            ordinal=row["ordinal"], data_type=row["data_type"], is_nullable=row["is_nullable"],
            has_default=row["has_default"], comment=row["comment"],
            max_length=extra.get("max_length"), precision=extra.get("precision"),
            scale=extra.get("scale")))

    primary: dict[tuple[str, str], tuple[str, ...]] = {}
    unique: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for row in conn.execute(KEYS_SQL, params).fetchall():
        key = (row["schema_name"], row["table_name"])
        cols = tuple(row["columns"] or ())
        if row["indisprimary"]:
            primary[key] = cols
        else:
            unique.setdefault(key, []).append(cols)

    tables = []
    for row in conn.execute(TABLES_SQL, params).fetchall():
        key = (row["schema_name"], row["table_name"])
        qualified = f"{key[0]}.{key[1]}"
        if _excluded(qualified, exclude):
            continue
        tables.append(Table(schema=key[0], name=key[1], kind=row["kind"],
                            columns=tuple(columns.get(key, ())),
                            primary_key=primary.get(key, ()),
                            unique_keys=tuple(unique.get(key, ())),
                            est_rows=row["est_rows"], comment=row["comment"]))

    known = {t.qualified_name for t in tables}
    declared = [Relationship(from_table=row["from_table"], from_columns=tuple(row["from_columns"] or ()),
                             to_table=row["to_table"], to_columns=tuple(row["to_columns"] or ()),
                             kind="declared", constraint=row["constraint_name"])
                for row in conn.execute(FOREIGN_KEYS_SQL, params).fetchall()
                if row["from_table"] in known and row["to_table"] in known]

    catalog = Catalog(captured_at=dt.datetime.now(dt.timezone.utc), tables=tuple(tables),
                      relationships=tuple(infer_relationships(tables, declared=declared)))
    log.info("metadata catalog read", extra={"schemas": list(schemas), "tables": catalog.table_count,
                                             "columns": catalog.column_count,
                                             "relationships": catalog.relationship_count,
                                             "digest": catalog.digest[:12]})
    return catalog


def infer_relationships(tables: Iterable[Table],
                        declared: Iterable[Relationship] = ()) -> list[Relationship]:
    """Declared foreign keys, plus the parent/child links the crawler's schema only implies.

    A table is treated as a child of another when its primary key *starts with* the parent's whole
    primary key and is longer - exactly the shape the crawler uses for its collection tables
    (``PRIMARY KEY (recall_id, seq)`` under ``PRIMARY KEY (recall_id)``). Equal keys are not a child
    relationship: that is a side table with one row per parent row, not a collection.
    """
    tables = list(tables)
    out = list(declared)
    pairs = {(r.from_table, r.to_table) for r in out}
    for parent in tables:
        if not parent.primary_key:
            continue
        width = len(parent.primary_key)
        for child in tables:
            if child.qualified_name == parent.qualified_name:
                continue
            if len(child.primary_key) <= width:
                continue
            if child.primary_key[:width] != parent.primary_key:
                continue
            if (child.qualified_name, parent.qualified_name) in pairs:
                continue
            pairs.add((child.qualified_name, parent.qualified_name))
            out.append(Relationship(from_table=child.qualified_name, from_columns=parent.primary_key,
                                    to_table=parent.qualified_name, to_columns=parent.primary_key,
                                    kind="inferred", constraint=None))
    return out
