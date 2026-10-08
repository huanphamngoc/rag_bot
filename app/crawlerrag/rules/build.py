"""From a rule to a SELECT, and from a source row to a document.

The generated statement keeps the shape the hand-written ones had: ``record_key`` first, then the
parent columns some value spec actually references, then one correlated ``array_agg`` per child
collection, then a ``{where}`` placeholder that the caller fills with ``true`` (full load) or a key
list (incremental).

Identifiers are quoted and child filter values are bound parameters. The rule files are trusted
configuration, but generated SQL that interpolates values is a habit that outlives the trust.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import psycopg

from crawlerrag.model import Document
from crawlerrag.rules.models import ChildSpec, DocTypeRule


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


# ---------------------------------------------------------------- the document
def build_document(rule: DocTypeRule, row: Mapping[str, Any]) -> Document:
    body_lines = []
    for line in rule.body:
        rendered = line.value.text(row)
        if rendered:
            body_lines.append(f"{line.label}: {rendered}")
    metadata: dict[str, Any] = {}
    for name, spec in rule.metadata.items():
        value = spec.value(row)
        if isinstance(value, (list, tuple)):
            value = [v for v in value if v is not None]
        if value is None or value == [] or value == "":
            continue
        metadata[name] = value.isoformat() if hasattr(value, "isoformat") else value
    return Document(
        doc_id=rule.doc_id(str(row["record_key"])),
        doc_type=rule.doc_type,
        source_id=rule.source.source_id,
        title=rule.title.text(row),
        body="\n".join(body_lines),
        url=(rule.url.text(row) or None) if rule.url is not None else None,
        metadata=metadata,
        is_active=bool(row[rule.source.active_column]),
    )


# ---------------------------------------------------------------- the statement
def referenced_columns(rule: DocTypeRule) -> set[str]:
    """Parent columns the rule reads, plus the two it always needs."""
    wanted = {rule.source.key, rule.source.active_column}
    if rule.source.order_by:
        wanted.add(rule.source.order_by)
    for _, spec in rule.value_specs():
        wanted |= spec.columns()
    return wanted - set(rule.children)


def _child_sql(alias: str, child: ChildSpec, parent_key_alias: str, params: dict[str, Any]) -> str:
    selected = f'c.{quote(child.select)}'
    if child.distinct:
        # array_agg(DISTINCT x) already orders by x, which is why a DISTINCT child needs no order_by.
        aggregate = f"array_agg(DISTINCT {selected})"
    elif child.order_by:
        aggregate = f"array_agg({selected} ORDER BY c.{quote(child.order_by)})"
    else:
        aggregate = f"array_agg({selected})"
    conditions = [f'c.{quote(child_col)} = {parent_key_alias}.{quote(parent_col)}'
                  for child_col, parent_col in child.join.items()]
    if child.where is not None:
        name = f"child_{alias}_{child.where.column}"
        operator = "=" if child.where.eq is not None else "<>"
        params[name] = child.where.eq if child.where.eq is not None else child.where.ne
        conditions.append(f'c.{quote(child.where.column)} {operator} %({name})s')
    return (f"       (SELECT {aggregate} FILTER (WHERE {selected} IS NOT NULL)\n"
            f"          FROM {quote(child.db_schema)}.{quote(child.table)} c\n"
            f"         WHERE {' AND '.join(conditions)}) AS {alias}")


def build_sql(rule: DocTypeRule) -> tuple[str, dict[str, Any]]:
    """The SELECT for one document type, with a ``{where}`` placeholder, and its static parameters."""
    params: dict[str, Any] = {}
    key = quote(rule.source.key)
    columns = [f"p.{key} AS record_key"]
    columns += [f"p.{quote(name)}" for name in sorted(referenced_columns(rule))]
    columns += [_child_sql(alias, child, "p", params) for alias, child in rule.children.items()]
    order = rule.source.order_by or rule.source.key
    selected = ",\n       ".join(columns).lstrip()
    sql = (f"SELECT {selected}\n"
           f"  FROM {quote(rule.source.db_schema)}.{quote(rule.source.table)} p\n"
           f" WHERE {{where}}\n"
           f" ORDER BY p.{quote(order)}")
    return sql, params


def key_predicate(rule: DocTypeRule) -> str:
    return f'p.{quote(rule.source.key)} = ANY(%(keys)s)'


def fetch_rows(conn: psycopg.Connection, rule: DocTypeRule,
               keys: Sequence[str] | None) -> list[dict]:
    """Every row (``keys=None``) or the rows of these record keys, in the caller's snapshot."""
    sql, params = build_sql(rule)
    if keys is None:
        return conn.execute(sql.format(where="true"), params).fetchall()
    if not keys:
        return []
    params["keys"] = list(keys)
    return conn.execute(sql.format(where=key_predicate(rule)), params).fetchall()
