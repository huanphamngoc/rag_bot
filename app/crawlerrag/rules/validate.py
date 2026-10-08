"""Checking the rule files against the metadata extracted from Postgres.

A rule names tables and columns of a database this project only reads, so the names are checked before
a row is read: a typo is an error, and the run stops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from crawlerrag.meta.models import Catalog, Table
from crawlerrag.rules.loader import RuleSet
from crawlerrag.rules.models import DocTypeRule

BOOLEAN_TYPES = ("boolean", "bool")


@dataclass(frozen=True)
class Finding:
    level: str                      # "error" | "warning"
    doc_type: str | None
    message: str

    def render(self) -> str:
        where = f"{self.doc_type}: " if self.doc_type else ""
        return f"[{self.level}] {where}{self.message}"


def errors(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "error"]


def warnings(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "warning"]


def _columns(table: Table) -> set[str]:
    return {c.name for c in table.columns}


def validate_doc_type(rule: DocTypeRule, catalog: Catalog) -> list[Finding]:
    found: list[Finding] = []

    def error(message: str) -> None:
        found.append(Finding(level="error", doc_type=rule.doc_type, message=message))

    def warn(message: str) -> None:
        found.append(Finding(level="warning", doc_type=rule.doc_type, message=message))

    parent = catalog.table(rule.source.qualified_name)
    if parent is None:
        error(f"source table {rule.source.qualified_name} is not in the catalog")
        return found                       # nothing else can be checked without it
    parent_columns = _columns(parent)

    # ---- the key the whole incremental strategy rests on
    if rule.source.key not in parent_columns:
        error(f"key column {rule.source.key} is not in {parent.qualified_name}")
    elif not parent.is_unique((rule.source.key,)):
        error(f"key column {rule.source.key} is not a unique key of {parent.qualified_name} "
              f"(primary key: {', '.join(parent.primary_key) or 'none'}); record_key must identify "
              "exactly one row")

    # ---- the soft-delete flag
    active = parent.column(rule.source.active_column)
    if active is None:
        error(f"active_column {rule.source.active_column} is not in {parent.qualified_name}")
    elif active.data_type not in BOOLEAN_TYPES:
        error(f"active_column {rule.source.active_column} is {active.data_type}, expected boolean")

    if rule.source.order_by and rule.source.order_by not in parent_columns:
        error(f"order_by column {rule.source.order_by} is not in {parent.qualified_name}")

    # ---- child collections
    for alias, child in rule.children.items():
        if alias in parent_columns:
            error(f"child name {alias} collides with a column of {parent.qualified_name}; "
                  "both would land in the same row")
        table = catalog.table(child.qualified_name)
        if table is None:
            error(f"child {alias}: table {child.qualified_name} is not in the catalog")
            continue
        child_columns = _columns(table)
        for column, what in ((child.select, "select"), (child.order_by, "order_by")):
            if column and column not in child_columns:
                error(f"child {alias}: {what} column {column} is not in {table.qualified_name}")
        if child.where is not None and child.where.column not in child_columns:
            error(f"child {alias}: filter column {child.where.column} is not in {table.qualified_name}")
        for child_col, parent_col in child.join.items():
            if child_col not in child_columns:
                error(f"child {alias}: join column {child_col} is not in {table.qualified_name}")
            if parent_col not in parent_columns:
                error(f"child {alias}: join column {parent_col} is not in {parent.qualified_name}")
        if not catalog.related(table.qualified_name, parent.qualified_name):
            warn(f"child {alias}: no declared or inferred relationship from {table.qualified_name} to "
                 f"{parent.qualified_name}; the join on {', '.join(child.join)} is taken on trust")

    # ---- every column a value spec reads
    known = parent_columns | set(rule.children)
    for label, spec in rule.value_specs():
        for column in sorted(spec.columns() - known):
            error(f"{label} reads {column}, which is neither a column of {parent.qualified_name} nor a "
                  "child collection")

    # ---- the quality rules
    for check in rule.quality:
        if check.column and check.column not in known:
            error(f"quality {check.label()}: {check.column} is neither a column of "
                  f"{parent.qualified_name} nor a child collection")

    return found


def validate_ruleset(ruleset: RuleSet, catalog: Catalog) -> list[Finding]:
    found: list[Finding] = []
    for rule in ruleset.doc_types.values():
        found += validate_doc_type(rule, catalog)
    return found
