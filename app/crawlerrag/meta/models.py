"""What a database looks like, as plain data.

A catalog is a snapshot of the crawler's schema: tables, their columns and keys, and the relationships
between them. It exists so the YAML rules can be checked against the real database, and so a change in
the source schema is visible (the digest changes) instead of surfacing later as a document with a
missing line.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Sequence

KINDS = ("declared", "inferred")


@dataclass(frozen=True)
class Column:
    schema: str
    table: str
    name: str
    ordinal: int
    data_type: str                       # as format_type() prints it: text, date, boolean, text[], ...
    is_nullable: bool
    has_default: bool = False
    max_length: int | None = None
    precision: int | None = None
    scale: int | None = None
    comment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ordinal": self.ordinal, "data_type": self.data_type,
                "is_nullable": self.is_nullable, "has_default": self.has_default,
                "max_length": self.max_length, "precision": self.precision, "scale": self.scale,
                "comment": self.comment}


@dataclass(frozen=True)
class Table:
    schema: str
    name: str
    kind: str
    columns: tuple[Column, ...] = ()
    primary_key: tuple[str, ...] = ()
    unique_keys: tuple[tuple[str, ...], ...] = ()
    est_rows: int | None = None
    comment: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)

    def is_unique(self, columns: Sequence[str]) -> bool:
        wanted = tuple(columns)
        return wanted == self.primary_key or wanted in self.unique_keys

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "name": self.name, "kind": self.kind,
                "primary_key": list(self.primary_key),
                "unique_keys": [list(k) for k in self.unique_keys],
                "est_rows": self.est_rows, "comment": self.comment,
                "columns": [c.to_dict() for c in self.columns]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Table":
        columns = tuple(Column(schema=data["schema"], table=data["name"], **c) for c in data["columns"])
        return cls(schema=data["schema"], name=data["name"], kind=data["kind"], columns=columns,
                   primary_key=tuple(data.get("primary_key") or ()),
                   unique_keys=tuple(tuple(k) for k in data.get("unique_keys") or ()),
                   est_rows=data.get("est_rows"), comment=data.get("comment"))


@dataclass(frozen=True)
class Relationship:
    from_table: str                      # schema-qualified
    from_columns: tuple[str, ...]
    to_table: str
    to_columns: tuple[str, ...]
    kind: str                            # declared (a foreign key) | inferred (from the keys)
    constraint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"from_table": self.from_table, "from_columns": list(self.from_columns),
                "to_table": self.to_table, "to_columns": list(self.to_columns),
                "kind": self.kind, "constraint": self.constraint}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relationship":
        return cls(from_table=data["from_table"], from_columns=tuple(data["from_columns"]),
                   to_table=data["to_table"], to_columns=tuple(data["to_columns"]),
                   kind=data["kind"], constraint=data.get("constraint"))


@dataclass(frozen=True)
class Catalog:
    captured_at: dt.datetime
    tables: tuple[Table, ...] = ()
    relationships: tuple[Relationship, ...] = ()

    # ---- lookups
    def table(self, qualified_name: str) -> Table | None:
        return next((t for t in self.tables if t.qualified_name == qualified_name), None)

    def related(self, from_table: str, to_table: str) -> Relationship | None:
        return next((r for r in self.relationships
                     if r.from_table == from_table and r.to_table == to_table), None)

    # ---- counts, for the status output
    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def column_count(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    # ---- the fingerprint
    @property
    def digest(self) -> str:
        """Covers the structure, never the time it was read, nor the order rows came back in.

        Row estimates are left out too: ``reltuples`` moves after every autovacuum and the schema has
        not changed because of it.
        """
        shape = {
            "tables": sorted(
                [{"name": t.qualified_name, "kind": t.kind, "primary_key": list(t.primary_key),
                  "unique_keys": sorted([list(k) for k in t.unique_keys]),
                  "columns": sorted([[c.name, c.ordinal, c.data_type, c.is_nullable] for c in t.columns])}
                 for t in self.tables], key=lambda d: d["name"]),
            "relationships": sorted([r.to_dict() for r in self.relationships],
                                    key=lambda d: (d["from_table"], d["to_table"], d["kind"])),
        }
        return hashlib.sha256(json.dumps(shape, sort_keys=True).encode("utf-8")).hexdigest()

    # ---- plain data
    def to_dict(self) -> dict[str, Any]:
        return {"captured_at": self.captured_at.isoformat(),
                "digest": self.digest,
                "tables": [t.to_dict() for t in self.tables],
                "relationships": [r.to_dict() for r in self.relationships]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Catalog":
        captured = data["captured_at"]
        return cls(captured_at=dt.datetime.fromisoformat(captured) if isinstance(captured, str) else captured,
                   tables=tuple(Table.from_dict(t) for t in data.get("tables") or ()),
                   relationships=tuple(Relationship.from_dict(r) for r in data.get("relationships") or ()))

    def to_yaml(self) -> str:
        import yaml
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True, width=120)
