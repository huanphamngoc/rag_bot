"""The metadata catalog: tables, columns, relationships.

Read from Postgres (``tests/integration/test_catalog_db.py`` covers that against a real database);
here the parts that are pure logic. The one that matters most is *inferred* relationships. The
crawler's live database declares its foreign keys, but a database restored without constraints does
not, and the child's primary key still starts with the parent's (``PRIMARY KEY (recall_number,
product_ndc)`` under ``PRIMARY KEY (recall_number)``). So relationships are also derived from the
keys, and labelled as derived rather than declared.
"""
from __future__ import annotations

import datetime as dt

import pytest

from crawlerrag.meta.introspect import infer_relationships
from crawlerrag.meta.models import Catalog, Column, Relationship, Table

NOW = dt.datetime(2026, 10, 5, 12, 0, tzinfo=dt.timezone.utc)


def col(table, name, ordinal, data_type="text", nullable=True, schema="drug"):
    return Column(schema=schema, table=table, name=name, ordinal=ordinal, data_type=data_type,
                  is_nullable=nullable)


def table(name, pk, columns, schema="drug", **kw):
    return Table(schema=schema, name=name, kind="table", primary_key=tuple(pk),
                 columns=tuple(col(name, c, i + 1, schema=schema) for i, c in enumerate(columns)), **kw)


RECALL = table("recall", ["recall_number"], ["recall_number", "classification", "is_active"])
NDC = table("recall_product_ndc", ["recall_number", "product_ndc"], ["recall_number", "product_ndc"])
UNRELATED = table("other", ["other_id"], ["other_id", "note"])


# ---------------------------------------------------------------- inferred relationships
def test_a_child_whose_key_starts_with_the_parents_key_is_inferred():
    rels = infer_relationships([RECALL, NDC])
    assert [(r.from_table, r.to_table, r.kind) for r in rels] == \
        [("drug.recall_product_ndc", "drug.recall", "inferred")]


def test_the_inferred_relationship_names_the_joining_columns():
    rel = infer_relationships([RECALL, NDC])[0]
    assert rel.from_columns == ("recall_number",) and rel.to_columns == ("recall_number",)


def test_tables_that_share_no_key_are_not_related():
    assert infer_relationships([RECALL, UNRELATED]) == []


def test_a_table_is_not_related_to_itself():
    assert infer_relationships([RECALL]) == []


def test_a_child_with_the_same_key_as_the_parent_is_not_a_child():
    """Same primary key means one row per parent row - a side table, not a collection."""
    same = table("recall_extra", ["recall_number"], ["recall_number", "note"])
    assert infer_relationships([RECALL, same]) == []


def test_a_parent_without_a_primary_key_relates_to_nothing():
    keyless = table("recall", [], ["recall_number"])
    assert infer_relationships([keyless, NDC]) == []


def test_a_declared_relationship_is_not_replaced_by_an_inferred_one():
    declared = Relationship(from_table="drug.recall_product_ndc", from_columns=("recall_number",),
                            to_table="drug.recall", to_columns=("recall_number",), kind="declared",
                            constraint="recall_product_ndc_recall_number_fkey")
    rels = infer_relationships([RECALL, NDC], declared=[declared])
    assert [r.kind for r in rels] == ["declared"]


# ---------------------------------------------------------------- the catalog itself
def cat(tables=(RECALL, NDC), rels=()):
    return Catalog(captured_at=NOW, tables=tuple(tables), relationships=tuple(rels))


def test_a_table_is_found_by_its_qualified_name():
    assert cat().table("drug.recall").name == "recall"
    assert cat().table("drug.nope") is None


def test_a_column_is_found_by_name():
    assert cat().table("drug.recall").column("classification").data_type == "text"
    assert cat().table("drug.recall").column("nope") is None


def test_the_digest_covers_tables_columns_and_keys():
    before = cat().digest
    wider = table("recall", ["recall_number"], ["recall_number", "classification", "is_active", "new_col"])
    assert cat(tables=(wider, NDC)).digest != before


def test_the_digest_ignores_when_the_catalog_was_captured():
    other = Catalog(captured_at=NOW + dt.timedelta(days=1), tables=cat().tables, relationships=())
    assert other.digest == cat().digest


def test_the_digest_ignores_the_order_tables_come_back_in():
    assert cat(tables=(NDC, RECALL)).digest == cat(tables=(RECALL, NDC)).digest


def test_the_catalog_round_trips_through_plain_data():
    full = cat(rels=infer_relationships([RECALL, NDC]))
    assert Catalog.from_dict(full.to_dict()).digest == full.digest


def test_the_yaml_export_is_readable_and_names_the_source():
    text = cat(rels=infer_relationships([RECALL, NDC])).to_yaml()
    assert "drug.recall" in text and "inferred" in text
    import yaml
    assert yaml.safe_load(text)["tables"]


def test_counts_are_reported_for_the_status_output():
    full = cat(rels=infer_relationships([RECALL, NDC]))
    assert (full.table_count, full.column_count, full.relationship_count) == (2, 5, 1)


@pytest.mark.parametrize("name", ["drug.recall", "drug.recall_product_ndc"])
def test_every_table_knows_its_qualified_name(name):
    assert cat().table(name).qualified_name == name


# ---------------------------------------------------------------- the exclude list in catalog.yaml
@pytest.mark.parametrize("name, excluded", [
    ("crawl.http_request_log", True),
    ("crawl.raw_record", True),
    ("crawl.record_change", False),
    ("drug.recall", False),
])
def test_tables_named_in_the_exclude_list_are_skipped(name, excluded):
    from crawlerrag.meta.introspect import _excluded
    assert _excluded(name, ["crawl.http_request_log", "crawl.raw_record"]) is excluded


def test_the_exclude_list_takes_patterns():
    from crawlerrag.meta.introspect import _excluded
    assert _excluded("crawl.raw_record_2024", ["crawl.raw_record*"])
    assert not _excluded("drug.recall", ["crawl.*"])


def test_the_shipped_catalog_rules_exclude_the_two_big_log_tables(ruleset):
    assert set(ruleset.catalog.exclude) == {"crawl.http_request_log", "crawl.raw_record"}
