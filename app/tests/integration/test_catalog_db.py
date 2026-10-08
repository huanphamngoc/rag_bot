"""Reading the metadata catalog out of the crawler's Postgres, read-only.

What the catalog is for: the YAML rules name tables and columns of a database this project does not
own, so the names have to be checked against the database itself.

The source database here is the fixture schema in ``source_schema.sql``, which was rebuilt from the
crawler's ``information_schema`` and therefore carries the columns, types and keys but **not** the
foreign keys. That makes it the useful case to test: the real crawler database declares 25 foreign
keys (measured 2026-10-05), so only a schema like this one exercises the inference. The catalog is also
read through ``analyst_ro``, which may only SELECT.
"""
from __future__ import annotations

import pytest

from crawlerrag.meta import catalog as meta
from crawlerrag.meta import introspect

SCHEMAS = ("crawl", "drug", "retail")


@pytest.fixture
def cat(env):
    return introspect.read_catalog(env["source"], SCHEMAS)


# ---------------------------------------------------------------- tables and columns
def test_the_tables_of_the_requested_schemas_are_found(cat):
    names = {t.qualified_name for t in cat.tables}
    assert {"crawl.record_change", "drug.recall", "drug.recall_product_ndc", "retail.cpsc_recall",
            "retail.cpsc_recall_hazard"} <= names


def test_tables_of_other_schemas_are_left_out(cat):
    assert all(t.schema in SCHEMAS for t in cat.tables)


def test_a_column_carries_the_type_postgres_reports(cat):
    recall = cat.table("drug.recall")
    assert recall.column("recall_number").data_type == "text"
    assert recall.column("recall_initiation_date").data_type == "date"
    assert recall.column("is_active").data_type == "boolean"


def test_a_not_null_column_is_marked_not_nullable(cat):
    assert cat.table("drug.recall").column("is_active").is_nullable is False
    assert cat.table("drug.recall").column("classification").is_nullable is True


def test_an_array_column_is_reported_as_an_array(cat):
    assert "[]" in cat.table("retail.cpsc_recall").column("injuries").data_type


def test_columns_keep_their_position_in_the_table(cat):
    ordinals = [c.ordinal for c in cat.table("drug.recall").columns]
    assert ordinals == sorted(ordinals)


def test_the_primary_key_is_read(cat):
    assert cat.table("drug.recall").primary_key == ("recall_number",)
    assert cat.table("drug.recall_product_ndc").primary_key == ("recall_number", "product_ndc")
    assert cat.table("retail.cpsc_recall_company").primary_key == ("recall_id", "role", "seq")


# ---------------------------------------------------------------- relationships
def test_this_schema_declares_no_foreign_keys_between_a_recall_and_its_children(cat):
    """The case the inference exists for. Where a foreign key *is* declared, the declared one wins."""
    declared = {(r.from_table, r.to_table) for r in cat.relationships if r.kind == "declared"}
    assert ("drug.recall_product_ndc", "drug.recall") not in declared


def test_a_child_table_is_inferred_from_its_key(cat):
    found = {(r.from_table, r.to_table, r.kind) for r in cat.relationships}
    assert ("drug.recall_product_ndc", "drug.recall", "inferred") in found


def test_every_cpsc_child_table_is_related_to_its_recall(cat):
    children = {r.from_table for r in cat.relationships if r.to_table == "retail.cpsc_recall"}
    assert {f"retail.cpsc_recall_{x}" for x in
            ("hazard", "product", "company", "major_retailer", "remedy_option", "country")} <= children


def test_the_change_log_is_not_related_to_the_tables_it_describes(cat):
    """crawl.record_change.record_key is a text key, not a foreign key: the join is the rule's job.

    It does have a real foreign key to crawl.crawl_run, which is what a declared relationship looks
    like here - and the reason the catalog keeps the two kinds apart.
    """
    targets = {r.to_table for r in cat.relationships if r.from_table == "crawl.record_change"}
    assert "drug.recall" not in targets and "retail.cpsc_recall" not in targets
    assert ("crawl.crawl_run", "declared") in {
        (r.to_table, r.kind) for r in cat.relationships if r.from_table == "crawl.record_change"}


# ---------------------------------------------------------------- the shipped rules validate against it
def test_the_shipped_rules_match_the_real_schema(cat, ruleset):
    from crawlerrag.rules.validate import errors, validate_ruleset
    found = validate_ruleset(ruleset, cat)
    assert errors(found) == []


def test_a_rule_naming_a_column_that_is_not_there_is_caught(cat, ruleset):
    from crawlerrag.rules.validate import errors, validate_doc_type
    rule = ruleset.doc_types["drug_recall"].model_copy(deep=True)
    rule.body[0].value.field = "recalling_firmm"
    assert errors(validate_doc_type(rule, cat))


# ---------------------------------------------------------------- persistence in the vector database
def test_the_catalog_is_stored_and_read_back_unchanged(env, cat):
    run_id = meta.save_catalog(env["vector"], cat, source="integration")
    assert run_id > 0
    assert meta.load_catalog(env["vector"]).digest == cat.digest


def test_the_stored_catalog_counts_what_it_found(env, cat):
    meta.save_catalog(env["vector"], cat, source="integration")
    row = env["vector"].execute("SELECT * FROM meta.catalog_run ORDER BY run_id DESC LIMIT 1").fetchone()
    assert (row["table_count"], row["digest"]) == (cat.table_count, cat.digest)
    assert row["column_count"] == cat.column_count > 50


def test_an_unchanged_schema_keeps_the_same_digest(env):
    first = introspect.read_catalog(env["source"], SCHEMAS)
    assert introspect.read_catalog(env["source"], SCHEMAS).digest == first.digest


def test_a_new_column_in_the_source_changes_the_digest(env):
    before = introspect.read_catalog(env["source"], SCHEMAS).digest
    env["admin"].execute("ALTER TABLE drug.recall ADD COLUMN scratch text")
    try:
        assert introspect.read_catalog(env["source"], SCHEMAS).digest != before
    finally:
        env["admin"].execute("ALTER TABLE drug.recall DROP COLUMN scratch")


def test_reading_the_catalog_needs_no_write_permission(env):
    """It runs as analyst_ro in production; this connection *is* analyst_ro."""
    introspect.read_catalog(env["source"], SCHEMAS)
    import psycopg
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        env["source"].execute("CREATE TABLE drug.nope (x int)")


def test_only_the_latest_catalog_is_returned(env, cat):
    meta.save_catalog(env["vector"], cat, source="first")
    meta.save_catalog(env["vector"], cat, source="second")
    row = env["vector"].execute("SELECT source FROM meta.catalog_run ORDER BY run_id DESC LIMIT 1").fetchone()
    assert row["source"] == "second"
