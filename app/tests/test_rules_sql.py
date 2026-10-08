"""The SELECT a rule generates.

One statement per document type, built from the rule: the key as ``record_key``, every parent column
some value spec actually references, and one correlated ``array_agg`` per child table. A ``{where}``
placeholder lets the same statement read every row (full load) or only the changed keys
(incremental), exactly as the hand-written SQL did.

Two things are not negotiable: identifiers are quoted, and child filter values are bound parameters.
The rule files are trusted configuration, but a generated statement that interpolates values is a
habit that outlives the trust.
"""
from __future__ import annotations

import pytest

from crawlerrag.rules import build
from crawlerrag.rules.models import DocTypeRule

RULE = {
    "doc_type": "thing",
    "version": 1,
    "source": {"source_id": "src", "schema": "pub", "table": "thing", "key": "thing_id",
               "order_by": "thing_id"},
    "children": {
        "tags": {"schema": "pub", "table": "thing_tag", "join": {"thing_id": "thing_id"},
                 "select": "tag", "order_by": "seq"},
        "kinds": {"schema": "pub", "table": "thing_tag", "join": {"thing_id": "thing_id"},
                  "select": "kind", "distinct": True},
        "makers": {"schema": "pub", "table": "thing_company", "join": {"thing_id": "thing_id"},
                   "select": "name", "order_by": "seq", "where": {"column": "role", "eq": "Maker"}},
        "others": {"schema": "pub", "table": "thing_company", "join": {"thing_id": "thing_id"},
                   "select": "name", "order_by": "seq", "where": {"column": "role", "ne": "Maker"}},
    },
    "title": {"template": "Thing {name}"},
    "body": [{"label": "Tags", "value": {"field": "tags"}},
             {"label": "Where", "value": {"join": ["city", "state"]}}],
    "metadata": {"year": {"field": "made_on", "format": "year"}},
}


@pytest.fixture(scope="module")
def sql_and_params():
    return build.build_sql(DocTypeRule.model_validate(RULE))


def test_the_key_is_selected_as_record_key(sql_and_params):
    sql, _ = sql_and_params
    assert '"thing_id" AS record_key' in sql


def test_every_referenced_parent_column_is_selected(sql_and_params):
    sql, _ = sql_and_params
    for column in ("name", "city", "state", "made_on", "thing_id", "is_active"):
        assert f'"{column}"' in sql, column


def test_columns_no_rule_mentions_are_not_selected(sql_and_params):
    """A document type reads what it needs; a wide table is not dragged across the wire."""
    sql, _ = sql_and_params
    assert "secret" not in sql


def test_referenced_columns_are_reported_for_validation():
    rule = DocTypeRule.model_validate(RULE)
    assert build.referenced_columns(rule) == {"thing_id", "is_active", "name", "city", "state", "made_on"}


def test_a_child_becomes_a_correlated_array_agg(sql_and_params):
    sql, _ = sql_and_params
    assert "array_agg" in sql
    assert "AS tags" in sql
    assert '"pub"."thing_tag"' in sql


def test_a_child_drops_null_elements(sql_and_params):
    sql, _ = sql_and_params
    assert "FILTER (WHERE" in sql


def test_an_ordered_child_keeps_the_source_order(sql_and_params):
    sql, _ = sql_and_params
    assert 'ORDER BY c."seq"' in sql


def test_a_distinct_child_aggregates_distinct_values(sql_and_params):
    sql, _ = sql_and_params
    assert "array_agg(DISTINCT" in sql


def test_a_child_filter_value_is_a_bound_parameter(sql_and_params):
    sql, params = sql_and_params
    assert "Maker" not in sql
    assert "Maker" in params.values()
    assert any("%(" in sql for _ in [0])


def test_both_child_filters_of_one_table_get_their_own_parameter(sql_and_params):
    _, params = sql_and_params
    assert len([k for k, v in params.items() if v == "Maker"]) == 2


def test_the_where_placeholder_appears_once(sql_and_params):
    sql, _ = sql_and_params
    assert sql.count("{where}") == 1


def test_the_statement_formats_for_a_full_load_and_for_keys(sql_and_params):
    sql, _ = sql_and_params
    assert "WHERE true" in sql.format(where="true")
    assert "ANY(%(keys)s)" in sql.format(where='p."thing_id" = ANY(%(keys)s)')


def test_the_order_by_of_the_parent_is_applied(sql_and_params):
    sql, _ = sql_and_params
    assert sql.rstrip().endswith('"thing_id"') or 'ORDER BY p."thing_id"' in sql


def test_identifiers_are_quoted():
    """A column called "order" or "select" must not break the statement."""
    rule = dict(RULE, body=[{"label": "Order", "value": {"field": "order"}}], children={}, metadata={})
    sql, _ = build.build_sql(DocTypeRule.model_validate(rule))
    assert '"order"' in sql
