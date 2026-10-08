"""Checking the YAML rules against the metadata extracted from Postgres.

A rule file names tables and columns of a database this project only reads. The catalog (table info,
column info, relationships) is pulled from Postgres, so a typo in a rule is caught before a single
row is read - and before the pipeline builds documents with a silently missing line.

Errors stop a run. Warnings do not: a child table joined on columns that no declared *or* inferred
relationship backs is suspicious, but the crawler's schema has no declared foreign keys between a
recall and its child tables, so refusing to run on that would refuse everything.
"""
from __future__ import annotations

import datetime as dt

import pytest

from crawlerrag.meta.models import Catalog, Column, Relationship, Table
from crawlerrag.rules.models import DocTypeRule
from crawlerrag.rules.validate import errors, validate_doc_type


def col(schema, table, name, ordinal, data_type="text", nullable=True):
    return Column(schema=schema, table=table, name=name, ordinal=ordinal, data_type=data_type,
                  is_nullable=nullable)


def catalog():
    thing = Table(schema="pub", name="thing", kind="table", primary_key=("thing_id",),
                  columns=(col("pub", "thing", "thing_id", 1, nullable=False),
                           col("pub", "thing", "name", 2),
                           col("pub", "thing", "city", 3),
                           col("pub", "thing", "made_on", 4, "date"),
                           col("pub", "thing", "is_active", 5, "boolean", nullable=False)))
    tag = Table(schema="pub", name="thing_tag", kind="table", primary_key=("thing_id", "seq"),
                columns=(col("pub", "thing_tag", "thing_id", 1, nullable=False),
                         col("pub", "thing_tag", "seq", 2, "smallint", nullable=False),
                         col("pub", "thing_tag", "tag", 3),
                         col("pub", "thing_tag", "role", 4)))
    loose = Table(schema="pub", name="loose", kind="table", primary_key=(),
                  columns=(col("pub", "loose", "thing_id", 1), col("pub", "loose", "note", 2)))
    rel = Relationship(from_table="pub.thing_tag", from_columns=("thing_id",),
                       to_table="pub.thing", to_columns=("thing_id",), kind="inferred", constraint=None)
    return Catalog(captured_at=dt.datetime(2026, 10, 5, tzinfo=dt.timezone.utc),
                   tables=(thing, tag, loose), relationships=(rel,))


BASE = {
    "doc_type": "thing",
    "version": 1,
    "source": {"source_id": "src", "schema": "pub", "table": "thing", "key": "thing_id"},
    "children": {"tags": {"schema": "pub", "table": "thing_tag", "join": {"thing_id": "thing_id"},
                          "select": "tag", "order_by": "seq"}},
    "title": {"template": "Thing {name}"},
    "body": [{"label": "City", "value": {"field": "city"}}, {"label": "Tags", "value": {"field": "tags"}}],
    "metadata": {"year": {"field": "made_on", "format": "year"}},
}


def check(**over):
    rule = DocTypeRule.model_validate({**BASE, **over})
    return validate_doc_type(rule, catalog())


def messages(findings):
    return " | ".join(f.message for f in findings)


def test_a_correct_rule_has_no_findings():
    assert check() == []


def test_an_unknown_source_table_is_an_error():
    found = check(source={**BASE["source"], "table": "thingg"})
    assert errors(found) and "pub.thingg" in messages(found)


def test_an_unknown_parent_column_is_an_error():
    found = check(body=[{"label": "X", "value": {"field": "citty"}}])
    assert errors(found) and "citty" in messages(found)


def test_an_unknown_column_in_a_template_is_an_error():
    found = check(title={"template": "Thing {naem}"})
    assert errors(found) and "naem" in messages(found)


def test_an_unknown_column_in_a_join_is_an_error():
    found = check(body=[{"label": "X", "value": {"join": ["city", "stat"]}}])
    assert errors(found) and "stat" in messages(found)


def test_an_unknown_column_in_metadata_is_an_error():
    found = check(metadata={"year": {"field": "made_onn", "format": "year"}})
    assert errors(found) and "made_onn" in messages(found)


def test_a_key_that_is_not_a_unique_key_of_the_table_is_an_error():
    """The watermark strategy assumes record_key identifies exactly one row, both ways."""
    found = check(source={**BASE["source"], "key": "name"})
    assert errors(found) and "name" in messages(found)


def test_a_non_boolean_active_column_is_an_error():
    found = check(source={**BASE["source"], "active_column": "city"})
    assert errors(found) and "city" in messages(found)


def test_a_missing_active_column_is_an_error():
    found = check(source={**BASE["source"], "active_column": "deleted"})
    assert errors(found) and "deleted" in messages(found)


def test_an_unknown_child_table_is_an_error():
    found = check(children={"tags": {**BASE["children"]["tags"], "table": "thing_tags"}})
    assert errors(found) and "thing_tags" in messages(found)


def test_an_unknown_child_select_column_is_an_error():
    found = check(children={"tags": {**BASE["children"]["tags"], "select": "tagg"}})
    assert errors(found) and "tagg" in messages(found)


def test_an_unknown_child_order_column_is_an_error():
    found = check(children={"tags": {**BASE["children"]["tags"], "order_by": "sek"}})
    assert errors(found) and "sek" in messages(found)


def test_an_unknown_child_filter_column_is_an_error():
    found = check(children={"tags": {**BASE["children"]["tags"], "where": {"column": "rol", "eq": "x"}}})
    assert errors(found) and "rol" in messages(found)


def test_a_join_column_missing_on_the_child_is_an_error():
    found = check(children={"tags": {**BASE["children"]["tags"], "join": {"thing_fk": "thing_id"}}})
    assert errors(found) and "thing_fk" in messages(found)


def test_a_join_column_missing_on_the_parent_is_an_error():
    found = check(children={"tags": {**BASE["children"]["tags"], "join": {"thing_id": "id"}}})
    assert errors(found) and "id" in messages(found)


def test_a_child_name_that_collides_with_a_parent_column_is_an_error():
    """Both end up as keys of the same row dict; the child would shadow the column."""
    found = check(children={"city": {**BASE["children"]["tags"]}})
    assert errors(found) and "city" in messages(found)


def test_a_join_no_relationship_backs_is_only_a_warning():
    found = check(children={**BASE["children"],
                            "notes": {"schema": "pub", "table": "loose", "join": {"thing_id": "thing_id"},
                                      "select": "note"}})
    assert errors(found) == []
    assert [f.level for f in found] == ["warning"]


def test_a_quality_rule_on_an_unknown_column_is_an_error():
    found = check(quality=[{"rule": "not_null", "column": "citty"}])
    assert errors(found) and "citty" in messages(found)


def test_a_quality_rule_may_name_a_child_alias():
    assert check(quality=[{"rule": "not_null", "column": "tags", "severity": "warn"}]) == []


def test_findings_name_the_doc_type():
    found = check(source={**BASE["source"], "table": "nope"})
    assert all(f.doc_type == "thing" for f in found)


def test_the_whole_ruleset_can_be_validated_at_once(ruleset):
    """The shipped rules validate against a catalog describing the crawler tables they read."""
    from crawlerrag.rules.validate import validate_ruleset
    found = validate_ruleset(ruleset, catalog())
    # Nothing in this toy catalog matches the real tables, so every doc type must complain.
    assert {f.doc_type for f in errors(found)} == {"drug_recall", "cpsc_recall"}


@pytest.mark.parametrize("level", ["error", "warning"])
def test_findings_render_for_the_terminal(level):
    from crawlerrag.rules.validate import Finding
    assert "thing" in Finding(level=level, doc_type="thing", message="something").render()


# ---------------------------------------------------------------- whole-message matching
def _rule(**kwargs):
    from crawlerrag.rules.models import QualifyRule
    return QualifyRule(id="r", kind="reject", message="m", **kwargs)


@pytest.mark.parametrize("message, matched", [
    ("thanks", True),
    ("Thanks", True),
    ("thanks!", True),
    ("  thanks.  ", True),
    ("cam on", True),                      # diacritics are folded on both sides
    ("thanks to whom was it reported?", False),
    ("smoke", False),
])
def test_equals_any_matches_the_whole_message_only(message, matched):
    assert _rule(equals_any=["thanks", "cảm ơn"]).matches(message) is matched


def test_ok_as_a_substring_pattern_would_catch_ordinary_questions():
    """Why equals_any had to exist. Kept as a test so the reason stays visible."""
    assert _rule(patterns=["ok"]).matches("Which smoke detectors were recalled?") is True
    assert _rule(equals_any=["ok"]).matches("Which smoke detectors were recalled?") is False


def test_a_rule_needs_exactly_one_way_of_matching():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        _rule(patterns=["a"], equals_any=["b"])
    with pytest.raises(pydantic.ValidationError):
        _rule()
