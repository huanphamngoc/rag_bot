"""Loading the YAML rule folder.

The rules are the only place a document type is described, so a mistake in them must fail at load
time with the file name in the message - never half-load and build documents with a silently missing
line. These tests also pin the two guarantees the pipeline depends on: the rule digest changes when
the document text would change (which forces a full re-check), and YAML is parsed safely.
"""
from __future__ import annotations

import pytest

from crawlerrag.rules import RuleError, load_rules

MINIMAL = """
doc_type: thing
version: 1
source:
  source_id: some_source
  schema: pub
  table: thing
  key: thing_id
title:
  field: name
body:
  - label: Name
    value: {field: name}
"""


def write(dirpath, name, text):
    (dirpath / name).write_text(text, encoding="utf-8")


@pytest.fixture
def folder(tmp_path):
    """A loadable folder: one doc type plus the two shared files."""
    (tmp_path / "doc_types").mkdir()
    write(tmp_path / "doc_types", "thing.yaml", MINIMAL)
    write(tmp_path, "catalog.yaml", "schemas: [pub]\n")
    write(tmp_path, "qualify.yaml", "version: 1\nlimits: {min_chars: 3, max_chars: 2000}\n")
    return tmp_path


# ---------------------------------------------------------------- the rules that ship with the app
def test_the_shipped_rules_load(ruleset):
    assert set(ruleset.doc_types) == {"drug_recall", "cpsc_recall"}


def test_every_shipped_doc_type_knows_its_crawler_source(ruleset):
    assert ruleset.doc_types["drug_recall"].source.source_id == "openfda_enforcement"
    assert ruleset.doc_types["cpsc_recall"].source.source_id == "cpsc_recall"


def test_the_shipped_rules_declare_which_schemas_to_introspect(ruleset):
    assert set(ruleset.catalog.schemas) >= {"crawl", "drug", "retail"}


def test_the_shipped_qualify_rules_are_loaded(ruleset):
    assert ruleset.qualify.limits.max_chars > 0


# ---------------------------------------------------------------- failures name the file
def test_a_folder_without_doc_types_is_refused(tmp_path):
    with pytest.raises(RuleError, match="doc_types"):
        load_rules(tmp_path)


def test_broken_yaml_names_the_file(folder):
    write(folder / "doc_types", "thing.yaml", "doc_type: [unclosed\n")
    with pytest.raises(RuleError, match="thing.yaml"):
        load_rules(folder)


def test_an_unknown_key_names_the_file_and_the_key(folder):
    write(folder / "doc_types", "thing.yaml", MINIMAL + "\nbodyy: []\n")
    with pytest.raises(RuleError, match="thing.yaml"):
        load_rules(folder)


def test_a_missing_required_section_is_refused(folder):
    write(folder / "doc_types", "thing.yaml", "doc_type: thing\nversion: 1\n")
    with pytest.raises(RuleError, match="thing.yaml"):
        load_rules(folder)


def test_the_doc_type_must_match_the_file_name(folder):
    write(folder / "doc_types", "other.yaml", MINIMAL)
    with pytest.raises(RuleError, match="other"):
        load_rules(folder)


def test_two_files_cannot_declare_the_same_doc_type(folder):
    write(folder / "doc_types", "thing2.yaml", MINIMAL.replace("doc_type: thing", "doc_type: thing2"))
    (folder / "doc_types" / "thing2.yaml").write_text(MINIMAL, encoding="utf-8")
    with pytest.raises(RuleError):
        load_rules(folder)


def test_yaml_object_tags_are_not_executed(folder):
    """safe_load only: a rule file is configuration, never code."""
    write(folder / "doc_types", "thing.yaml", MINIMAL + "\nextra: !!python/object/apply:os.system ['echo hi']\n")
    with pytest.raises(RuleError):
        load_rules(folder)


def test_an_unknown_quality_rule_is_refused(folder):
    write(folder / "doc_types", "thing.yaml", MINIMAL + "\nquality:\n  - {rule: no_such_rule, column: name}\n")
    with pytest.raises(RuleError):
        load_rules(folder)


def test_scd2_cannot_track_an_unknown_attribute(folder):
    write(folder / "doc_types", "thing.yaml", MINIMAL + "\nscd2:\n  track: [titel]\n")
    with pytest.raises(RuleError):
        load_rules(folder)


def test_an_attribute_cannot_be_both_tracked_and_overwritten(folder):
    write(folder / "doc_types", "thing.yaml",
          MINIMAL + "\nscd2:\n  track: [title, body]\n  overwrite: [body, url]\n")
    with pytest.raises(RuleError, match="body"):
        load_rules(folder)


# ---------------------------------------------------------------- defaults
def test_scd2_defaults_to_versioning_the_document_text(folder):
    rule = load_rules(folder).doc_types["thing"]
    assert rule.scd2.track == ["title", "body"]
    assert rule.scd2.overwrite == ["url", "metadata"]
    assert rule.scd2.version_on_activation_change is True


def test_the_active_column_defaults_to_is_active(folder):
    assert load_rules(folder).doc_types["thing"].source.active_column == "is_active"


# ---------------------------------------------------------------- the digest that drives a full re-check
def test_the_digest_changes_when_the_document_text_would_change(folder):
    before = load_rules(folder).doc_types["thing"].text_digest
    write(folder / "doc_types", "thing.yaml", MINIMAL.replace("label: Name", "label: Full name"))
    assert load_rules(folder).doc_types["thing"].text_digest != before


def test_the_digest_ignores_rules_that_do_not_touch_the_text(folder):
    """Tightening a quality rule must not trigger a full rebuild of every document."""
    before = load_rules(folder).doc_types["thing"].text_digest
    write(folder / "doc_types", "thing.yaml", MINIMAL + "\nquality:\n  - {rule: not_null, column: name}\n")
    assert load_rules(folder).doc_types["thing"].text_digest == before


def test_the_digest_is_stable_across_loads(folder):
    assert load_rules(folder).doc_types["thing"].text_digest == load_rules(folder).doc_types["thing"].text_digest


def test_comments_and_key_order_do_not_change_the_digest(folder):
    before = load_rules(folder).doc_types["thing"].text_digest
    write(folder / "doc_types", "thing.yaml", "# a comment\n" + MINIMAL.replace(
        "doc_type: thing\nversion: 1\n", "version: 1\ndoc_type: thing\n"))
    assert load_rules(folder).doc_types["thing"].text_digest == before
