"""Business rules from YAML, enforced on the extracted rows before anything is written.

The gate sits between extract and stage, so a source that went wrong (a key that turned up NULL, a
column that is suddenly empty for every row, a value outside the allowed set) does not quietly replace
good documents. An ``error`` aborts that document type's batch and leaves the watermark where it was,
so the next run reads the same window again. A ``warn`` is recorded and the batch continues.
"""
from __future__ import annotations

import pytest

from crawlerrag.ingest import pipeline

from .conftest import FakeEmbedder


def rule_with(env, doc_type, quality):
    """The shipped rule for a doc type, with its quality rules replaced."""
    rule = env["ruleset"].doc_types[doc_type].model_copy(deep=True)
    from crawlerrag.rules.models import QualityRule
    rule.quality = [QualityRule.model_validate(q) for q in quality]
    return pipeline.DocType.from_rule(rule)


def ingest(env, doc_types, **kw):
    embedder = FakeEmbedder()
    return pipeline.run(env["vector"], env["source"], env["settings"], doc_types,
                        embedder_factory=lambda s: embedder, **kw), embedder


def seed(env):
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0001-2017", firm="Cantrell Drug Company")
        c.drug(run, "D-0002-2017", firm="Acme Pharma")


def findings(env):
    return env["vector"].execute(
        "SELECT * FROM ingest.quality_finding ORDER BY finding_id").fetchall()


def watermark(env, doc_type):
    row = env["vector"].execute("SELECT change_id FROM ingest.watermark WHERE doc_type = %s",
                                (doc_type,)).fetchone()
    return row["change_id"] if row else None


# ---------------------------------------------------------------- passing rules
def test_rules_that_hold_let_the_batch_through(env):
    seed(env)
    dt = rule_with(env, "drug_recall", [{"rule": "not_null", "column": "recall_number"},
                                        {"rule": "unique", "column": "recall_number"}])
    result, _ = ingest(env, [dt])
    assert result.batches[0].status == "succeeded"
    assert findings(env) == []


def test_allowed_values_that_hold_let_the_batch_through(env):
    seed(env)
    dt = rule_with(env, "drug_recall",
                   [{"rule": "allowed_values", "column": "classification", "values": ["Class I", "Class II"]}])
    assert ingest(env, [dt])[0].batches[0].status == "succeeded"


# ---------------------------------------------------------------- an error stops the batch
def test_a_null_in_a_required_column_fails_the_batch(env):
    seed(env)
    env["admin"].execute("UPDATE drug.recall SET classification = NULL WHERE recall_number = 'D-0001-2017'")
    dt = rule_with(env, "drug_recall", [{"rule": "not_null", "column": "classification"}])
    result, _ = ingest(env, [dt])
    assert result.batches[0].status == "failed"
    assert env["vector"].execute("SELECT count(*) AS n FROM rag.document").fetchone()["n"] == 0


def test_a_failed_gate_leaves_the_watermark_alone_so_the_window_is_read_again(env):
    seed(env)
    env["admin"].execute("UPDATE drug.recall SET classification = NULL")
    dt = rule_with(env, "drug_recall", [{"rule": "not_null", "column": "classification"}])
    ingest(env, [dt])
    assert watermark(env, "drug_recall") is None
    env["admin"].execute("UPDATE drug.recall SET classification = 'Class II'")
    result, _ = ingest(env, [dt])
    assert result.batches[0].status == "succeeded" and result.batches[0].stats.inserted == 2


def test_the_finding_says_which_rule_broke_and_how_often(env):
    seed(env)
    env["admin"].execute("UPDATE drug.recall SET classification = NULL")
    dt = rule_with(env, "drug_recall", [{"rule": "not_null", "column": "classification"}])
    ingest(env, [dt])
    rows = findings(env)
    assert len(rows) == 1
    assert (rows[0]["level"], rows[0]["rule"], rows[0]["column_name"]) == ("error", "not_null", "classification")
    assert rows[0]["failed_rows"] == 2
    assert "classification" in rows[0]["message"]


def test_a_value_outside_the_allowed_set_fails_the_batch(env):
    seed(env)
    env["admin"].execute("UPDATE drug.recall SET classification = 'Class IV' WHERE recall_number = 'D-0002-2017'")
    dt = rule_with(env, "drug_recall",
                   [{"rule": "allowed_values", "column": "classification", "values": ["Class I", "Class II"]}])
    result, _ = ingest(env, [dt])
    assert result.batches[0].status == "failed"
    assert "Class IV" in findings(env)[0]["message"]


def test_too_many_nulls_fails_when_the_rule_says_error(env):
    seed(env)
    env["admin"].execute("UPDATE drug.recall SET reason_for_recall = NULL WHERE recall_number = 'D-0001-2017'")
    dt = rule_with(env, "drug_recall",
                   [{"rule": "max_null_fraction", "column": "reason_for_recall", "max": 0.2}])
    assert ingest(env, [dt])[0].batches[0].status == "failed"


def test_a_source_that_suddenly_returns_nothing_fails_when_a_minimum_is_set(env):
    """A publisher that answers 200 with an empty list must not empty the index."""
    seed(env)
    dt = rule_with(env, "drug_recall", [{"rule": "min_rows", "min": 5}])
    result, _ = ingest(env, [dt])
    assert result.batches[0].status == "failed"
    assert "2" in findings(env)[0]["message"]


# ---------------------------------------------------------------- a warning only records
def test_a_warning_is_recorded_and_the_batch_continues(env):
    seed(env)
    env["admin"].execute("UPDATE drug.recall SET reason_for_recall = NULL WHERE recall_number = 'D-0001-2017'")
    dt = rule_with(env, "drug_recall", [{"rule": "max_null_fraction", "column": "reason_for_recall",
                                        "max": 0.2, "severity": "warn"}])
    result, _ = ingest(env, [dt])
    assert result.batches[0].status == "succeeded"
    assert [r["level"] for r in findings(env)] == ["warning"]
    assert env["vector"].execute("SELECT count(*) AS n FROM rag.document").fetchone()["n"] == 2


def test_a_finding_is_tied_to_the_batch_that_found_it(env):
    seed(env)
    env["admin"].execute("UPDATE drug.recall SET reason_for_recall = NULL")
    dt = rule_with(env, "drug_recall", [{"rule": "max_null_fraction", "column": "reason_for_recall",
                                        "max": 0.2, "severity": "warn"}])
    result, _ = ingest(env, [dt])
    assert findings(env)[0]["batch_id"] == result.batches[0].batch_id


# ---------------------------------------------------------------- rules may look at child collections
def test_a_rule_can_require_a_child_collection(env):
    seed(env)
    dt = rule_with(env, "drug_recall", [{"rule": "not_null", "column": "product_ndcs", "severity": "warn"}])
    result, _ = ingest(env, [dt])
    assert result.batches[0].status == "succeeded"
    assert findings(env)[0]["failed_rows"] == 2          # neither seeded recall has an NDC


# ---------------------------------------------------------------- the rules that ship with the app
def test_the_shipped_rules_pass_on_real_looking_rows(env):
    seed(env)
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall"])
    result, _ = ingest(env, doc_types)
    assert result.batches[0].status == "succeeded"
    assert [r for r in findings(env) if r["level"] == "error"] == []
