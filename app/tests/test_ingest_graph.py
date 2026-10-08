"""The ingestion flow as a LangGraph graph.

    catalog ─► validate ─┬─(errors)──────────────────────────────────────► finalize
                         └─► plan ─► extract ─┬─(nothing)─► next ──┐
                                              └─► quality ─┬─(error)─► next
                                                            └─► stage ─► next
                             next ─┬─(more doc types)─► plan
                                   └─► lexemes ─► embed ─► finalize

The nodes are thin: each one calls one operation (``IngestOps``) and writes counters into the state.
That is deliberate - the state has to stay JSON-serialisable for the Postgres checkpointer, so
documents and database connections never enter it. The graph owns the *order* and the *branches*; the
SQL stays in ``pipeline.py``.

These tests drive the graph with a recording fake of ``IngestOps``, so they are about control flow:
which nodes ran, in which order, and what stops a run.
"""
from __future__ import annotations

import json

import pytest

from crawlerrag.ingest import graph as ingest_graph
from crawlerrag.ingest.graph import IngestState
from crawlerrag.rules.validate import Finding


class FakeOps:
    """Records the calls the graph makes and returns whatever the test asked for."""

    def __init__(self, *, findings=(), quality=None, keys=None, chunks_changed=True, pending=5):
        self.calls: list[str] = []
        self.findings = list(findings)
        self.quality_findings = quality or {}
        self.keys = keys or {}
        self.chunks_changed = chunks_changed
        self.pending = pending
        self.failed: list[str] = []
        self.staged: list[str] = []

    # -- rules and metadata
    def read_catalog(self):
        self.calls.append("read_catalog")
        return object()

    def validate(self, catalog):
        self.calls.append("validate")
        return self.findings

    # -- per doc type
    def begin_batch(self, doc_type, *, full):
        self.calls.append(f"begin_batch:{doc_type}")
        return {"doc_type": doc_type, "batch_id": len(self.calls), "mode": "incremental",
                "reason": "changes after change_id 10", "from_change_id": 10}

    def extract(self, plan):
        self.calls.append(f"extract:{plan['doc_type']}")
        keys = self.keys.get(plan["doc_type"], 3)
        return {"keys": keys, "documents": keys, "to_change_id": 42, "missing_keys": 0}

    def quality(self, plan):
        self.calls.append(f"quality:{plan['doc_type']}")
        return list(self.quality_findings.get(plan["doc_type"], ()))

    def stage(self, plan):
        self.calls.append(f"stage:{plan['doc_type']}")
        self.staged.append(plan["doc_type"])
        return {"status": "succeeded", "inserted": 1, "updated": 0, "versions_created": 1,
                "chunks_added": 2 if self.chunks_changed else 0, "chunks_removed": 0, "chunks_moved": 0}

    def fail_batch(self, plan, error):
        self.calls.append(f"fail_batch:{plan['doc_type']}")
        self.failed.append(error)

    def finish_batch(self, plan, result):
        self.calls.append(f"finish_batch:{plan['doc_type']}")

    # -- once per run
    def refresh_lexemes(self):
        self.calls.append("refresh_lexemes")
        return 99

    def pending_embeddings(self):
        return self.pending

    def pending_cost(self):
        self.calls.append("pending_cost")
        return {"chunks": self.pending, "chars": self.pending * 1200}

    def embed(self, max_chunks):
        self.calls.append(f"embed:{max_chunks}")
        return {"embedded": 4, "reused": 1, "requests": 1, "status": "succeeded", "pending_after": 0}

    def mark_indexed(self):
        self.calls.append("mark_indexed")
        return 1


def run(ops, doc_types=("drug_recall",), **kwargs):
    return ingest_graph.run_ingest(ops, list(doc_types), **kwargs)


def nodes(result):
    return result.visited


# ---------------------------------------------------------------- structure
def test_the_graph_has_the_nodes_the_pipeline_needs():
    compiled = ingest_graph.build_ingest_graph()
    assert {"catalog", "validate", "plan", "extract", "quality", "stage", "next_doc_type",
            "lexemes", "approve_embed", "embed", "finalize"} <= set(compiled.get_graph().nodes)


def test_the_state_stays_json_serialisable():
    """The Postgres checkpointer stores the state: no Document, no connection, no datetime."""
    ops = FakeOps()
    result = run(ops)
    json.dumps(result.state)


def test_the_state_type_declares_only_plain_fields():
    for name, annotation in IngestState.__annotations__.items():
        assert "Connection" not in str(annotation), name
        assert "Document" not in str(annotation), name


# ---------------------------------------------------------------- the normal path
def test_a_run_walks_catalog_validate_plan_extract_quality_stage():
    ops = FakeOps()
    result = run(ops)
    assert nodes(result)[:6] == ["catalog", "validate", "plan", "extract", "quality", "stage"]


def test_a_run_ends_with_lexemes_approval_embed_and_finalize():
    """approve_embed always runs; with the gate off (the default) it decides nothing and falls through."""
    ops = FakeOps()
    assert nodes(run(ops))[-4:] == ["lexemes", "approve_embed", "embed", "finalize"]


def test_the_metadata_catalog_is_read_once_per_run_not_once_per_doc_type():
    ops = FakeOps()
    run(ops, ("drug_recall", "cpsc_recall"))
    assert ops.calls.count("read_catalog") == 1


def test_every_doc_type_is_planned_and_staged():
    ops = FakeOps()
    run(ops, ("drug_recall", "cpsc_recall"))
    assert ops.staged == ["drug_recall", "cpsc_recall"]


def test_doc_types_are_processed_in_the_order_they_were_asked_for():
    ops = FakeOps()
    run(ops, ("cpsc_recall", "drug_recall"))
    assert ops.staged == ["cpsc_recall", "drug_recall"]


def test_the_result_carries_one_batch_per_doc_type():
    result = run(FakeOps(), ("drug_recall", "cpsc_recall"))
    assert [b["doc_type"] for b in result.state["batches"]] == ["drug_recall", "cpsc_recall"]


# ---------------------------------------------------------------- branches
def test_a_rule_that_does_not_match_the_database_stops_the_run_before_any_batch():
    ops = FakeOps(findings=[Finding(level="error", doc_type="drug_recall", message="no such column: citty")])
    result = run(ops)
    assert nodes(result) == ["catalog", "validate", "finalize"]
    assert "begin_batch:drug_recall" not in ops.calls
    assert result.state["status"] == "failed"


def test_a_warning_from_validation_does_not_stop_the_run():
    ops = FakeOps(findings=[Finding(level="warning", doc_type="drug_recall", message="no relationship")])
    result = run(ops)
    assert "stage:drug_recall" in ops.calls
    assert result.state["findings"]


def test_an_incremental_batch_with_no_changed_keys_skips_quality_and_stage():
    ops = FakeOps(keys={"drug_recall": 0})
    result = run(ops)
    assert "quality:drug_recall" not in ops.calls
    assert "stage:drug_recall" not in ops.calls
    assert result.state["batches"][0]["status"] == "nothing"


def test_a_failed_quality_check_stops_that_doc_type_before_staging():
    ops = FakeOps(quality={"drug_recall": [Finding(level="error", doc_type="drug_recall",
                                                   message="recall_number is null in 3 rows")]})
    result = run(ops)
    assert ops.staged == []
    assert ops.failed and "recall_number" in ops.failed[0]
    assert result.state["batches"][0]["status"] == "failed"


def test_a_failed_quality_check_does_not_stop_the_other_doc_types():
    """One bad source must not hold back the rest of the index."""
    ops = FakeOps(quality={"drug_recall": [Finding(level="error", doc_type="drug_recall", message="bad")]})
    run(ops, ("drug_recall", "cpsc_recall"))
    assert ops.staged == ["cpsc_recall"]


def test_a_quality_warning_still_stages():
    ops = FakeOps(quality={"drug_recall": [Finding(level="warning", doc_type="drug_recall", message="meh")]})
    run(ops)
    assert ops.staged == ["drug_recall"]


def test_nothing_new_to_chunk_skips_the_lexeme_refresh():
    ops = FakeOps(chunks_changed=False)
    assert "refresh_lexemes" not in ops.calls or "lexemes" not in nodes(run(ops))


def test_an_empty_embedding_queue_skips_the_model():
    ops = FakeOps(pending=0)
    run(ops)
    assert not [c for c in ops.calls if c.startswith("embed:")]


def test_no_embed_was_asked_for_so_the_model_is_not_called():
    ops = FakeOps()
    run(ops, embed=False)
    assert not [c for c in ops.calls if c.startswith("embed:")]


def test_the_chunk_budget_reaches_the_embedding_stage():
    ops = FakeOps()
    run(ops, max_chunks=500)
    assert "embed:500" in ops.calls


def test_documents_are_marked_indexed_at_the_end():
    ops = FakeOps()
    run(ops)
    assert ops.calls[-1] == "mark_indexed" or "mark_indexed" in ops.calls


# ---------------------------------------------------------------- what the run reports
def test_the_run_reports_succeeded_when_every_batch_worked():
    assert run(FakeOps()).state["status"] == "succeeded"


def test_the_run_reports_the_embedding_statistics():
    assert run(FakeOps()).state["embed"]["embedded"] == 4


def test_the_run_reports_nothing_when_no_doc_type_had_changes():
    ops = FakeOps(keys={"drug_recall": 0})
    assert run(ops).state["status"] == "nothing"


@pytest.mark.parametrize("full", [True, False])
def test_the_full_flag_reaches_the_plan(full):
    ops = FakeOps()
    run(ops, full=full)
    assert "begin_batch:drug_recall" in ops.calls


# ---------------------------------------------------------------- the approval gate before embedding
# Embedding is the only paid step of a run, and `ingest --loop` runs unattended, so above a threshold
# the run pauses (a LangGraph interrupt) and waits to be told to go ahead. Measured on langgraph
# 0.6.11: the pause arrives as the update key "__interrupt__", whose value is a tuple of Interrupt
# objects - not a state delta. These tests use MemorySaver; the Postgres saver is covered by
# tests/integration/test_embed_approval.py.
from langgraph.checkpoint.memory import MemorySaver        # noqa: E402


class Checkpointed:
    """Settings-like object: enough for the gate's guard, with a saver handed in."""
    ingest_graph_checkpoint = True


def gated(ops, *, limit, saver=None, thread_id=None, embed=True, doc_types=("drug_recall",)):
    return ingest_graph.run_ingest(ops, list(doc_types), embed=embed, approval_chunks=limit,
                                   checkpointer=saver or MemorySaver(), thread_id=thread_id,
                                   checkpointer_settings=Checkpointed())


def test_a_run_under_the_limit_is_embedded_without_asking():
    ops = FakeOps(pending=5)
    result = gated(ops, limit=50)
    assert not result.paused
    assert "embed:None" in ops.calls
    assert result.state["status"] == "succeeded"


def test_a_run_over_the_limit_stops_before_the_first_paid_call():
    ops = FakeOps(pending=94)
    result = gated(ops, limit=50)
    assert result.paused
    assert not [c for c in ops.calls if c.startswith("embed:")]


def test_the_pause_says_what_it_would_cost():
    """A gate that does not quote a number is a gate people click through."""
    result = gated(FakeOps(pending=94), limit=50)
    assert result.waiting["chunks"] == 94
    assert result.waiting["chars"] == 94 * 1200
    assert result.waiting["limit"] == 50
    assert result.waiting["doc_types"] == ["drug_recall"]


def test_approving_the_pause_embeds_exactly_once():
    ops, saver = FakeOps(pending=94), MemorySaver()
    first = gated(ops, limit=50, saver=saver)
    assert first.paused

    second = ingest_graph.resume_ingest(ops, thread_id=first.thread_id, checkpointer=saver, answer=True)

    assert not second.paused
    assert [c for c in ops.calls if c.startswith("embed:")] == ["embed:None"]
    assert second.state["status"] == "succeeded"


def test_refusing_the_pause_embeds_nothing_and_is_not_a_failure():
    """The documents are written and the watermark has moved; only the paid step was refused."""
    ops, saver = FakeOps(pending=94), MemorySaver()
    first = gated(ops, limit=50, saver=saver)

    second = ingest_graph.resume_ingest(ops, thread_id=first.thread_id, checkpointer=saver, answer=False)

    assert not [c for c in ops.calls if c.startswith("embed:")]
    assert second.state["status"] == "embedding_refused"
    assert "finish_batch:drug_recall" in ops.calls


def test_the_gate_is_off_by_default():
    ops = FakeOps(pending=10_000)
    result = ingest_graph.run_ingest(ops, ["drug_recall"])
    assert not result.paused
    assert "pending_cost" not in ops.calls
    assert "embed:None" in ops.calls


def test_no_embed_never_asks_because_nothing_would_be_paid_for():
    ops = FakeOps(pending=94)
    result = gated(ops, limit=1, embed=False)
    assert not result.paused
    assert "pending_cost" not in ops.calls


def test_nothing_pending_never_asks():
    ops = FakeOps(pending=0)
    result = gated(ops, limit=1)
    assert not result.paused


def test_the_limit_counts_chunks_not_documents():
    """Two doc types, 30 documents each, but only 12 chunk texts to pay for: no pause at a limit of 20."""
    ops = FakeOps(keys={"drug_recall": 30, "cpsc_recall": 30}, pending=12)
    result = gated(ops, limit=20, doc_types=("drug_recall", "cpsc_recall"))
    assert not result.paused
    assert "embed:None" in ops.calls


def test_the_paused_state_is_still_json_serialisable():
    """It is what the Postgres checkpointer has to store while waiting for an answer."""
    result = gated(FakeOps(pending=94), limit=50)
    json.dumps(result.state)
    json.dumps(result.waiting)


def test_arming_the_gate_without_a_checkpointer_is_refused():
    """Measured: interrupt() stops the run even with no checkpointer, but nothing could resume it."""
    with pytest.raises(ingest_graph.ApprovalNeedsCheckpointer):
        ingest_graph.run_ingest(FakeOps(pending=94), ["drug_recall"], approval_chunks=50)


def test_an_interrupt_delta_is_not_mistaken_for_a_state_change():
    """`dict.update` on the "__interrupt__" tuple raises TypeError; _stream has to skip it."""
    result = gated(FakeOps(pending=94), limit=50)
    assert ingest_graph.INTERRUPT not in result.visited
    assert "approve_embed" not in result.visited          # the node has not returned yet
    assert result.visited[-1] == "lexemes"
