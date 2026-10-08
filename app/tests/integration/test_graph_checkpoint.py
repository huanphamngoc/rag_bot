"""The LangGraph checkpointer, against the real vector database.

What it adds over the watermark: the watermark says which *source changes* are already reflected in
the documents, which is what makes a re-run cheap. It says nothing about where in the flow a run
stopped. The checkpointer does: the state after each node is stored, so a run that died in the
embedding stage resumes at that node instead of walking the whole graph again.

The two are complementary and both are needed - which is why these tests check that a resumed run
does *not* redo the staging, and that the watermark is still the thing that decides what gets read.
"""
from __future__ import annotations

import pytest

from crawlerrag.ingest import graph as ingest_graph
from crawlerrag.ingest import pipeline

from .conftest import FakeEmbedder


@pytest.fixture(autouse=True)
def checkpointing_on(env):
    """The other integration tests run without it; these are about it."""
    env["settings"].ingest_graph_checkpoint = True
    return env


def seed(env):
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0001-2017", firm="Cantrell Drug Company")
        c.drug(run, "D-0002-2017", firm="Acme Pharma")


def ops(env, embedder=None):
    return pipeline.PostgresOps(env["vector"], env["source"], env["settings"],
                                pipeline.doc_types_for(env["settings"], ["drug_recall"]),
                                embedder_factory=lambda s: embedder or FakeEmbedder())


def checkpoints(env, thread_id):
    return env["vector"].execute(
        "SELECT checkpoint_id FROM graph.checkpoints WHERE thread_id = %s ORDER BY checkpoint_id",
        (thread_id,)).fetchall()


# ---------------------------------------------------------------- setup
def test_the_checkpointer_keeps_its_tables_out_of_the_rag_schemas(env):
    with ingest_graph.postgres_checkpointer(env["settings"]) as saver:
        assert saver is not None
    rows = env["vector"].execute(
        "SELECT table_schema FROM information_schema.tables WHERE table_name = 'checkpoints'").fetchall()
    assert [r["table_schema"] for r in rows] == ["graph"]


def test_a_run_records_checkpoints_for_its_thread(env):
    seed(env)
    result = ingest_graph.run_ingest(ops(env), ["drug_recall"], thread_id="test-1",
                                    checkpointer_settings=env["settings"])
    assert result.state["status"] == "succeeded"
    assert len(checkpoints(env, "test-1")) > 1


def test_two_runs_keep_their_own_threads_apart(env):
    seed(env)
    ingest_graph.run_ingest(ops(env), ["drug_recall"], thread_id="a", checkpointer_settings=env["settings"])
    ingest_graph.run_ingest(ops(env), ["drug_recall"], thread_id="b", checkpointer_settings=env["settings"])
    assert checkpoints(env, "a") and checkpoints(env, "b")


def test_the_checkpointer_can_be_turned_off(env):
    seed(env)
    env["settings"].ingest_graph_checkpoint = False
    with ingest_graph.postgres_checkpointer(env["settings"]) as saver:
        assert saver is None
    result = ingest_graph.run_ingest(ops(env), ["drug_recall"], checkpointer_settings=env["settings"])
    assert result.state["status"] == "succeeded"


# ---------------------------------------------------------------- resuming
def test_a_run_that_died_in_the_embedding_stage_resumes_there(env):
    """The documents were already staged and committed: the resumed run must not stage them again."""
    seed(env)
    broken = FakeEmbedder()
    calls = {"n": 0}
    real_embed = broken.embed

    def fail_once(texts, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated crash in the embedding stage")
        return real_embed(texts, **kw)

    broken.embed = fail_once
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest_graph.run_ingest(ops(env, broken), ["drug_recall"], thread_id="resume",
                                checkpointer_settings=env["settings"])
    staged = env["vector"].execute("SELECT count(*) AS n FROM rag.document").fetchone()["n"]
    assert staged == 2
    pending = env["vector"].execute("SELECT count(*) AS n FROM rag.chunk WHERE embedding IS NULL").fetchone()
    assert pending["n"] > 0

    resumed = ingest_graph.resume_ingest(ops(env), thread_id="resume", checkpointer_settings=env["settings"])
    assert resumed.visited == ["embed", "finalize"]
    assert env["vector"].execute("SELECT count(*) AS n FROM rag.chunk WHERE embedding IS NULL"
                                 ).fetchone()["n"] == 0
    assert env["vector"].execute("SELECT count(*) AS n FROM ingest.batch").fetchone()["n"] == 1


def test_resuming_a_thread_that_finished_does_nothing(env):
    seed(env)
    ingest_graph.run_ingest(ops(env), ["drug_recall"], thread_id="done",
                            checkpointer_settings=env["settings"])
    resumed = ingest_graph.resume_ingest(ops(env), thread_id="done", checkpointer_settings=env["settings"])
    assert resumed.visited == []


def test_resuming_an_unknown_thread_is_refused(env):
    with pytest.raises(ingest_graph.NoSuchRun):
        ingest_graph.resume_ingest(ops(env), thread_id="never-ran", checkpointer_settings=env["settings"])


# ---------------------------------------------------------------- the watermark is still in charge
def test_the_watermark_not_the_checkpoint_decides_what_is_read(env):
    seed(env)
    ingest_graph.run_ingest(ops(env), ["drug_recall"], thread_id="first",
                            checkpointer_settings=env["settings"])
    result = ingest_graph.run_ingest(ops(env), ["drug_recall"], thread_id="second",
                                     checkpointer_settings=env["settings"])
    assert result.state["batches"][0]["status"] == "nothing"
