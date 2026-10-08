"""The approval gate in front of the embedding step, against real Postgres and the real checkpointer.

Embedding is the only paid step of an ingestion run, and the scheduler runs it unattended, so above
``INGEST_EMBED_APPROVAL_CHUNKS`` distinct chunk texts the run stops and waits (a LangGraph interrupt
stored by ``PostgresSaver`` in the ``graph`` schema).

What these tests are really about is that the pause is *safe*: the documents and the watermark are
already committed when it happens, so waiting - or never answering at all - loses no work and leaves
nothing half-written. ``tests/test_ingest_graph.py`` covers the branch logic with MemorySaver.
"""
from __future__ import annotations

import pytest

from crawlerrag.ingest import pipeline
from crawlerrag.ingest.graph import NoSuchRun

from .conftest import FakeEmbedder
from .test_incremental import chunks, doc, head, seed, watermark


def armed(env, limit: int):
    """The gate needs the checkpointer: a pause with nowhere to be stored could not be answered."""
    env["settings"].ingest_graph_checkpoint = True
    env["settings"].ingest_embed_approval_chunks = limit
    return env


def ingest(env, embedder=None, **kw):
    embedder = embedder or FakeEmbedder()
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall", "cpsc_recall"])
    result = pipeline.run(env["vector"], env["source"], env["settings"], doc_types,
                          embedder_factory=lambda s: embedder, **kw)
    return result, embedder


def answer(env, thread_id, *, approved, embedder=None):
    embedder = embedder or FakeEmbedder()
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall", "cpsc_recall"])
    result = pipeline.answer_approval(env["vector"], env["source"], env["settings"], doc_types,
                                      thread_id=thread_id, approved=approved,
                                      embedder_factory=lambda s: embedder)
    return result, embedder


def pending(env) -> int:
    return env["vector"].execute(
        "SELECT count(*) AS n FROM rag.chunk WHERE embedding IS NULL").fetchone()["n"]


# ---------------------------------------------------------------- the pause itself
def test_a_first_load_over_the_limit_stops_before_embedding(env):
    seed(env)
    result, embedder = ingest(armed(env, 1))

    assert result.paused
    assert embedder.requests == 0


def test_the_pause_quotes_the_cost_it_would_have_paid(env):
    seed(env)
    result, _ = ingest(armed(env, 1))

    assert result.waiting["chunks"] == pending(env)
    assert result.waiting["chars"] > 0
    assert result.waiting["limit"] == 1


def test_the_documents_are_already_written_when_it_pauses(env):
    """The pause is in front of the paid step only - staging has already committed."""
    seed(env)
    ingest(armed(env, 1))

    assert doc(env, "drug_recall:D-0001-2017") is not None
    assert doc(env, "cpsc_recall:100") is not None


def test_the_watermark_has_already_moved_when_it_pauses(env):
    """So an unanswered pause does not make the next run read the same window again."""
    seed(env)
    ingest(armed(env, 1))

    assert watermark(env, "drug_recall") == head(env, "openfda_enforcement")
    assert watermark(env, "cpsc_recall") == head(env, "cpsc_recall")


def test_the_chunks_are_there_but_have_no_vector_yet(env):
    seed(env)
    ingest(armed(env, 1))

    rows = chunks(env, "drug_recall:D-0001-2017")
    assert rows and not any(r["has_vector"] for r in rows)


# ---------------------------------------------------------------- answering it
def test_approving_embeds_the_waiting_chunks(env):
    seed(env)
    first, _ = ingest(armed(env, 1))

    second, embedder = answer(env, first.thread_id, approved=True)

    assert not second.paused
    assert embedder.requests > 0
    assert pending(env) == 0


def test_approving_does_not_read_the_source_again(env):
    """Resuming continues at the gate; extract and stage are behind the checkpoint."""
    seed(env)
    first, _ = ingest(armed(env, 1))
    before = env["vector"].execute("SELECT count(*) AS n FROM ingest.batch").fetchone()["n"]

    answer(env, first.thread_id, approved=True)

    after = env["vector"].execute("SELECT count(*) AS n FROM ingest.batch").fetchone()["n"]
    assert after == before


def test_refusing_leaves_the_chunks_unembedded(env):
    seed(env)
    first, _ = ingest(armed(env, 1))

    second, embedder = answer(env, first.thread_id, approved=False)

    assert embedder.requests == 0
    assert pending(env) > 0
    assert second.status == "embedding_refused"


def test_a_refusal_is_not_a_failed_batch(env):
    """The load succeeded; only the paid step was declined."""
    seed(env)
    first, _ = ingest(armed(env, 1))
    answer(env, first.thread_id, approved=False)

    rows = env["vector"].execute("SELECT DISTINCT status FROM ingest.batch").fetchall()
    assert {r["status"] for r in rows} == {"succeeded"}


def test_after_a_refusal_the_embed_command_is_the_way_through(env):
    """`crawlerrag embed` is the deliberate escape hatch, so a refusal is never a dead end."""
    seed(env)
    first, _ = ingest(armed(env, 1))
    answer(env, first.thread_id, approved=False)

    from crawlerrag.ingest import embed as embed_mod
    embedder = FakeEmbedder()
    embed_mod.embed_pending(env["vector"], embedder)

    assert pending(env) == 0


def test_an_unknown_thread_is_refused(env):
    seed(env)
    ingest(armed(env, 1))

    with pytest.raises(NoSuchRun):
        answer(env, "0" * 32, approved=True)


def test_two_paused_runs_keep_their_own_threads(env):
    """A second run pauses on its own thread; answering one does not touch the other."""
    seed(env)
    first, _ = ingest(armed(env, 1))
    c = env["crawler"]
    with c.run("openfda_enforcement") as run_id:
        c.drug(run_id, "D-0099-2017", firm="Second Wave Pharma")
    second, _ = ingest(env)

    assert second.paused
    assert second.thread_id != first.thread_id

    answer(env, second.thread_id, approved=True)
    assert pending(env) == 0


# ---------------------------------------------------------------- the gate out of the way
def test_under_the_limit_nothing_is_asked(env):
    seed(env)
    result, embedder = ingest(armed(env, 10_000))

    assert not result.paused
    assert embedder.requests > 0
    assert pending(env) == 0


def test_the_limit_is_measured_in_chunk_texts_actually_sent(env):
    """Two chunks are pending, but their text is byte-identical: only one is bought and the other is
    copied (``reuse_embeddings``). So a limit of 1 does not pause - the gate counts what gets paid for,
    not what is waiting. Quoting the bigger number would teach people to click through the gate."""
    c = env["crawler"]
    with c.run("cpsc_recall") as run_id:
        c.cpsc(run_id, "300", title="Twin")
        c.cpsc(run_id, "301", title="Twin")
    with env["admin"].transaction():          # the recall number is in the body, so it has to match too
        env["admin"].execute("UPDATE retail.cpsc_recall SET recall_number = '24300' "
                             "WHERE recall_id IN ('300', '301')")
    result, _ = ingest(armed(env, 1))

    assert not result.paused
    assert result.embed.embedded == 1 and result.embed.reused == 1
