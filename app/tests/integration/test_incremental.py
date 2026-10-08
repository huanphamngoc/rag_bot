"""The incremental load, end to end against real Postgres + pgvector (no model API: FakeEmbedder).

The document types come from rules/*.yaml, and rag.document keeps every version (SCD Type 2), so the
helpers below read the *current* version of a document. The history itself is covered by
test_scd2_db.py.
"""
from __future__ import annotations

import psycopg
import pytest

from crawlerrag.ingest import pipeline
from crawlerrag.rag import retrieve

from .conftest import DSN_ADMIN, DSN_VECTOR, FakeCrawler, FakeEmbedder, _connect

LONG = " ".join(f"Sentence {i} describes the recalled power bank and how it overheats." for i in range(30))


def ingest(env, embedder=None, **kw):
    embedder = embedder or FakeEmbedder()
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall", "cpsc_recall"])
    result = pipeline.run(env["vector"], env["source"], env["settings"], doc_types,
                          embedder_factory=lambda s: embedder, **kw)
    return result, embedder


def batch(result, doc_type):
    return next(b for b in result.batches if b.doc_type == doc_type)


def head(env, source_id):
    return pipeline.source_head(env["admin"], source_id)


def watermark(env, doc_type):
    row = env["vector"].execute("SELECT change_id FROM ingest.watermark WHERE doc_type = %s", (doc_type,)).fetchone()
    return row["change_id"] if row else None


def doc(env, doc_id):
    """The current version, which is what every assertion here is about."""
    return env["vector"].execute("SELECT * FROM rag.document WHERE doc_id = %s AND is_current",
                                 (doc_id,)).fetchone()


def chunks(env, doc_id):
    return env["vector"].execute(
        "SELECT c.ord, c.text, c.embedding IS NOT NULL AS has_vector FROM rag.chunk c "
        "JOIN rag.document d USING (doc_sk) WHERE d.doc_id = %s AND d.is_current ORDER BY c.ord",
        (doc_id,)).fetchall()


def seed(env):
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0001-2017", firm="Cantrell Drug Company")
        c.drug(run, "D-0002-2017", ndcs=("0093-1234",))
    with c.run("cpsc_recall") as run:
        c.cpsc(run, "100")
        c.cpsc(run, "101", title="Long recall", description=LONG)


# ---------------------------------------------------------------- first load / nothing new
def test_first_load_is_full_and_sets_the_watermark_to_the_source_head(env):
    seed(env)
    result, emb = ingest(env)
    d, c = batch(result, "drug_recall"), batch(result, "cpsc_recall")
    assert (d.mode, d.reason, d.status) == ("full", "first load: no watermark", "succeeded")
    assert (d.stats.inserted, c.stats.inserted) == (2, 2)
    assert watermark(env, "drug_recall") == head(env, "openfda_enforcement")
    assert watermark(env, "cpsc_recall") == head(env, "cpsc_recall")
    total = env["vector"].execute("SELECT count(*) AS n, count(embedding) AS e FROM rag.chunk").fetchone()
    assert total["n"] == total["e"] > 4                     # the long CPSC recall has several chunks
    assert result.embed.embedded == len(set(emb.texts)) == total["n"]
    run = env["vector"].execute("SELECT * FROM ingest.embed_run ORDER BY run_id DESC LIMIT 1").fetchone()
    assert (run["status"], run["embedded"], run["truncated"]) == ("succeeded", total["n"], 0)
    assert run["input_tokens"] == sum(len(t.split()) for t in emb.texts)
    assert run["input_chars"] == sum(len(t) for t in emb.texts)
    assert env["vector"].execute("SELECT count(*) AS n FROM rag.document WHERE indexed_at IS NULL").fetchone()["n"] == 0
    assert "National drug codes involved: 0093-1234" in doc(env, "drug_recall:D-0002-2017")["body"]


def test_no_new_change_means_nothing_is_read_or_embedded(env):
    seed(env)
    ingest(env)
    result, emb = ingest(env)
    assert [b.status for b in result.batches] == ["nothing", "nothing"]
    assert emb.requests == 0 and result.embed is None


def test_identical_chunk_text_is_sent_to_the_model_once(env):
    c = env["crawler"]
    with c.run("cpsc_recall") as run:            # two recalls whose documents render to the same body
        c.cpsc(run, "200", title="Twin")
        c.cpsc(run, "201", title="Twin")
    with env["admin"].transaction():             # same recall number too, so the text is byte-identical
        env["admin"].execute("UPDATE retail.cpsc_recall SET recall_number = '24200' WHERE recall_id IN ('200', '201')")
    result, emb = ingest(env)
    pair = env["vector"].execute("SELECT count(DISTINCT text_hash) AS texts, count(*) AS chunks, "
                                 "count(embedding) AS vectors FROM rag.chunk").fetchone()
    assert pair["texts"] < pair["chunks"] == pair["vectors"]
    assert len(emb.texts) == len(set(emb.texts)) == pair["texts"]
    assert result.embed.embedded == pair["texts"] and result.embed.reused == pair["chunks"] - pair["texts"]


# ---------------------------------------------------------------- updates
def test_an_update_rechunks_only_that_document_and_keeps_vectors_of_unchanged_chunks(env):
    seed(env)
    ingest(env)
    before = chunks(env, "cpsc_recall:101")
    c = env["crawler"]
    with c.run("cpsc_recall") as run:
        c.cpsc(run, "101", title="Long recall", description=LONG, remedies=("Refund", "Free replacement"),
               change="update")
    result, emb = ingest(env)
    b = batch(result, "cpsc_recall")
    assert (b.mode, b.keys, b.stats.updated, b.stats.unchanged) == ("incremental", 1, 1, 0)
    assert batch(result, "drug_recall").status == "nothing"
    after = chunks(env, "cpsc_recall:101")
    unchanged_texts = {r["text"] for r in before} & {r["text"] for r in after}
    assert unchanged_texts and b.stats.vectors_carried == len(unchanged_texts)
    # only text that did not exist before is sent to the model
    assert set(emb.texts) == {r["text"] for r in after} - {r["text"] for r in before}
    assert all(r["has_vector"] for r in after)
    assert "Free replacement" in doc(env, "cpsc_recall:101")["body"]


def test_a_child_row_change_reaches_the_parent_document(env):
    seed(env)
    ingest(env)
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0001-2017", firm="Cantrell Drug Company", ndcs=("51662-1234",), change="update")
    ingest(env)
    assert "51662-1234" in doc(env, "drug_recall:D-0001-2017")["body"]


def test_a_new_url_is_applied_without_new_chunks_or_embeddings(env):
    seed(env)
    ingest(env)
    c = env["crawler"]
    with c.run("cpsc_recall") as run:
        c.cpsc(run, "100", url="https://www.cpsc.gov/Recalls/1-corrected", change="update")
    result, emb = ingest(env)
    b = batch(result, "cpsc_recall")
    assert (b.stats.refreshed, b.stats.updated, b.stats.chunks_added) == (1, 0, 0)
    assert emb.requests == 0
    assert doc(env, "cpsc_recall:100")["url"] == "https://www.cpsc.gov/Recalls/1-corrected"


def test_an_update_that_does_not_change_the_document_costs_nothing(env):
    seed(env)
    ingest(env)
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:      # logged as an update, rendered text identical
        c.drug(run, "D-0002-2017", ndcs=("0093-1234",), change="update")
    result, emb = ingest(env)
    b = batch(result, "drug_recall")
    assert (b.keys, b.stats.unchanged, b.stats.updated) == (1, 1, 0)
    assert emb.requests == 0


def test_a_new_record_in_a_later_run_is_added(env):
    seed(env)
    ingest(env)
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0100-2024", firm="Lannett Company", year=2024)
    result, _ = ingest(env)
    assert batch(result, "drug_recall").stats.inserted == 1
    assert doc(env, "drug_recall:D-0100-2024")["metadata"]["year"] == 2024


# ---------------------------------------------------------------- soft deletes
def test_deactivate_hides_a_document_and_reactivate_brings_it_back_without_embedding(env):
    seed(env)
    ingest(env)
    c, settings = env["crawler"], env["settings"]
    with c.run("openfda_enforcement") as run:
        c.set_active(run, "openfda_enforcement", "D-0001-2017", False)
    result, _ = ingest(env)
    assert batch(result, "drug_recall").stats.deactivated == 1
    hits = retrieve.search(env["vector"], settings, FakeEmbedder(), "Cantrell Drug Company", top_k=10).hits
    assert "drug_recall:D-0001-2017" not in {h.doc_id for h in hits}

    with c.run("openfda_enforcement") as run:
        c.set_active(run, "openfda_enforcement", "D-0001-2017", True)
    result, emb = ingest(env)
    assert batch(result, "drug_recall").stats.reactivated == 1 and emb.requests == 0
    hits = retrieve.search(env["vector"], settings, FakeEmbedder(), "Cantrell Drug Company", top_k=10).hits
    assert "drug_recall:D-0001-2017" in {h.doc_id for h in hits}


def test_a_changed_key_whose_row_is_gone_is_deactivated(env):
    seed(env)
    ingest(env)
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        with env["admin"].transaction():
            env["admin"].execute("DELETE FROM drug.recall WHERE recall_number = 'D-0002-2017'")
            c._log(run, "openfda_enforcement", "D-0002-2017", "deactivate")
    result, _ = ingest(env)
    assert batch(result, "drug_recall").stats.deactivated == 1
    assert doc(env, "drug_recall:D-0002-2017")["is_active"] is False


# ---------------------------------------------------------------- consistency
def test_extract_reads_one_read_only_repeatable_read_snapshot(env, monkeypatch):
    """A change committed while the batch is reading is not half-seen: it waits for the next run."""
    seed(env)
    ingest(env)
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0003-2017", firm="First")
    seen = {}
    real_fetch = pipeline.fetch_rows

    def fetch_with_a_concurrent_commit(conn, doc_type, keys):
        seen[doc_type.doc_type] = conn.execute(
            "SELECT current_setting('transaction_isolation') AS iso, "
            "current_setting('transaction_read_only') AS ro").fetchone()
        if doc_type.doc_type == "drug_recall" and "late" not in seen:
            seen["late"] = True
            other = FakeCrawler(_connect(DSN_ADMIN))
            with other.run("openfda_enforcement") as late_run:
                other.drug(late_run, "D-0004-2017", firm="Late commit")
            other.conn.close()
        return real_fetch(conn, doc_type, keys)

    monkeypatch.setattr(pipeline, "fetch_rows", fetch_with_a_concurrent_commit)
    result, _ = ingest(env)
    b = batch(result, "drug_recall")
    assert seen["drug_recall"] == {"iso": "repeatable read", "ro": "on"}
    assert b.keys == 1 and doc(env, "drug_recall:D-0003-2017") is not None
    assert doc(env, "drug_recall:D-0004-2017") is None                 # committed after the snapshot
    assert watermark(env, "drug_recall") == b.to_change_id < head(env, "openfda_enforcement")

    monkeypatch.setattr(pipeline, "fetch_rows", real_fetch)
    result, _ = ingest(env)
    assert batch(result, "drug_recall").stats.inserted == 1
    assert doc(env, "drug_recall:D-0004-2017") is not None


def test_a_crash_mid_batch_keeps_the_watermark_and_the_rerun_finishes_cheaply(env, monkeypatch):
    seed(env)
    env["settings"].ingest_page_docs = 1                 # one document per transaction
    real_stage = pipeline.stage_page
    calls = {"n": 0}

    def crash_on_second_page(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real_stage(*args, **kw)

    monkeypatch.setattr(pipeline, "stage_page", crash_on_second_page)
    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest(env)
    assert watermark(env, "drug_recall") is None                       # nothing claimed as done
    status = env["vector"].execute("SELECT status, error FROM ingest.batch ORDER BY batch_id DESC LIMIT 1").fetchone()
    assert status["status"] == "failed" and "simulated crash" in status["error"]
    assert env["vector"].execute("SELECT count(*) AS n FROM rag.document WHERE is_current"
                                 ).fetchone()["n"] == 1   # page 1 committed

    monkeypatch.setattr(pipeline, "stage_page", real_stage)
    result, _ = ingest(env)
    d = batch(result, "drug_recall")
    assert (d.stats.inserted, d.stats.unchanged) == (1, 1)             # page 1 was not redone
    assert watermark(env, "drug_recall") == head(env, "openfda_enforcement")


def test_a_model_outage_does_not_hold_back_documents_or_watermark(env):
    seed(env)
    result, _ = ingest(env, embedder=FakeEmbedder(fail=True))
    assert result.embed.status == "failed" and result.embed.embedded == 0
    assert watermark(env, "drug_recall") == head(env, "openfda_enforcement")
    pending = env["vector"].execute("SELECT count(*) AS n FROM rag.chunk WHERE embedding IS NULL").fetchone()["n"]
    assert pending > 0
    run = env["vector"].execute("SELECT status, error FROM ingest.embed_run ORDER BY run_id DESC LIMIT 1").fetchone()
    assert run["status"] == "failed" and "fake outage" in run["error"]

    result, emb = ingest(env)
    assert [b.status for b in result.batches] == ["nothing", "nothing"]
    assert result.embed.embedded == pending and result.embed.pending_after == 0


def test_a_per_minute_quota_is_waited_out_instead_of_failing_the_run(env):
    from crawlerrag.ingest import embed as embed_mod
    seed(env)
    ingest(env, embed=False)
    waits = []
    emb = FakeEmbedder(quota_refusals=2)
    stats = embed_mod.embed_pending(env["vector"], emb, quota_wait_s=65, sleep=waits.append)
    assert (stats.status, stats.quota_waits, stats.pending_after) == ("succeeded", 2, 0)
    assert waits == [65, 65]
    run = env["vector"].execute("SELECT quota_waits, status FROM ingest.embed_run ORDER BY run_id DESC LIMIT 1").fetchone()
    assert (run["quota_waits"], run["status"]) == (2, "succeeded")


def test_a_quota_that_never_returns_stops_the_run_with_its_progress_kept(env):
    from crawlerrag.ingest import embed as embed_mod
    seed(env)
    ingest(env, embed=False)
    stats = embed_mod.embed_pending(env["vector"], FakeEmbedder(quota_refusals=99), max_quota_waits=3,
                                    sleep=lambda s: None)
    assert (stats.status, stats.quota_waits, stats.embedded) == ("failed", 3, 0)
    assert "429" in stats.errors[0] and stats.pending_after > 0


def test_max_chunks_stops_the_embedding_and_the_next_run_continues(env):
    seed(env)
    result, _ = ingest(env, max_chunks=2)
    assert (result.embed.embedded, result.embed.status) == (2, "stopped") and result.embed.pending_after > 0
    result, _ = ingest(env)
    assert result.embed.pending_after == 0


# ---------------------------------------------------------------- full reconcile
def test_full_reconcile_repairs_a_change_the_log_never_saw(env):
    seed(env)
    ingest(env)
    c = env["crawler"]
    c.drug(None, "D-0001-2017", firm="Cantrell Drug Company", reason="Rebuilt reason text", log=False)
    result, _ = ingest(env)
    assert batch(result, "drug_recall").status == "nothing"            # the log has no entry
    result, emb = ingest(env, full=True)
    d = batch(result, "drug_recall")
    assert (d.mode, d.stats.updated, d.stats.unchanged) == ("full", 1, 1)
    d1 = {r["text"] for r in chunks(env, "drug_recall:D-0001-2017")}
    assert emb.texts and set(emb.texts) <= d1                          # only the repaired document's new text
    assert "Rebuilt reason text" in doc(env, "drug_recall:D-0001-2017")["body"]


def test_changed_chunk_settings_trigger_a_full_rebuild(env):
    seed(env)
    ingest(env)
    env["settings"].rag_chunk_chars = 300
    result, emb = ingest(env)
    c = batch(result, "cpsc_recall")
    assert c.mode == "full" and "chunk=400/60 -> " in c.reason and c.reason.endswith("chunk=300/60")
    # Only the documents whose chunk boundaries actually move are re-chunked: the short recall comes out
    # as one chunk at 400 and at 300 characters, so it is left alone.
    assert (c.stats.updated, c.stats.unchanged) == (1, 1) and c.stats.chunks_added > 0
    new_texts = {r["text"] for r in env["vector"].execute("SELECT text FROM rag.chunk").fetchall()}
    assert set(emb.texts) <= new_texts and len(emb.texts) < len(new_texts)   # identical chunks kept their vector
    assert env["vector"].execute("SELECT count(*) AS n FROM rag.chunk WHERE embedding IS NULL").fetchone()["n"] == 0
    sig = env["vector"].execute("SELECT signature FROM ingest.watermark WHERE doc_type = 'cpsc_recall'").fetchone()
    assert sig["signature"].startswith("v1|rule=") and sig["signature"].endswith("|chunk=300/60")


# ---------------------------------------------------------------- guards
def test_a_watermark_beyond_the_source_is_refused(env):
    seed(env)
    ingest(env)
    env["vector"].execute("UPDATE ingest.watermark SET change_id = 10000")
    with pytest.raises(pipeline.WatermarkAhead, match="another crawler database"):
        ingest(env)
    assert env["vector"].execute("SELECT status FROM ingest.batch ORDER BY batch_id DESC LIMIT 1"
                                 ).fetchone()["status"] == "failed"


def test_two_ingests_never_run_at_once(env):
    seed(env)
    other = psycopg.connect(DSN_VECTOR, autocommit=True)
    try:
        other.execute("SELECT pg_advisory_lock(hashtext(%s))", (pipeline.LOCK_NAME,))
        with pytest.raises(pipeline.AlreadyRunning):
            ingest(env)
    finally:
        other.close()


def test_the_source_role_cannot_write(env):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        env["source"].execute("INSERT INTO crawl.crawl_run (source_id) VALUES ('x')")
