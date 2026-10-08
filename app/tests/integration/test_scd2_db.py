"""SCD Type 2 on rag.document, against real Postgres + pgvector.

Every version of a document is kept: one row per version with ``version``, ``valid_from``, ``valid_to``
and ``is_current``, and the chunks hang off the version's surrogate key ``doc_sk``. Retrieval only ever
sees the current version of an active document.

The expensive part is the vectors, so the two cheap paths are tested hardest:

* a changed document re-chunks and carries the vector of every chunk whose text did not change;
* a document that was only deactivated or reactivated has identical text, so its chunks are *moved* to
  the new version - the same ``chunk_id`` rows, the same vectors, no model call.
"""
from __future__ import annotations

import pytest

from crawlerrag.ingest import pipeline
from crawlerrag.rag import retrieve

from .conftest import FakeEmbedder

LONG = " ".join(f"Sentence {i} describes the recalled power bank and how it overheats." for i in range(30))


def ingest(env, embedder=None, **kw):
    embedder = embedder or FakeEmbedder()
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall", "cpsc_recall"])
    result = pipeline.run(env["vector"], env["source"], env["settings"], doc_types,
                          embedder_factory=lambda s: embedder, **kw)
    return result, embedder


def batch(result, doc_type):
    return next(b for b in result.batches if b.doc_type == doc_type)


def versions(env, doc_id):
    return env["vector"].execute(
        "SELECT doc_sk, version, is_current, valid_from, valid_to, change_reason, is_active, content_hash, url "
        "FROM rag.document WHERE doc_id = %s ORDER BY version", (doc_id,)).fetchall()


def current(env, doc_id):
    return env["vector"].execute("SELECT * FROM rag.document WHERE doc_id = %s AND is_current",
                                 (doc_id,)).fetchone()


def chunk_rows(env, doc_id):
    return env["vector"].execute(
        "SELECT c.chunk_id, c.ord, c.text, c.embedding IS NOT NULL AS has_vector "
        "FROM rag.chunk c JOIN rag.document d USING (doc_sk) "
        "WHERE d.doc_id = %s AND d.is_current ORDER BY c.ord", (doc_id,)).fetchall()


def seed(env):
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0001-2017", firm="Cantrell Drug Company")
    with c.run("cpsc_recall") as run:
        c.cpsc(run, "101", title="Long recall", description=LONG)


# ---------------------------------------------------------------- version 1
def test_a_first_load_writes_version_one_as_the_current_version(env):
    seed(env)
    ingest(env)
    rows = versions(env, "drug_recall:D-0001-2017")
    assert len(rows) == 1
    assert (rows[0]["version"], rows[0]["is_current"], rows[0]["valid_to"]) == (1, True, None)
    assert rows[0]["change_reason"] == "first version"


def test_the_first_version_starts_being_valid_when_it_was_loaded(env):
    seed(env)
    ingest(env)
    row = current(env, "drug_recall:D-0001-2017")
    assert row["valid_from"] is not None and row["valid_from"] <= row["updated_at"]


def test_the_version_records_which_source_change_it_reflects(env):
    seed(env)
    result, _ = ingest(env)
    row = current(env, "drug_recall:D-0001-2017")
    assert row["source_change_id"] == batch(result, "drug_recall").to_change_id
    assert row["batch_id"] == batch(result, "drug_recall").batch_id


# ---------------------------------------------------------------- a changed document
def test_changed_text_closes_the_old_version_and_opens_a_new_one(env):
    seed(env)
    ingest(env)
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].drug(run, "D-0001-2017", firm="Cantrell Drug Company",
                            reason="Contamination with particulate matter", change="update")
    ingest(env)
    rows = versions(env, "drug_recall:D-0001-2017")
    assert [r["version"] for r in rows] == [1, 2]
    assert [r["is_current"] for r in rows] == [False, True]
    assert rows[0]["valid_to"] == rows[1]["valid_from"]          # no gap, no overlap
    assert "text" in rows[1]["change_reason"]


def test_the_old_version_keeps_the_text_it_had(env):
    seed(env)
    ingest(env)
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].drug(run, "D-0001-2017", firm="Cantrell Drug Company",
                            reason="Contamination with particulate matter", change="update")
    ingest(env)
    old = env["vector"].execute("SELECT body FROM rag.document WHERE doc_id = %s AND version = 1",
                                ("drug_recall:D-0001-2017",)).fetchone()
    assert "Lack of sterility assurance" in old["body"]
    assert "Contamination" in current(env, "drug_recall:D-0001-2017")["body"]


def test_the_batch_counts_the_version_it_created(env):
    seed(env)
    ingest(env)
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].drug(run, "D-0001-2017", firm="Cantrell", change="update")
    result, _ = ingest(env)
    assert batch(result, "drug_recall").stats.versions_created == 1


def test_chunks_follow_the_new_version_and_carry_the_vectors_they_can(env):
    seed(env)
    ingest(env)
    before = chunk_rows(env, "cpsc_recall:101")
    with env["crawler"].run("cpsc_recall") as run:
        env["crawler"].cpsc(run, "101", title="Long recall", description=LONG,
                            remedies=("Refund", "Free replacement"), change="update")
    result, emb = ingest(env)
    after = chunk_rows(env, "cpsc_recall:101")
    assert all(r["has_vector"] for r in after)
    carried = {r["text"] for r in before} & {r["text"] for r in after}
    assert carried and batch(result, "cpsc_recall").stats.vectors_carried == len(carried)
    assert set(emb.texts) == {r["text"] for r in after} - {r["text"] for r in before}


def test_no_chunk_is_left_on_a_closed_version(env):
    """The invariant retrieval relies on: chunks exist only for the current version."""
    seed(env)
    ingest(env)
    with env["crawler"].run("cpsc_recall") as run:
        env["crawler"].cpsc(run, "101", title="Long recall", description=LONG + " More.", change="update")
    ingest(env)
    stray = env["vector"].execute(
        "SELECT count(*) AS n FROM rag.chunk c JOIN rag.document d USING (doc_sk) WHERE NOT d.is_current"
    ).fetchone()
    assert stray["n"] == 0


# ---------------------------------------------------------------- an untracked change
def test_a_new_url_updates_the_current_version_without_a_new_one(env):
    seed(env)
    ingest(env)
    with env["crawler"].run("cpsc_recall") as run:
        env["crawler"].cpsc(run, "101", title="Long recall", description=LONG,
                            url="https://www.cpsc.gov/Recalls/1-corrected", change="update")
    result, emb = ingest(env)
    rows = versions(env, "cpsc_recall:101")
    assert [r["version"] for r in rows] == [1]
    assert rows[0]["url"] == "https://www.cpsc.gov/Recalls/1-corrected"
    assert (batch(result, "cpsc_recall").stats.refreshed, emb.requests) == (1, 0)


# ---------------------------------------------------------------- soft delete
def test_a_deactivation_opens_a_version_and_moves_the_chunks_untouched(env):
    seed(env)
    ingest(env)
    before = chunk_rows(env, "cpsc_recall:101")
    with env["crawler"].run("cpsc_recall") as run:
        env["crawler"].set_active(run, "cpsc_recall", "101", False)
    result, emb = ingest(env)
    rows = versions(env, "cpsc_recall:101")
    assert [(r["version"], r["is_current"], r["is_active"]) for r in rows] == [(1, False, True), (2, True, False)]
    assert rows[1]["change_reason"] == "deactivated at the source"
    after = chunk_rows(env, "cpsc_recall:101")
    assert [r["chunk_id"] for r in after] == [r["chunk_id"] for r in before]   # the same rows, moved
    assert emb.requests == 0
    assert batch(result, "cpsc_recall").stats.chunks_moved == len(before)


def test_the_text_does_not_change_when_only_the_flag_does(env):
    seed(env)
    ingest(env)
    before = current(env, "cpsc_recall:101")["content_hash"]
    with env["crawler"].run("cpsc_recall") as run:
        env["crawler"].set_active(run, "cpsc_recall", "101", False)
    ingest(env)
    assert current(env, "cpsc_recall:101")["content_hash"] == before


def test_a_reactivation_opens_a_third_version(env):
    seed(env)
    ingest(env)
    c = env["crawler"]
    with c.run("cpsc_recall") as run:
        c.set_active(run, "cpsc_recall", "101", False)
    ingest(env)
    with c.run("cpsc_recall") as run:
        c.set_active(run, "cpsc_recall", "101", True)
    result, emb = ingest(env)
    rows = versions(env, "cpsc_recall:101")
    assert [r["version"] for r in rows] == [1, 2, 3]
    assert rows[2]["change_reason"] == "reactivated at the source"
    assert emb.requests == 0


def test_a_deactivated_document_is_not_retrieved(env):
    seed(env)
    ingest(env)
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].set_active(run, "openfda_enforcement", "D-0001-2017", False)
    ingest(env)
    hits = retrieve.search(env["vector"], env["settings"], FakeEmbedder(), "Cantrell Drug Company",
                           top_k=10).hits
    assert "drug_recall:D-0001-2017" not in {h.doc_id for h in hits}


def test_only_the_current_version_is_retrieved(env):
    seed(env)
    ingest(env)
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].drug(run, "D-0001-2017", firm="Cantrell Drug Company",
                            reason="Contamination with particulate matter", change="update")
    ingest(env)
    hits = retrieve.search(env["vector"], env["settings"], FakeEmbedder(), "Contamination", top_k=10).hits
    assert hits and all("Lack of sterility" not in h.text for h in hits)


# ---------------------------------------------------------------- invariants the database enforces
def test_a_document_can_have_only_one_current_version(env):
    import psycopg
    seed(env)
    ingest(env)
    row = current(env, "drug_recall:D-0001-2017")
    with pytest.raises(psycopg.errors.UniqueViolation):
        env["vector"].execute(
            "INSERT INTO rag.document (doc_id, doc_type, source_id, title, body, content_hash, version, "
            "is_current, valid_from, change_reason) VALUES (%s, %s, %s, 'x', 'y', %s, 99, true, now(), 'test')",
            (row["doc_id"], row["doc_type"], row["source_id"], row["content_hash"]))


def test_two_versions_cannot_share_a_number(env):
    import psycopg
    seed(env)
    ingest(env)
    row = current(env, "drug_recall:D-0001-2017")
    with pytest.raises(psycopg.errors.UniqueViolation):
        env["vector"].execute(
            "INSERT INTO rag.document (doc_id, doc_type, source_id, title, body, content_hash, version, "
            "is_current, valid_from, valid_to, change_reason) "
            "VALUES (%s, %s, %s, 'x', 'y', %s, 1, false, now(), now(), 'test')",
            (row["doc_id"], row["doc_type"], row["source_id"], row["content_hash"]))


# ---------------------------------------------------------------- re-chunking is not history
def test_changed_chunk_settings_do_not_create_versions(env):
    seed(env)
    ingest(env)
    env["settings"].rag_chunk_chars = 300
    result, _ = ingest(env)
    assert batch(result, "cpsc_recall").mode == "full"
    assert [r["version"] for r in versions(env, "cpsc_recall:101")] == [1]
    assert batch(result, "cpsc_recall").stats.versions_created == 0
    assert batch(result, "cpsc_recall").stats.chunks_added > 0


# ---------------------------------------------------------------- history for people
def test_the_history_of_a_document_can_be_read_back(env):
    seed(env)
    ingest(env)
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].drug(run, "D-0001-2017", firm="Cantrell Drug Company", reason="Other", change="update")
    ingest(env)
    rows = pipeline.document_history(env["vector"], "drug_recall:D-0001-2017")
    assert [r["version"] for r in rows] == [1, 2]
    assert rows[0]["valid_to"] is not None and rows[1]["valid_to"] is None
