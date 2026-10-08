"""Migrating an index that already holds paid-for vectors.

This is the test that answers "is it safe to run this against the real vector database?". The live index
holds 27,989 documents and 35,796 embedded chunks that cost about $1.66 and an hour of Vertex AI quota to
produce. V005 turns ``rag.document`` into a Type 2 dimension and re-points ``rag.chunk`` from ``doc_id``
to ``doc_sk``, and it must not lose a single vector.

So: build the schema as it was at V004, fill it the way the old pipeline filled it, migrate, and check
that the same chunk rows are still there with the same vectors. Then run an ingest on top, because the
watermark signature now carries a digest of the rule file - the first run after this deployment sees
"the rules changed", and it must still cost nothing.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from crawlerrag import migrate as migrate_mod
from crawlerrag.ingest import pipeline
from crawlerrag.rag import index, retrieve
from crawlerrag.rag.chunking import split_text, text_hash

from .conftest import DIM, FakeEmbedder

OLD = ("V001", "V002", "V003", "V004")


def _vector(text: str) -> str:
    """The same deterministic vector FakeEmbedder would have produced for this text."""
    digest = hashlib.sha256(text.encode()).digest()
    return "[" + ",".join(repr(b / 255 + 0.01) for b in digest[:DIM]) + "]"


def _fill_as_the_old_pipeline_did(vector, crawler, admin, settings) -> None:
    with crawler.run("openfda_enforcement") as run:
        crawler.drug(run, "D-0001-2017", firm="Cantrell Drug Company")
        crawler.drug(run, "D-0002-2017", firm="Acme Pharma")
    doc_type = pipeline.doc_types_for(settings, ["drug_recall"])[0]
    rows = admin.execute("SELECT r.recall_number AS record_key, r.*, NULL::text[] AS product_ndcs "
                         "FROM drug.recall r ORDER BY r.recall_number").fetchall()
    for row in rows:
        doc = doc_type.build(row)
        vector.execute(
            "INSERT INTO rag.document (doc_id, doc_type, source_id, title, body, url, metadata, "
            "content_hash, is_active, indexed_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, true, now())",
            (doc.doc_id, doc.doc_type, doc.source_id, doc.title, doc.body, doc.url,
             json.dumps(doc.metadata), doc.content_hash))
        for ord_, text in enumerate(split_text(doc.text, size=settings.rag_chunk_chars,
                                               overlap=settings.rag_chunk_overlap)):
            vector.execute("INSERT INTO rag.chunk (doc_id, ord, text, text_hash, embedding, embed_model, "
                           "embedded_at) VALUES (%s, %s, %s, %s, %s, 'fake-embed', now())",
                           (doc.doc_id, ord_, text, text_hash(text), _vector(text)))
    head = admin.execute("SELECT max(change_id) AS head FROM crawl.record_change").fetchone()["head"]
    vector.execute("INSERT INTO ingest.watermark (doc_type, source_id, change_id, signature, batch_id) "
                   "VALUES ('drug_recall', 'openfda_enforcement', %s, %s, 1)",
                   (head, f"v1|chunk={settings.rag_chunk_chars}/{settings.rag_chunk_overlap}"))
    index.refresh_lexeme_stats(vector)


@pytest.fixture
def migrated(env, tmp_path, monkeypatch):
    """A V004 index with data in it, then migrated. Returns what it looked like before."""
    vector = env["vector"]
    vector.execute("DROP SCHEMA IF EXISTS rag, ingest, meta, graph CASCADE")
    vector.execute("DROP TABLE IF EXISTS public.schema_migration")

    everything = migrate_mod.MIGRATIONS_DIR
    old_dir = tmp_path / "v004"
    old_dir.mkdir()
    for path in sorted(everything.glob("V*__*.sql")):
        if path.name.split("__", 1)[0] in OLD:
            (old_dir / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(migrate_mod, "MIGRATIONS_DIR", old_dir)
    migrate_mod.migrate(vector)
    index.init_index(vector, FakeEmbedder(), dim=DIM)
    _fill_as_the_old_pipeline_did(vector, env["crawler"], env["admin"], env["settings"])

    before = {
        "documents": vector.execute("SELECT count(*) AS n FROM rag.document").fetchone()["n"],
        "chunks": vector.execute("SELECT count(*) AS n FROM rag.chunk").fetchone()["n"],
        "vectors": vector.execute("SELECT count(embedding) AS n FROM rag.chunk").fetchone()["n"],
        "texts": {r["text"] for r in vector.execute("SELECT text FROM rag.chunk").fetchall()},
        "ids": [r["chunk_id"] for r in vector.execute("SELECT chunk_id FROM rag.chunk "
                                                      "ORDER BY chunk_id").fetchall()],
    }
    assert before["documents"] == 2 and before["vectors"] == before["chunks"] > 0

    monkeypatch.setattr(migrate_mod, "MIGRATIONS_DIR", everything)
    before["applied"] = migrate_mod.migrate(vector)
    return before


# ---------------------------------------------------------------- what the migration did
def test_only_the_new_migrations_are_applied(migrated):
    assert [name.split("__", 1)[0] for name in migrated["applied"]] == ["V005", "V006"]


def test_no_chunk_and_no_vector_is_lost(migrated, env):
    after = env["vector"].execute("SELECT count(*) AS chunks, count(embedding) AS vectors "
                                  "FROM rag.chunk").fetchone()
    assert (after["chunks"], after["vectors"]) == (migrated["chunks"], migrated["vectors"])


def test_the_chunk_rows_are_the_same_rows(migrated, env):
    """Nothing is re-inserted, so the HNSW entries are untouched too."""
    ids = [r["chunk_id"] for r in env["vector"].execute("SELECT chunk_id FROM rag.chunk "
                                                        "ORDER BY chunk_id").fetchall()]
    assert ids == migrated["ids"]


def test_the_chunk_text_is_untouched(migrated, env):
    texts = {r["text"] for r in env["vector"].execute("SELECT text FROM rag.chunk").fetchall()}
    assert texts == migrated["texts"]


def test_every_document_is_version_one_valid_since_it_was_created(migrated, env):
    rows = env["vector"].execute("SELECT version, is_current, valid_from, valid_to, change_reason, "
                                 "created_at FROM rag.document").fetchall()
    assert len(rows) == migrated["documents"]
    assert {r["version"] for r in rows} == {1}
    assert all(r["is_current"] and r["valid_to"] is None for r in rows)
    assert all(r["valid_from"] == r["created_at"] for r in rows)
    assert {r["change_reason"] for r in rows} == {"first version"}


def test_every_chunk_points_at_its_document(migrated, env):
    orphans = env["vector"].execute(
        "SELECT count(*) AS n FROM rag.chunk c LEFT JOIN rag.document d USING (doc_sk) "
        "WHERE d.doc_sk IS NULL").fetchone()
    assert orphans["n"] == 0


def test_retrieval_still_finds_the_migrated_documents(migrated, env):
    hits = retrieve.search(env["vector"], env["settings"], FakeEmbedder(), "Cantrell Drug Company",
                           top_k=5).hits
    assert "drug_recall:D-0001-2017" in {h.doc_id for h in hits}


def test_the_history_view_works_right_away(migrated, env):
    rows = pipeline.document_history(env["vector"], "drug_recall:D-0001-2017")
    assert [r["version"] for r in rows] == [1]


# ---------------------------------------------------------------- the first run after the deployment
def ingest(env, **kw):
    embedder = FakeEmbedder()
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall"])
    return pipeline.run(env["vector"], env["source"], env["settings"], doc_types,
                        embedder_factory=lambda s: embedder, **kw), embedder


def test_the_first_run_sees_the_rules_changed_and_re_checks_everything(migrated, env):
    """The signature now carries a digest of the rule file, so the stored one no longer matches."""
    result, _ = ingest(env)
    batch = result.batches[0]
    assert batch.mode == "full" and "rule" in batch.reason


def test_that_first_run_embeds_nothing_and_rewrites_no_chunk(migrated, env):
    """The rules build the same text, so there is nothing to pay for and nothing to re-chunk."""
    before = env["vector"].execute("SELECT count(*) AS n, min(chunk_id) AS lo, max(chunk_id) AS hi "
                                   "FROM rag.chunk").fetchone()
    result, embedder = ingest(env)
    stats = result.batches[0].stats
    assert embedder.requests == 0
    assert (stats.chunks_added, stats.chunks_removed, stats.updated) == (0, 0, 0)
    assert stats.unchanged == migrated["documents"]
    after = env["vector"].execute("SELECT count(*) AS n, min(chunk_id) AS lo, max(chunk_id) AS hi "
                                  "FROM rag.chunk").fetchone()
    assert after == before


def test_that_first_run_opens_no_version(migrated, env):
    ingest(env)
    assert env["vector"].execute("SELECT count(*) AS n FROM rag.document").fetchone()["n"] == \
        migrated["documents"]


def test_the_plan_for_that_first_run_promises_nothing_to_embed(migrated, env):
    """`plan` is what a careful operator runs before `ingest`; it must not predict a rebuild."""
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall"])
    with env["source"] as src:
        rows = pipeline.plan(env["vector"], src, env["settings"], doc_types)
    assert rows[0].mode == "full"
    assert (rows[0].chunks_to_embed, rows[0].changed_docs, rows[0].new_versions) == (0, 0, 0)
    assert rows[0].unchanged_docs == migrated["documents"]


def test_the_watermark_is_rewritten_with_the_new_signature(migrated, env):
    ingest(env)
    row = env["vector"].execute("SELECT signature FROM ingest.watermark WHERE doc_type = 'drug_recall'"
                                ).fetchone()
    assert row["signature"].startswith("v1|rule=")


def test_the_run_after_that_is_incremental_again(migrated, env):
    ingest(env)
    result, embedder = ingest(env)
    assert result.batches[0].status == "nothing" and embedder.requests == 0
