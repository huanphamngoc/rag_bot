"""What the ingestion does with a source that is not well behaved.

test_incremental.py covers the normal change-log path and test_quality_gate.py the checks on the rows.
This file is about the cases that are not in either: a source that cannot be reached at all, a record
that changed several times before anyone read the log, a row that is almost entirely null, a change for
a record that was never there, and a rule edit that changes nothing about the text.
"""
from __future__ import annotations

import shutil

import psycopg
import pytest

from crawlerrag.ingest import pipeline
from crawlerrag.rules import clear_cache

from .conftest import DSN_RO, FakeEmbedder, _connect
from .test_incremental import batch, chunks, doc, head, ingest, seed, watermark


# ---------------------------------------------------------------- the source is gone
def test_a_source_that_cannot_be_read_keeps_the_watermark_where_it_was(env):
    """A broken source must not look like "no changes": the window has to be read again next time."""
    ingest(env)
    before = watermark(env, "drug_recall")
    env["source"].close()

    with pytest.raises(psycopg.Error):
        ingest(env)

    assert watermark(env, "drug_recall") == before


def test_a_dead_source_fails_before_a_batch_is_opened(env):
    """Measured: the first node reads the catalog from the source, so the run dies there and leaves no
    half-finished batch behind. Nothing has to be cleaned up and no batch row claims work it never did."""
    ingest(env)
    before = env["vector"].execute("SELECT count(*) AS n FROM ingest.batch").fetchone()["n"]
    env["source"].close()

    with pytest.raises(psycopg.Error):
        ingest(env)

    rows = env["vector"].execute("SELECT status FROM ingest.batch").fetchall()
    assert len(rows) == before
    assert {r["status"] for r in rows} == {"succeeded"}


def test_the_run_after_a_failure_picks_the_same_window_up_again(env):
    """The point of holding the watermark: nothing is skipped because of the outage."""
    _, first = ingest(env)
    env["crawler"].drug(_run(env), "D-0003-2017", firm="Later Pharma")
    env["source"].close()
    with pytest.raises(psycopg.Error):
        ingest(env)

    env["source"] = _connect(DSN_RO)
    result, embedder = ingest(env)

    assert doc(env, "drug_recall:D-0003-2017") is not None
    assert watermark(env, "drug_recall") == head(env, "openfda_enforcement")


def _run(env):
    """A finished crawl run id, for a change logged outside the seed helper."""
    with env["crawler"].run("openfda_enforcement") as run_id:
        pass
    return run_id


# ---------------------------------------------------------------- the same key, several changes
def test_a_record_changed_three_times_before_the_first_read_is_built_once(env):
    """The change log is a list of keys to re-read, not a list of edits to replay."""
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        c.drug(run, "D-0010-2017", firm="First Name")
        c.drug(run, "D-0010-2017", change="update", firm="Second Name")
        c.drug(run, "D-0010-2017", change="update", firm="Third Name")

    result, embedder = ingest(env)

    row = doc(env, "drug_recall:D-0010-2017")
    assert row["version"] == 1                      # one document, not three versions
    assert "Third Name" in row["title"]             # the state the source ended in
    assert batch(result, "drug_recall").keys == 1
    assert len([t for t in embedder.texts if "Third Name" in t]) == 1


def test_the_window_covers_every_change_id_of_that_key(env):
    """Three log rows, so the watermark has to move past all three or they are read forever."""
    c = env["crawler"]
    with c.run("openfda_enforcement") as run:
        for name in ("First", "Second", "Third"):
            c.drug(run, "D-0011-2017", change="update", firm=name)

    ingest(env)

    assert watermark(env, "drug_recall") == head(env, "openfda_enforcement")
    second, _ = ingest(env)
    assert batch(second, "drug_recall").status == "nothing"


# ---------------------------------------------------------------- a row that is almost empty
def test_a_record_with_nothing_but_its_key_still_becomes_a_document(env):
    """openFDA reports arrive incomplete. The title falls back and the empty lines are left out,
    rather than the document being skipped or a line reading "Reason for recall: None"."""
    with env["crawler"].run("openfda_enforcement") as run:
        env["admin"].execute("INSERT INTO drug.recall (recall_number) VALUES ('D-0404-2017')")
        env["admin"].execute(
            "INSERT INTO crawl.record_change (source_id, record_key, run_id, change_type) "
            "VALUES ('openfda_enforcement', 'D-0404-2017', %s, 'insert')", (run,))

    ingest(env)

    row = doc(env, "drug_recall:D-0404-2017")
    assert row["title"] == "FDA drug recall D-0404-2017 - unknown firm"
    body = chunks(env, "drug_recall:D-0404-2017")[0]["text"]
    assert "Recalling firm: unknown firm" in body
    for absent in ("Reason for recall", "Product description", "Firm location", "Recall initiated"):
        assert absent not in body


def test_that_document_is_chunked_and_embedded_like_any_other(env):
    with env["crawler"].run("openfda_enforcement") as run:
        env["admin"].execute("INSERT INTO drug.recall (recall_number) VALUES ('D-0405-2017')")
        env["admin"].execute(
            "INSERT INTO crawl.record_change (source_id, record_key, run_id, change_type) "
            "VALUES ('openfda_enforcement', 'D-0405-2017', %s, 'insert')", (run,))

    _, embedder = ingest(env)

    rows = chunks(env, "drug_recall:D-0405-2017")
    assert len(rows) == 1 and rows[0]["has_vector"]


def test_a_child_collection_with_no_rows_leaves_its_line_out(env):
    """`template` renders empty when a column is empty, so the label disappears with it."""
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].drug(run, "D-0012-2017", ndcs=())

    ingest(env)

    body = chunks(env, "drug_recall:D-0012-2017")[0]["text"]
    assert "National drug codes involved" not in body


def test_the_same_record_with_a_child_row_gets_that_line(env):
    """The other half of the case above, so an empty collection is not confused with a broken join."""
    with env["crawler"].run("openfda_enforcement") as run:
        env["crawler"].drug(run, "D-0013-2017", ndcs=("0093-1234", "0093-5678"))

    ingest(env)

    body = chunks(env, "drug_recall:D-0013-2017")[0]["text"]
    assert "National drug codes involved: 0093-1234, 0093-5678" in body


# ---------------------------------------------------------------- a change for a record that is not there
def test_a_change_for_a_key_that_was_never_loaded_and_is_gone_is_not_an_error(env):
    """The crawler can log a key and delete the row before this app reads the log."""
    seed(env)
    ingest(env)
    with env["crawler"].run("openfda_enforcement") as run:
        env["admin"].execute(
            "INSERT INTO crawl.record_change (source_id, record_key, run_id, change_type) "
            "VALUES ('openfda_enforcement', 'D-9999-2017', %s, 'deactivate')", (run,))

    result, embedder = ingest(env)

    assert result.status == "succeeded"
    assert doc(env, "drug_recall:D-9999-2017") is None
    assert embedder.requests == 0
    assert watermark(env, "drug_recall") == head(env, "openfda_enforcement")


# ---------------------------------------------------------------- a rule edit that changes no text
def test_bumping_the_doc_type_version_re_checks_everything_but_embeds_nothing(env, tmp_path):
    """`version` is part of the text digest, so bumping it switches the next run to a full re-check.
    That is the cheap half of the deal: nothing about the text changed, so nothing is paid for again."""
    seed(env)
    ingest(env)

    rules = tmp_path / "rules"
    shutil.copytree(env["settings"].rules_dir, rules)
    path = rules / "doc_types" / "drug_recall.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("version: 1", "version: 2", 1),
                    encoding="utf-8")
    env["settings"].rules_dir = str(rules)
    clear_cache()

    result, embedder = ingest(env)

    assert batch(result, "drug_recall").mode == "full"
    assert embedder.requests == 0
    assert batch(result, "drug_recall").stats.versions_created == 0


def test_that_full_re_check_opens_no_version_and_rewrites_no_chunk(env, tmp_path):
    seed(env)
    ingest(env)
    before = {r["ord"]: r["text"] for r in chunks(env, "drug_recall:D-0001-2017")}

    rules = tmp_path / "rules"
    shutil.copytree(env["settings"].rules_dir, rules)
    path = rules / "doc_types" / "drug_recall.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("version: 1", "version: 2", 1),
                    encoding="utf-8")
    env["settings"].rules_dir = str(rules)
    clear_cache()
    ingest(env)

    assert doc(env, "drug_recall:D-0001-2017")["version"] == 1
    assert {r["ord"]: r["text"] for r in chunks(env, "drug_recall:D-0001-2017")} == before


def test_the_run_after_the_bump_is_incremental_again(env, tmp_path):
    seed(env)
    ingest(env)
    rules = tmp_path / "rules"
    shutil.copytree(env["settings"].rules_dir, rules)
    path = rules / "doc_types" / "drug_recall.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("version: 1", "version: 2", 1),
                    encoding="utf-8")
    env["settings"].rules_dir = str(rules)
    clear_cache()
    ingest(env)

    second, _ = ingest(env)

    assert batch(second, "drug_recall").status == "nothing"
