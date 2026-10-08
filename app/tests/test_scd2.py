"""Which SCD Type 2 action a document change deserves.

``rag.document`` keeps every version of a document: a row per version with ``valid_from`` /
``valid_to`` / ``is_current``, and chunks hanging off the version's surrogate key. Deciding what to do
is pure logic, so it is tested without a database here;
``tests/integration/test_scd2_db.py`` then proves the rows and the vectors actually move.

Four actions, and the difference between the middle two is money:

* ``insert``                  - no version yet;
* ``new_version``             - tracked text changed: close the old version, open a new one, re-chunk
                                (chunks whose text is unchanged carry their vector over);
* ``new_version_move_chunks`` - only the soft-delete flag changed, so the text is identical: open a new
                                version and *move* the existing chunks to it. No re-chunking, no
                                embedding, nothing to pay for;
* ``overwrite``               - an untracked attribute changed (url, metadata): Type 1, update in place;
* ``unchanged``               - nothing to do.
"""
from __future__ import annotations

import pytest

from crawlerrag.ingest.scd2 import Current, decide
from crawlerrag.model import Document
from crawlerrag.rules.models import Scd2Spec


def doc(**over):
    fields = {"doc_id": "drug_recall:D-1", "doc_type": "drug_recall", "source_id": "openfda_enforcement",
              "title": "FDA drug recall D-1 - Acme", "body": "Reason for recall: sterility",
              "url": None, "metadata": {"year": 2026}, "is_active": True}
    fields.update(over)
    return Document(**fields)


def current(document, **over):
    fields = {"doc_sk": 7, "version": 3, "content_hash": document.tracked_hash(["title", "body"]),
              "is_active": document.is_active, "url": document.url, "metadata": document.metadata}
    fields.update(over)
    return Current(**fields)


SPEC = Scd2Spec()


def test_a_document_with_no_version_yet_is_inserted():
    action = decide(None, doc(), SPEC)
    assert (action.kind, action.reason) == ("insert", "first version")


def test_changed_text_opens_a_new_version():
    before = current(doc())
    action = decide(before, doc(body="Reason for recall: contamination"), SPEC)
    assert action.kind == "new_version"
    assert "text" in action.reason


def test_a_changed_title_opens_a_new_version():
    assert decide(current(doc()), doc(title="FDA drug recall D-1 - Acme Pharma"), SPEC).kind == "new_version"


def test_a_deactivated_document_opens_a_new_version_and_moves_its_chunks():
    action = decide(current(doc()), doc(is_active=False), SPEC)
    assert (action.kind, action.reason) == ("new_version_move_chunks", "deactivated at the source")


def test_a_reactivated_document_opens_a_new_version_and_moves_its_chunks():
    before = current(doc(), is_active=False)
    action = decide(before, doc(is_active=True), SPEC)
    assert (action.kind, action.reason) == ("new_version_move_chunks", "reactivated at the source")


def test_activation_changes_can_be_configured_not_to_make_a_version():
    spec = Scd2Spec(version_on_activation_change=False)
    assert decide(current(doc()), doc(is_active=False), spec).kind == "overwrite"


def test_a_new_url_is_overwritten_in_place():
    before = current(doc())
    action = decide(before, doc(url="https://www.cpsc.gov/Recalls/1"), SPEC)
    assert action.kind == "overwrite"


def test_new_metadata_is_overwritten_in_place():
    before = current(doc())
    assert decide(before, doc(metadata={"year": 2026, "state": "NY"}), SPEC).kind == "overwrite"


def test_an_identical_document_is_left_alone():
    assert decide(current(doc()), doc(), SPEC).kind == "unchanged"


def test_text_wins_over_an_untracked_change():
    """One action per document: a document whose text and url both changed gets a version."""
    before = current(doc())
    assert decide(before, doc(body="other", url="https://x.gov"), SPEC).kind == "new_version"


def test_text_wins_over_a_deactivation():
    before = current(doc())
    action = decide(before, doc(body="other", is_active=False), SPEC)
    assert action.kind == "new_version"


# ---------------------------------------------------------------- re-chunking is not a version
def test_changed_chunk_settings_re_chunk_without_a_new_version():
    """Chunk size is how the text is stored, not what the record says: no history event."""
    action = decide(current(doc()), doc(), SPEC, rechunk_all=True)
    assert action.kind == "rechunk"


def test_changed_chunk_settings_still_version_a_changed_document():
    action = decide(current(doc()), doc(body="other"), SPEC, rechunk_all=True)
    assert action.kind == "new_version"


# ---------------------------------------------------------------- what counts as "tracked"
def test_the_default_tracked_hash_is_the_historical_content_hash():
    """The 27,989 documents already in the index were hashed as sha256(title + "\\n" + body).

    Changing that formula would make every document look changed on the next run, so the default
    tracking set must keep producing exactly the old hash."""
    import hashlib
    d = doc()
    expected = hashlib.sha256(f"{d.title}\n{d.body}".encode("utf-8")).hexdigest()
    assert d.tracked_hash(["title", "body"]) == expected == d.content_hash


def test_a_doc_type_may_also_track_the_url():
    spec = Scd2Spec(track=["title", "body", "url"], overwrite=["metadata"])
    before = current(doc(), content_hash=doc().tracked_hash(spec.track))
    assert decide(before, doc(url="https://x.gov"), spec).kind == "new_version"


def test_tracking_more_attributes_changes_the_hash():
    d = doc()
    assert d.tracked_hash(["title", "body"]) != d.tracked_hash(["title", "body", "url"])


@pytest.mark.parametrize("attrs", [["body", "title"], ["title", "body"]])
def test_the_tracked_hash_does_not_depend_on_the_order_attributes_are_listed(attrs):
    assert doc().tracked_hash(attrs) == doc().tracked_hash(["title", "body"])
