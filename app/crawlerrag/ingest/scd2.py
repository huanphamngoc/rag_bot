"""Deciding what a changed document does to its history.

``rag.document`` is a Type 2 dimension: one row per version, ``is_current`` on exactly one of them. The
rule file says which attributes are *tracked* (a change opens a version) and which are *overwritten*
(a change updates the current row). Everything here is pure, so the interesting part - which of the
five actions a change deserves - is tested without a database.

The distinction that costs or saves money:

* ``new_version`` re-chunks the document. Chunks whose text is unchanged carry their vector over, new
  text has to be embedded.
* ``new_version_move_chunks`` happens when only the soft-delete flag moved. The text is identical, so
  the existing chunk rows are simply pointed at the new version: nothing is embedded again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crawlerrag.model import Document
from crawlerrag.rules.models import Scd2Spec

INSERT = "insert"
NEW_VERSION = "new_version"
NEW_VERSION_MOVE_CHUNKS = "new_version_move_chunks"
OVERWRITE = "overwrite"
RECHUNK = "rechunk"
UNCHANGED = "unchanged"


@dataclass(frozen=True)
class Current:
    """The current version of a document, as read from rag.document."""
    doc_sk: int
    version: int
    content_hash: str
    is_active: bool
    url: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Action:
    kind: str
    reason: str                     # stored in rag.document.change_reason

    @property
    def opens_version(self) -> bool:
        return self.kind in (INSERT, NEW_VERSION, NEW_VERSION_MOVE_CHUNKS)

    @property
    def writes_chunks(self) -> bool:
        return self.kind in (INSERT, NEW_VERSION, RECHUNK)


def _overwritten_changed(prev: Current, doc: Document, attrs: list[str]) -> bool:
    """Only url and metadata are compared: title and body are never in ``overwrite`` (the validator
    forbids an attribute in both lists, and ``track`` always holds at least one of them)."""
    if "url" in attrs and prev.url != doc.url:
        return True
    return "metadata" in attrs and prev.metadata != doc.metadata


def decide(prev: Current | None, doc: Document, scd2: Scd2Spec, *,
           rechunk_all: bool = False) -> Action:
    """What to do with this document, given the version currently on record.

    ``rechunk_all`` is set when the chunk settings or the rule's text shape changed: the stored text is
    the same record, cut differently, so it re-chunks without opening a version.
    """
    tracked = doc.tracked_hash(scd2.track)
    if prev is None:
        return Action(INSERT, "first version")
    if prev.content_hash != tracked:
        return Action(NEW_VERSION, "the tracked text changed at the source")
    if prev.is_active != doc.is_active and scd2.version_on_activation_change:
        reason = "reactivated at the source" if doc.is_active else "deactivated at the source"
        # Identical text, so the chunks move - unless they have to be re-cut anyway.
        return Action(NEW_VERSION if rechunk_all else NEW_VERSION_MOVE_CHUNKS, reason)
    if rechunk_all:
        return Action(RECHUNK, "re-chunked: chunk settings or document shape changed")
    if prev.is_active != doc.is_active:
        return Action(OVERWRITE, "the activation flag changed")
    if _overwritten_changed(prev, doc, scd2.overwrite):
        return Action(OVERWRITE, "untracked fields refreshed")
    return Action(UNCHANGED, "")
