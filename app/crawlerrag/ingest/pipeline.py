"""Ingestion: crawler database -> documents -> chunks -> vectors, incrementally.

The order of the work is the LangGraph graph in :mod:`crawlerrag.ingest.graph`; this module holds the
operations it calls, which is where all the SQL lives:

1. **plan**     - full load when there is no watermark yet, when the rule's text shape or the chunk
                  settings changed, or when asked (--full); otherwise incremental from the watermark.
2. **extract**  - ONE read-only REPEATABLE READ transaction on the crawler database: the upper bound
                  (max change_id of the source), the record keys changed in (watermark, bound], and the
                  current rows of those keys. One snapshot, so the rows are exactly as of the bound.
3. **quality**  - the business rules from rules/*.yaml, on those rows. An error fails the batch and
                  leaves the watermark alone, so the same window is read again next time.
4. **stage**    - Type 2 history: a document whose tracked text changed closes its version and opens a
                  new one; one whose soft-delete flag changed opens a version and *moves* its chunks
                  (identical text, nothing to embed); an untracked change updates the current row. Pages
                  of ``INGEST_PAGE_DOCS`` documents per transaction; the LAST page also writes the
                  watermark, so documents and watermark commit together.
5. **embed**    - chunks without a vector, each request committed on its own.

Why the crawler's change_id is a safe watermark (docs/DESIGN.md section 4): the crawler writes a batch
of normalised rows and its crawl.record_change rows in one transaction, and runs at most one crawl per
source at a time, so the change ids of one source become visible in increasing order.
"""
from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import psycopg
from psycopg import IsolationLevel
from psycopg.types.json import Jsonb

from crawlerrag.db import scalar
from crawlerrag.ingest import embed as embed_mod
from crawlerrag.ingest import quality as quality_mod
from crawlerrag.ingest.documents import DocType, available, doc_types_for, doc_types_from, fetch_rows
from crawlerrag.ingest.scd2 import (INSERT, NEW_VERSION, NEW_VERSION_MOVE_CHUNKS, OVERWRITE, RECHUNK,
                                    UNCHANGED, Current, decide)
from crawlerrag.meta import catalog as meta_catalog
from crawlerrag.meta import introspect
from crawlerrag.model import Document
from crawlerrag.rag import index
from crawlerrag.rag.chunking import split_text, text_hash
from crawlerrag.rag.providers import embedding_provider
from crawlerrag.rules import load_rules_cached
from crawlerrag.rules.validate import Finding, validate_ruleset

log = logging.getLogger(__name__)

LOCK_NAME = "crawlerrag.ingest"

__all__ = ["AlreadyRunning", "WatermarkAhead", "DocType", "PostgresOps", "RunResult", "BatchResult",
           "StageStats", "run", "plan", "source_head", "fetch_rows", "stage_page", "signature",
           "doc_types_for", "doc_types_from", "document_history", "available", "LOCK_NAME"]


class AlreadyRunning(RuntimeError):
    """Another ingest holds the lock on this vector database."""


class WatermarkAhead(RuntimeError):
    """The stored watermark is beyond the source's change log: not the database it was built from."""


class QualityFailed(RuntimeError):
    """A business rule with severity error did not hold on the extracted rows."""


def signature(doc_type: DocType, settings) -> str:
    return doc_type.signature(settings)


# ---------------------------------------------------------------- extract (source, read-only)
@dataclass
class Extract:
    doc_type: DocType
    mode: str                       # "full" | "incremental"
    from_change_id: int             # exclusive
    to_change_id: int               # inclusive, read in the same snapshot as the rows
    keys: list[str] | None          # changed record keys; None in a full load
    rows: list[dict]
    documents: list[Document]
    missing_keys: list[str]         # changed keys whose row no longer exists


def source_head(conn: psycopg.Connection, source_id: str) -> int:
    return int(scalar(conn, "SELECT coalesce(max(change_id), 0) FROM crawl.record_change WHERE source_id = %s",
                      (source_id,)) or 0)


def extract(src: psycopg.Connection, doc_type: DocType, *, mode: str, from_change_id: int) -> Extract:
    src.isolation_level = IsolationLevel.REPEATABLE_READ
    src.read_only = True
    with src.transaction():
        to_change_id = source_head(src, doc_type.source_id)
        if to_change_id < from_change_id:
            raise WatermarkAhead(
                f"{doc_type.doc_type}: watermark {from_change_id} is beyond the source's last change "
                f"{to_change_id} ({doc_type.source_id}). This vector database was built from another crawler "
                "database (or that database was restored); run `ingest --full` to rebuild from this one.")
        keys: list[str] | None = None
        if mode == "incremental":
            keys = [r["record_key"] for r in src.execute(
                "SELECT DISTINCT record_key FROM crawl.record_change "
                "WHERE source_id = %s AND change_id > %s AND change_id <= %s",
                (doc_type.source_id, from_change_id, to_change_id)).fetchall()]
        rows = fetch_rows(src, doc_type, keys)
    documents = [doc_type.build(r) for r in rows]
    missing = sorted(set(keys) - {str(r["record_key"]) for r in rows}) if keys is not None else []
    return Extract(doc_type, mode, from_change_id, to_change_id, keys, rows, documents, missing)


# ---------------------------------------------------------------- stage (target)
@dataclass
class StageStats:
    inserted: int = 0
    updated: int = 0              # tracked text changed: a new version, re-chunked
    refreshed: int = 0            # only url / metadata changed: current row updated in place
    unchanged: int = 0
    deactivated: int = 0
    reactivated: int = 0
    versions_created: int = 0     # Type 2 rows opened (insert, text change, (de)activation)
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_moved: int = 0         # carried to a new version without re-chunking
    vectors_carried: int = 0      # unchanged chunk text inside a changed document keeps its vector


INSERT_VERSION = """
INSERT INTO rag.document (doc_id, doc_type, source_id, title, body, url, metadata, content_hash,
                          is_active, version, is_current, valid_from, change_reason, source_change_id,
                          batch_id)
VALUES (%(doc_id)s, %(doc_type)s, %(source_id)s, %(title)s, %(body)s, %(url)s, %(metadata)s,
        %(content_hash)s, %(is_active)s, %(version)s, true, now(), %(change_reason)s, %(change_id)s,
        %(batch_id)s)
RETURNING doc_sk
"""

CLOSE_VERSION = ("UPDATE rag.document SET is_current = false, valid_to = now(), updated_at = now() "
                 "WHERE doc_sk = %s")

OVERWRITE_CURRENT = """
UPDATE rag.document
   SET url = %(url)s, metadata = %(metadata)s, is_active = %(is_active)s, updated_at = now(),
       source_change_id = %(change_id)s, batch_id = %(batch_id)s
 WHERE doc_sk = %(doc_sk)s
"""


def _current_versions(tgt: psycopg.Connection, doc_ids: Sequence[str]) -> dict[str, Current]:
    rows = tgt.execute(
        "SELECT doc_sk, doc_id, version, content_hash, is_active, url, metadata "
        "FROM rag.document WHERE doc_id = ANY(%s) AND is_current", (list(doc_ids),)).fetchall()
    return {r["doc_id"]: Current(doc_sk=r["doc_sk"], version=r["version"], content_hash=r["content_hash"],
                                 is_active=r["is_active"], url=r["url"], metadata=r["metadata"] or {})
            for r in rows}


def _open_version(tgt: psycopg.Connection, doc: Document, *, version: int, reason: str,
                  tracked_hash: str, change_id: int | None, batch_id: int | None) -> int:
    return int(scalar(tgt, INSERT_VERSION, {
        "doc_id": doc.doc_id, "doc_type": doc.doc_type, "source_id": doc.source_id, "title": doc.title,
        "body": doc.body, "url": doc.url, "metadata": Jsonb(doc.metadata), "content_hash": tracked_hash,
        "is_active": doc.is_active, "version": version, "change_reason": reason, "change_id": change_id,
        "batch_id": batch_id}))


def _write_chunks(tgt: psycopg.Connection, settings, entries: Sequence[tuple[Document, int, int | None]],
                  stats: StageStats) -> None:
    """Re-chunk these documents, carrying over the vector of every chunk whose text is unchanged.

    ``entries`` is (document, new doc_sk, previous doc_sk or None). The previous version's chunks are
    deleted here, which is the invariant retrieval leans on: chunks exist only for the current version.
    """
    if not entries:
        return
    new_sks = [new for _, new, _ in entries]
    affected = new_sks + [old for _, _, old in entries if old is not None]
    tgt.execute("DROP TABLE IF EXISTS carry")
    tgt.execute("CREATE TEMP TABLE carry (text_hash char(64) PRIMARY KEY, embedding vector, "
                "embed_model text) ON COMMIT DROP")
    tgt.execute("""
        INSERT INTO carry (text_hash, embedding, embed_model)
        SELECT DISTINCT ON (text_hash) text_hash, embedding, embed_model
          FROM rag.chunk WHERE doc_sk = ANY(%s) AND embedding IS NOT NULL
         ORDER BY text_hash, chunk_id
    """, (affected,))
    stats.chunks_removed += tgt.execute("DELETE FROM rag.chunk WHERE doc_sk = ANY(%s)", (affected,)).rowcount
    with tgt.cursor() as cur, cur.copy("COPY rag.chunk (doc_sk, ord, text, text_hash) FROM STDIN") as copy:
        for doc, new_sk, _ in entries:
            for ord_, text in enumerate(split_text(doc.text, size=settings.rag_chunk_chars,
                                                   overlap=settings.rag_chunk_overlap)):
                copy.write_row((new_sk, ord_, text, text_hash(text)))
                stats.chunks_added += 1
    stats.vectors_carried += tgt.execute("""
        UPDATE rag.chunk c SET embedding = k.embedding, embed_model = k.embed_model, embedded_at = now()
          FROM carry k
         WHERE c.doc_sk = ANY(%s) AND c.embedding IS NULL AND c.text_hash = k.text_hash
    """, (new_sks,)).rowcount


def _chunk_hashes(settings, doc: Document) -> list[str]:
    return [text_hash(t) for t in split_text(doc.text, size=settings.rag_chunk_chars,
                                             overlap=settings.rag_chunk_overlap)]


def _stored_chunk_hashes(tgt: psycopg.Connection, previous: dict[str, Current]) -> dict[int, list[str]]:
    """The chunk text hashes already stored for these versions, in order, in one query."""
    sks = [cur.doc_sk for cur in previous.values()]
    if not sks:
        return {}
    out: dict[int, list[str]] = {}
    for row in tgt.execute("SELECT doc_sk, text_hash FROM rag.chunk WHERE doc_sk = ANY(%s) "
                           "ORDER BY doc_sk, ord", (sks,)).fetchall():
        out.setdefault(row["doc_sk"], []).append(row["text_hash"])
    return out


def stage_page(tgt: psycopg.Connection, settings, doc_type: DocType, docs: Sequence[Document],
               stats: StageStats, *, rechunk_all: bool = False, batch_id: int | None = None,
               change_id: int | None = None) -> None:
    """Write one page of documents (the caller holds the transaction)."""
    if not docs:
        return
    scd2 = doc_type.scd2
    previous = _current_versions(tgt, [d.doc_id for d in docs])
    stored = _stored_chunk_hashes(tgt, previous) if rechunk_all else {}
    rechunk: list[tuple[Document, int, int | None]] = []
    for doc in docs:
        prev = previous.get(doc.doc_id)
        action = decide(prev, doc, scd2, rechunk_all=rechunk_all)
        tracked = doc.tracked_hash(scd2.track)
        if action.kind == RECHUNK and stored.get(prev.doc_sk) == _chunk_hashes(settings, doc):
            # The signature changed but this document's chunks come out byte-identical (a rule edit that
            # does not touch the text). Rewriting 36k chunk rows and their HNSW entries for nothing is
            # the most expensive way to do nothing at all.
            action = decide(prev, doc, scd2, rechunk_all=False)
        if action.kind == UNCHANGED:
            stats.unchanged += 1
            continue
        if action.kind == OVERWRITE:
            tgt.execute(OVERWRITE_CURRENT, {"url": doc.url, "metadata": Jsonb(doc.metadata),
                                            "is_active": doc.is_active, "doc_sk": prev.doc_sk,
                                            "change_id": change_id, "batch_id": batch_id})
            stats.refreshed += 1
            if prev.is_active != doc.is_active:
                stats.deactivated += not doc.is_active
                stats.reactivated += doc.is_active
            continue
        if action.kind == RECHUNK:
            tgt.execute(OVERWRITE_CURRENT, {"url": doc.url, "metadata": Jsonb(doc.metadata),
                                            "is_active": doc.is_active, "doc_sk": prev.doc_sk,
                                            "change_id": change_id, "batch_id": batch_id})
            tgt.execute("UPDATE rag.document SET indexed_at = NULL WHERE doc_sk = %s", (prev.doc_sk,))
            stats.updated += 1
            rechunk.append((doc, prev.doc_sk, None))
            continue

        version = 1 if prev is None else prev.version + 1
        if prev is not None:
            tgt.execute(CLOSE_VERSION, (prev.doc_sk,))
        new_sk = _open_version(tgt, doc, version=version, reason=action.reason, tracked_hash=tracked,
                               change_id=change_id, batch_id=batch_id)
        stats.versions_created += 1
        if action.kind == INSERT:
            stats.inserted += 1
            rechunk.append((doc, new_sk, None))
        elif action.kind == NEW_VERSION:
            stats.updated += 1
            if prev is not None and prev.is_active != doc.is_active:
                stats.deactivated += not doc.is_active
                stats.reactivated += doc.is_active
            rechunk.append((doc, new_sk, prev.doc_sk if prev else None))
        elif action.kind == NEW_VERSION_MOVE_CHUNKS:
            moved = tgt.execute("UPDATE rag.chunk SET doc_sk = %s WHERE doc_sk = %s",
                                (new_sk, prev.doc_sk)).rowcount
            stats.chunks_moved += moved
            stats.deactivated += not doc.is_active
            stats.reactivated += doc.is_active
            # The new row inherits an index state: every chunk it has is already embedded.
            tgt.execute("UPDATE rag.document SET indexed_at = (SELECT indexed_at FROM rag.document "
                        "WHERE doc_sk = %s) WHERE doc_sk = %s", (prev.doc_sk, new_sk))
    _write_chunks(tgt, settings, rechunk, stats)


# ---------------------------------------------------------------- soft deletes, as history
DEACTIVATE_WITH_VERSION = """
WITH target AS (
    SELECT doc_sk FROM rag.document
     WHERE is_current AND is_active AND {predicate}
), closed AS (
    UPDATE rag.document d SET is_current = false, valid_to = now(), updated_at = now()
      FROM target t WHERE d.doc_sk = t.doc_sk
  RETURNING d.doc_sk, d.doc_id, d.doc_type, d.source_id, d.title, d.body, d.url, d.metadata,
            d.content_hash, d.version, d.indexed_at
), opened AS (
    INSERT INTO rag.document (doc_id, doc_type, source_id, title, body, url, metadata, content_hash,
                              is_active, version, is_current, valid_from, change_reason,
                              source_change_id, batch_id, indexed_at)
    SELECT doc_id, doc_type, source_id, title, body, url, metadata, content_hash, false, version + 1,
           true, now(), %(reason)s, %(change_id)s, %(batch_id)s, indexed_at
      FROM closed
  RETURNING doc_sk, doc_id
), moved AS (
    UPDATE rag.chunk c SET doc_sk = o.doc_sk
      FROM opened o JOIN closed cl ON cl.doc_id = o.doc_id
     WHERE c.doc_sk = cl.doc_sk
  RETURNING 1
)
SELECT (SELECT count(*) FROM opened) AS versions, (SELECT count(*) FROM moved) AS moved
"""

DEACTIVATE_IN_PLACE = """
UPDATE rag.document SET is_active = false, updated_at = now()
 WHERE is_current AND is_active AND {predicate}
"""


def _deactivate(tgt: psycopg.Connection, doc_type: DocType, *, predicate: str, params: dict,
                reason: str, change_id: int | None, batch_id: int | None,
                stats: StageStats) -> int:
    if doc_type.scd2.version_on_activation_change:
        row = tgt.execute(DEACTIVATE_WITH_VERSION.format(predicate=predicate),
                          {**params, "reason": reason, "change_id": change_id,
                           "batch_id": batch_id}).fetchone()
        stats.versions_created += row["versions"]
        stats.chunks_moved += row["moved"]
        stats.deactivated += row["versions"]
        return int(row["versions"])
    closed = tgt.execute(DEACTIVATE_IN_PLACE.format(predicate=predicate), params).rowcount
    stats.deactivated += closed
    return closed


def deactivate(tgt: psycopg.Connection, doc_type: DocType, doc_ids: Sequence[str], *,
               stats: StageStats, change_id: int | None = None, batch_id: int | None = None) -> int:
    """Changed keys whose source row is gone."""
    if not doc_ids:
        return 0
    return _deactivate(tgt, doc_type, predicate="doc_id = ANY(%(ids)s)", params={"ids": list(doc_ids)},
                       reason="the source row is gone", change_id=change_id, batch_id=batch_id,
                       stats=stats)


def deactivate_absent(tgt: psycopg.Connection, doc_type: DocType, present: Sequence[str], *,
                      stats: StageStats, change_id: int | None = None,
                      batch_id: int | None = None) -> int:
    """Full load: documents of this type whose source row is not in the snapshot at all."""
    return _deactivate(tgt, doc_type,
                       predicate="doc_type = %(doc_type)s AND NOT (doc_id = ANY(%(present)s))",
                       params={"doc_type": doc_type.doc_type, "present": list(present)},
                       reason="not in the source any more", change_id=change_id, batch_id=batch_id,
                       stats=stats)


def write_watermark(tgt: psycopg.Connection, ex: Extract, sig: str, batch_id: int) -> None:
    tgt.execute("""
        INSERT INTO ingest.watermark (doc_type, source_id, change_id, signature, batch_id, updated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (doc_type) DO UPDATE SET source_id = EXCLUDED.source_id, change_id = EXCLUDED.change_id,
               signature = EXCLUDED.signature, batch_id = EXCLUDED.batch_id, updated_at = now()
    """, (ex.doc_type.doc_type, ex.doc_type.source_id, ex.to_change_id, sig, batch_id))


def document_history(conn: psycopg.Connection, doc_id: str) -> list[dict]:
    """Every version of one document, oldest first - what SCD Type 2 is for."""
    return conn.execute(
        "SELECT version, is_current, is_active, valid_from, valid_to, change_reason, source_change_id, "
        "batch_id, title, content_hash FROM rag.document WHERE doc_id = %s ORDER BY version",
        (doc_id,)).fetchall()


# ---------------------------------------------------------------- one batch
@dataclass
class BatchResult:
    doc_type: str
    batch_id: int
    mode: str
    reason: str
    from_change_id: int
    to_change_id: int
    keys: int
    status: str
    stats: StageStats = field(default_factory=StageStats)
    findings: list[quality_mod.QualityFinding] = field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        """The JSON-safe view the graph state carries."""
        return {"doc_type": self.doc_type, "batch_id": self.batch_id, "mode": self.mode,
                "reason": self.reason, "from_change_id": self.from_change_id,
                "to_change_id": self.to_change_id, "keys": self.keys, "status": self.status,
                "seconds": self.seconds, **asdict(self.stats)}


def read_watermark(tgt: psycopg.Connection, doc_type: str) -> dict | None:
    return tgt.execute("SELECT * FROM ingest.watermark WHERE doc_type = %s", (doc_type,)).fetchone()


def choose_mode(watermark: dict | None, sig: str, *, full: bool) -> tuple[str, str]:
    if full:
        return "full", "requested (--full)"
    if watermark is None:
        return "full", "first load: no watermark"
    if watermark["signature"] != sig:
        return "full", f"document rules or chunking changed: {watermark['signature']} -> {sig}"
    return "incremental", f"changes after change_id {watermark['change_id']}"


# ---------------------------------------------------------------- the operations the graph calls
class PostgresOps:
    """Every database operation of a run, in the order the graph calls them.

    The graph owns the order and the branches and keeps only JSON-safe counters in its state; the rows,
    the documents and the connections stay here.
    """

    def __init__(self, tgt: psycopg.Connection, src: psycopg.Connection, settings,
                 doc_types: Sequence[DocType], *, embedder_factory: Callable = embedding_provider):
        self.tgt, self.src, self.settings = tgt, src, settings
        self.doc_types = {dt.doc_type: dt for dt in doc_types}
        self.embedder_factory = embedder_factory
        self.results: list[BatchResult] = []
        self.lexemes: int | None = None
        self.embed_stats: embed_mod.EmbedStats | None = None
        self._extracts: dict[str, Extract] = {}
        self._started: dict[str, float] = {}
        self.catalog = None

    # ---- rules and metadata
    def read_catalog(self):
        ruleset = load_rules_cached(self.settings.rules_dir)
        catalog = introspect.read_catalog(self.src, ruleset.catalog.schemas, ruleset.catalog.exclude)
        meta_catalog.save_catalog(self.tgt, catalog, source="ingest")
        self.catalog = catalog
        return catalog

    def validate(self, catalog=None) -> list[Finding]:
        """The graph keeps no catalog in its state (it is not JSON), so the one just read is used."""
        return validate_ruleset(load_rules_cached(self.settings.rules_dir), catalog or self.catalog)

    # ---- per doc type
    def begin_batch(self, doc_type: str, *, full: bool) -> dict:
        dt = self.doc_types[doc_type]
        self._started[doc_type] = time.monotonic()
        sig = dt.signature(self.settings)
        wm = read_watermark(self.tgt, doc_type)
        mode, reason = choose_mode(wm, sig, full=full)
        from_id = int(wm["change_id"]) if wm else 0
        batch_id = int(scalar(self.tgt, """
            INSERT INTO ingest.batch (doc_type, source_id, mode, reason, from_change_id)
            VALUES (%s, %s, %s, %s, %s) RETURNING batch_id
        """, (doc_type, dt.source_id, mode, reason, from_id)))
        return {"doc_type": doc_type, "batch_id": batch_id, "mode": mode, "reason": reason,
                "from_change_id": from_id, "signature": sig,
                "rechunk_all": bool(wm is not None and wm["signature"] != sig)}

    def extract(self, plan: dict) -> dict:
        dt = self.doc_types[plan["doc_type"]]
        try:
            ex = extract(self.src, dt, mode=plan["mode"], from_change_id=plan["from_change_id"])
        except Exception as exc:
            self.fail_batch(plan, f"{type(exc).__name__}: {exc}")
            raise
        self._extracts[plan["doc_type"]] = ex
        return {"keys": len(ex.keys) if ex.keys is not None else len(ex.documents),
                "documents": len(ex.documents), "to_change_id": ex.to_change_id,
                "missing_keys": len(ex.missing_keys)}

    def quality(self, plan: dict) -> list[quality_mod.QualityFinding]:
        dt = self.doc_types[plan["doc_type"]]
        ex = self._extracts[plan["doc_type"]]
        findings = quality_mod.check_rows(dt.rule, ex.rows, mode=plan["mode"])
        for finding in findings:
            self.tgt.execute("""
                INSERT INTO ingest.quality_finding (batch_id, doc_type, level, rule, column_name,
                                                    failed_rows, checked_rows, message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (plan["batch_id"], finding.doc_type, finding.level, finding.rule, finding.column,
                  finding.failed_rows, finding.checked_rows, finding.message))
            log.warning("quality finding", extra={"doc_type": finding.doc_type, "level": finding.level,
                                                  "rule": finding.rule, "column": finding.column,
                                                  "failed_rows": finding.failed_rows})
        return findings

    def stage(self, plan: dict) -> dict:
        dt = self.doc_types[plan["doc_type"]]
        ex = self._extracts[plan["doc_type"]]
        stats = StageStats()
        try:
            docs = ex.documents
            page = self.settings.ingest_page_docs
            pages = [docs[i:i + page] for i in range(0, len(docs), page)] or [[]]
            for n, page_docs in enumerate(pages, start=1):
                with self.tgt.transaction():
                    stage_page(self.tgt, self.settings, dt, page_docs, stats,
                               rechunk_all=plan["rechunk_all"], batch_id=plan["batch_id"],
                               change_id=ex.to_change_id)
                    if n == len(pages):
                        if ex.mode == "full":
                            deactivate_absent(self.tgt, dt, [d.doc_id for d in docs], stats=stats,
                                              change_id=ex.to_change_id, batch_id=plan["batch_id"])
                        else:
                            deactivate(self.tgt, dt, [dt.doc_id(k) for k in ex.missing_keys], stats=stats,
                                       change_id=ex.to_change_id, batch_id=plan["batch_id"])
                        write_watermark(self.tgt, ex, plan["signature"], plan["batch_id"])
        except Exception as exc:
            self.fail_batch(plan, f"{type(exc).__name__}: {exc}")
            raise
        return {"status": "succeeded", **asdict(stats)}

    def finish_batch(self, plan: dict, result: dict) -> None:
        ex = self._extracts.get(plan["doc_type"])
        stats = StageStats(**{k: v for k, v in result.items() if k in StageStats.__annotations__})
        batch = BatchResult(doc_type=plan["doc_type"], batch_id=plan["batch_id"], mode=plan["mode"],
                            reason=plan["reason"], from_change_id=plan["from_change_id"],
                            to_change_id=ex.to_change_id if ex else plan["from_change_id"],
                            keys=result.get("keys", 0), status=result["status"], stats=stats,
                            seconds=round(time.monotonic() - self._started[plan["doc_type"]], 2))
        self.results.append(batch)
        self.tgt.execute("""
            UPDATE ingest.batch SET to_change_id = %s, status = %s, keys = %s, inserted = %s, updated = %s,
                   refreshed = %s, unchanged = %s, deactivated = %s, reactivated = %s, chunks_added = %s,
                   chunks_removed = %s, finished_at = now()
             WHERE batch_id = %s
        """, (batch.to_change_id, batch.status, batch.keys, stats.inserted, stats.updated, stats.refreshed,
              stats.unchanged, stats.deactivated, stats.reactivated, stats.chunks_added,
              stats.chunks_removed, plan["batch_id"]))
        log.info("ingest batch", extra={"doc_type": batch.doc_type, "batch_id": batch.batch_id,
                                        "mode": batch.mode, "status": batch.status, "keys": batch.keys,
                                        "window": f"({batch.from_change_id}, {batch.to_change_id}]",
                                        "inserted": stats.inserted, "updated": stats.updated,
                                        "versions": stats.versions_created,
                                        "deactivated": stats.deactivated,
                                        "chunks_added": stats.chunks_added,
                                        "chunks_moved": stats.chunks_moved, "seconds": batch.seconds})

    def fail_batch(self, plan: dict, error: str) -> None:
        self.tgt.execute("UPDATE ingest.batch SET status = 'failed', error = %s, finished_at = now() "
                         "WHERE batch_id = %s", (error[:2000], plan["batch_id"]))
        started = self._started.get(plan["doc_type"], time.monotonic())
        ex = self._extracts.get(plan["doc_type"])
        self.results.append(BatchResult(
            doc_type=plan["doc_type"], batch_id=plan["batch_id"], mode=plan["mode"], reason=plan["reason"],
            from_change_id=plan["from_change_id"],
            to_change_id=ex.to_change_id if ex else plan["from_change_id"], keys=0, status="failed",
            seconds=round(time.monotonic() - started, 2)))

    # ---- once per run
    def refresh_lexemes(self) -> int:
        self.lexemes = index.refresh_lexeme_stats(self.tgt)
        return self.lexemes

    def pending_embeddings(self) -> int:
        return embed_mod.pending_count(self.tgt)

    def pending_cost(self) -> dict:
        """What the embedding step would really send: distinct texts with no vector anywhere yet."""
        return embed_mod.pending_cost(self.tgt)

    def embed(self, max_chunks: int | None) -> dict:
        embedder = self.embedder_factory(self.settings)
        try:
            index.require_state(self.tgt, embedder)
            self.embed_stats = embed_mod.embed_pending(self.tgt, embedder, max_chunks=max_chunks)
        finally:
            embedder.close()
        return {"embedded": self.embed_stats.embedded, "reused": self.embed_stats.reused,
                "requests": self.embed_stats.requests, "status": self.embed_stats.status,
                "pending_after": self.embed_stats.pending_after,
                "quota_waits": self.embed_stats.quota_waits}

    def mark_indexed(self) -> int:
        return embed_mod.mark_indexed(self.tgt)


# ---------------------------------------------------------------- whole run
@contextlib.contextmanager
def exclusive(tgt: psycopg.Connection):
    """One writer at a time per vector database (ingest and embed share it): two runs would otherwise
    embed the same pending chunks twice."""
    if not scalar(tgt, "SELECT pg_try_advisory_lock(hashtext(%s))", (LOCK_NAME,)):
        raise AlreadyRunning("another ingest/embed is running against this vector database")
    try:
        yield
    finally:
        tgt.execute("SELECT pg_advisory_unlock(hashtext(%s))", (LOCK_NAME,))


@dataclass
class RunResult:
    batches: list[BatchResult]
    lexemes: int | None = None
    embed: embed_mod.EmbedStats | None = None
    findings: list[Finding] = field(default_factory=list)
    status: str = "succeeded"
    # Set when the run stopped at the approval gate (INGEST_EMBED_APPROVAL_CHUNKS): the cost it quoted
    # and the thread to resume. The documents and the watermark are already committed at this point.
    waiting: dict | None = None
    thread_id: str | None = None

    @property
    def paused(self) -> bool:
        return self.waiting is not None


def _result(ops: "PostgresOps", outcome) -> RunResult:
    return RunResult(batches=ops.results, lexemes=ops.lexemes, embed=ops.embed_stats,
                     findings=[Finding(level=f["level"], doc_type=f.get("doc_type"),
                                       message=f["message"]) for f in outcome.state.get("findings", [])],
                     status=outcome.state.get("status", "succeeded"),
                     waiting=outcome.waiting, thread_id=outcome.thread_id)


def run(tgt: psycopg.Connection, src: psycopg.Connection, settings, doc_types: Sequence[DocType], *,
        full: bool = False, embed: bool = True, max_chunks: int | None = None,
        embedder_factory: Callable = embedding_provider, thread_id: str | None = None) -> RunResult:
    """One ingestion run, through the LangGraph graph."""
    from crawlerrag.ingest import graph as ingest_graph
    with exclusive(tgt):
        ops = PostgresOps(tgt, src, settings, doc_types, embedder_factory=embedder_factory)
        outcome = ingest_graph.run_ingest(
            ops, [dt.doc_type for dt in doc_types], full=full, embed=embed, max_chunks=max_chunks,
            thread_id=thread_id, checkpointer_settings=settings,
            approval_chunks=getattr(settings, "ingest_embed_approval_chunks", 0) if embed else 0)
        return _result(ops, outcome)


def answer_approval(tgt: psycopg.Connection, src: psycopg.Connection, settings,
                    doc_types: Sequence[DocType], *, thread_id: str, approved: bool,
                    embedder_factory: Callable = embedding_provider) -> RunResult:
    """Tell a run parked at the approval gate to go ahead (or not) and let it finish."""
    from crawlerrag.ingest import graph as ingest_graph
    with exclusive(tgt):
        ops = PostgresOps(tgt, src, settings, doc_types, embedder_factory=embedder_factory)
        outcome = ingest_graph.resume_ingest(ops, thread_id=thread_id, checkpointer_settings=settings,
                                            answer=approved)
        return _result(ops, outcome)


# ---------------------------------------------------------------- read-only preview
@dataclass
class Plan:
    doc_type: str
    mode: str
    reason: str
    from_change_id: int
    to_change_id: int
    keys: int
    new_docs: int
    changed_docs: int
    unchanged_docs: int
    new_versions: int             # how many of them would open a Type 2 version
    chunks_to_embed: int          # distinct chunk texts with no vector anywhere in the index yet
    chars_to_embed: int


def plan(tgt: psycopg.Connection, src: psycopg.Connection, settings, doc_types: Sequence[DocType], *,
         full: bool = False) -> list[Plan]:
    """What `ingest` would do and roughly what it would cost, without writing anything."""
    out: list[Plan] = []
    known = {r["text_hash"] for r in tgt.execute(
        "SELECT DISTINCT text_hash FROM rag.chunk WHERE embedding IS NOT NULL").fetchall()}
    for dt in doc_types:
        sig = dt.signature(settings)
        wm = read_watermark(tgt, dt.doc_type)
        mode, reason = choose_mode(wm, sig, full=full)
        rechunk_all = wm is not None and wm["signature"] != sig
        from_id = int(wm["change_id"]) if wm else 0
        ex = extract(src, dt, mode=mode, from_change_id=from_id)
        previous = _current_versions(tgt, [d.doc_id for d in ex.documents])
        stored = _stored_chunk_hashes(tgt, previous) if rechunk_all else {}
        new = changed = unchanged = versions = 0
        texts: dict[str, int] = {}
        for doc in ex.documents:
            prev = previous.get(doc.doc_id)
            action = decide(prev, doc, dt.scd2, rechunk_all=rechunk_all)
            if action.kind == RECHUNK and stored.get(prev.doc_sk) == _chunk_hashes(settings, doc):
                action = decide(prev, doc, dt.scd2, rechunk_all=False)      # same chunks: nothing to do
            if action.kind == UNCHANGED:
                unchanged += 1
                continue
            versions += action.opens_version
            if action.kind == INSERT:
                new += 1
            elif action.kind in (NEW_VERSION, RECHUNK):
                changed += 1
            if not action.writes_chunks:
                continue
            for text in split_text(doc.text, size=settings.rag_chunk_chars,
                                   overlap=settings.rag_chunk_overlap):
                h = text_hash(text)
                if h not in known:
                    texts[h] = len(text)
        out.append(Plan(dt.doc_type, mode, reason, from_id, ex.to_change_id,
                        len(ex.keys) if ex.keys is not None else len(ex.documents),
                        new, changed, unchanged, versions, len(texts), sum(texts.values())))
    return out
