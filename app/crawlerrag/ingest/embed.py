"""Embedding stage: chunks without a vector -> the embedding model -> rag.chunk.embedding.

Resumable by construction: the queue is "embedding IS NULL", every batch is committed on its own, so an
interruption loses at most the batch in flight and a re-run continues where it stopped. Each distinct
chunk text is sent once: text already embedded elsewhere is copied, and identical pending texts share
one request slot. Each run is recorded in ingest.embed_run (texts, requests, characters, tokens).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import psycopg

from crawlerrag.db import scalar
from crawlerrag.rag.index import _vector_literal
from crawlerrag.rag.providers import EmbeddingProvider, ProviderError

log = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    run_id: int = 0
    reused: int = 0                     # chunks filled with a vector of identical text, not sent
    embedded: int = 0                   # distinct texts sent to the model
    requests: int = 0
    input_chars: int = 0
    input_tokens: int | None = None     # reported by the provider (Vertex: statistics.token_count)
    truncated: int | None = None        # texts the provider cut at its input limit
    pending_after: int = 0
    quota_waits: int = 0                # pauses for a per-minute quota (HTTP 429) before the batch went through
    status: str = "succeeded"
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def pending_count(conn: psycopg.Connection) -> int:
    return int(scalar(conn, "SELECT count(*) FROM rag.chunk WHERE embedding IS NULL") or 0)


PENDING_COST = """
SELECT count(*) AS chunks, coalesce(sum(chars), 0)::bigint AS chars
  FROM (SELECT DISTINCT ON (text_hash) text_hash, length(text) AS chars
          FROM rag.chunk WHERE embedding IS NULL ORDER BY text_hash) d
 WHERE NOT EXISTS (SELECT 1 FROM rag.chunk k
                    WHERE k.text_hash = d.text_hash AND k.embedding IS NOT NULL)
"""


def pending_cost(conn: psycopg.Connection) -> dict:
    """What the next embedding step would actually send to the model.

    Not the same as :func:`pending_count`: identical chunk text is embedded once and copied
    (``reuse_embeddings``), so a run with 94 pending chunks may only pay for 60 of them. The approval
    gate quotes this number, because quoting a bigger one would teach people to ignore it.
    """
    row = conn.execute(PENDING_COST).fetchone()
    return {"chunks": int(row["chunks"]), "chars": int(row["chars"])}


def reuse_embeddings(conn: psycopg.Connection, model: str) -> int:
    """Copy vectors between chunks whose text is byte-identical (same model only)."""
    return int(scalar(conn, """
        WITH source AS (
            SELECT DISTINCT ON (text_hash) text_hash, embedding
              FROM rag.chunk
             WHERE embedding IS NOT NULL AND embed_model = %(model)s
             ORDER BY text_hash, chunk_id
        ), copied AS (
            UPDATE rag.chunk c
               SET embedding = s.embedding, embed_model = %(model)s, embedded_at = now()
              FROM source s
             WHERE c.embedding IS NULL AND c.text_hash = s.text_hash
         RETURNING 1)
        SELECT count(*) FROM copied
    """, {"model": model}) or 0)


def mark_indexed(conn: psycopg.Connection) -> int:
    """Document versions whose every chunk has a vector."""
    return int(scalar(conn, """
        WITH done AS (
            UPDATE rag.document d SET indexed_at = now()
             WHERE d.indexed_at IS NULL
               AND EXISTS (SELECT 1 FROM rag.chunk c WHERE c.doc_sk = d.doc_sk)
               AND NOT EXISTS (SELECT 1 FROM rag.chunk c WHERE c.doc_sk = d.doc_sk AND c.embedding IS NULL)
         RETURNING 1)
        SELECT count(*) FROM done
    """) or 0)


QUOTA_WAIT_S = 65.0       # Vertex AI's embedding quota is per minute: wait past one window
MAX_QUOTA_WAITS = 30      # consecutive waits before giving up (the quota is not coming back)


def embed_pending(conn: psycopg.Connection, provider: EmbeddingProvider, *, max_chunks: int | None = None,
                  quota_wait_s: float = QUOTA_WAIT_S, max_quota_waits: int = MAX_QUOTA_WAITS,
                  sleep=time.sleep) -> EmbedStats:
    """Embed every pending chunk (or ``max_chunks`` texts).

    A batch refused with HTTP 429 after the provider's own retries is a quota window, not a failure:
    the 2026-10-04 run on Vertex AI hit "global_embed_content_requests_per_minute_per_base_model" at
    about 600 texts a minute, and the provider's backoff (a few seconds) ends inside that minute. The
    stage then waits ``quota_wait_s`` and sends the same batch again; any other error stops the run."""
    stats = EmbedStats()
    consecutive_waits = 0
    started = time.monotonic()
    stats.run_id = int(scalar(conn, "INSERT INTO ingest.embed_run (embed_model) VALUES (%s) RETURNING run_id",
                              (provider.model,)))
    stats.reused = reuse_embeddings(conn, provider.model)
    batch_size = max(1, min(provider.batch_size, 256))
    while True:
        remaining = None if max_chunks is None else max_chunks - stats.embedded
        if remaining is not None and remaining <= 0:
            break
        rows = conn.execute("SELECT DISTINCT ON (text_hash) text_hash, text FROM rag.chunk WHERE embedding IS NULL "
                            "ORDER BY text_hash LIMIT %s",
                            (batch_size if remaining is None else min(batch_size, remaining),)).fetchall()
        if not rows:
            break
        texts = [r["text"] for r in rows]
        try:
            vectors = provider.embed(texts)
        except ProviderError as exc:
            if getattr(exc, "status", None) == 429 and consecutive_waits < max_quota_waits:
                consecutive_waits += 1
                stats.quota_waits += 1
                log.warning("embedding quota reached, waiting for the next window",
                            extra={"wait_s": quota_wait_s, "consecutive": consecutive_waits,
                                   "embedded": stats.embedded})
                sleep(quota_wait_s)
                continue
            stats.errors.append(str(exc))
            stats.status = "failed"
            log.error("embedding batch failed, stopping", extra={"error": str(exc)[:500]})
            break
        consecutive_waits = 0
        with conn.transaction():
            for row, vector in zip(rows, vectors):
                filled = conn.execute("UPDATE rag.chunk SET embedding = %s, embed_model = %s, embedded_at = now() "
                                      "WHERE text_hash = %s AND embedding IS NULL",
                                      (_vector_literal(vector), provider.model, row["text_hash"])).rowcount
                stats.reused += max(0, filled - 1)
        stats.embedded += len(rows)
        stats.requests += 1
        stats.input_chars += sum(len(t) for t in texts)
        usage = getattr(provider, "last_usage", None)
        if usage:
            stats.input_tokens = (stats.input_tokens or 0) + usage["tokens"]
            stats.truncated = (stats.truncated or 0) + usage["truncated"]
        if stats.requests % 50 == 0:
            log.info("embedding progress", extra={"embedded": stats.embedded, "pending": pending_count(conn),
                                                  "elapsed_s": round(time.monotonic() - started, 1)})
    stats.pending_after = pending_count(conn)
    if stats.status == "succeeded" and stats.pending_after:
        stats.status = "stopped"            # --max-chunks reached; the rest waits for the next run
    stats.seconds = round(time.monotonic() - started, 1)
    conn.execute("""
        UPDATE ingest.embed_run SET reused = %s, embedded = %s, requests = %s, input_chars = %s, input_tokens = %s,
               truncated = %s, quota_waits = %s, pending_after = %s, status = %s, error = %s, finished_at = now()
         WHERE run_id = %s
    """, (stats.reused, stats.embedded, stats.requests, stats.input_chars, stats.input_tokens, stats.truncated,
          stats.quota_waits, stats.pending_after, stats.status, "; ".join(stats.errors)[:2000] or None,
          stats.run_id))
    mark_indexed(conn)
    log.info("embedding finished", extra={"embedded": stats.embedded, "reused": stats.reused,
                                          "requests": stats.requests, "pending": stats.pending_after,
                                          "elapsed_s": stats.seconds})
    return stats
