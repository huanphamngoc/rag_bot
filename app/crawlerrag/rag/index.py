"""State of the vector index in Postgres (pgvector): model pinning, ANN index, keyword statistics.

Ported from the crawler's rag/store.py. Building documents and embedding them moved to
crawlerrag.ingest (incremental, driven by the crawler's change log).
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

import psycopg

from crawlerrag.db import scalar
from crawlerrag.rag.providers import EmbeddingProvider

log = logging.getLogger(__name__)

VECTOR_TYPE = re.compile(r"^vector(?:\((\d+)\))?$")
HNSW_INDEX = "chunk_embedding_idx"


class IndexStateError(RuntimeError):
    """The stored vectors and the configured model do not belong to the same vector space."""


def current_dim(conn: psycopg.Connection) -> int | None:
    row = conn.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod) AS type
          FROM pg_attribute a
         WHERE a.attrelid = 'rag.chunk'::regclass AND a.attname = 'embedding'
        """
    ).fetchone()
    match = VECTOR_TYPE.match(row["type"]) if row else None
    return int(match.group(1)) if match and match.group(1) else None


def read_state(conn: psycopg.Connection) -> dict | None:
    return conn.execute("SELECT * FROM rag.index_state WHERE id = 1").fetchone()


def embedded_count(conn: psycopg.Connection) -> int:
    return int(scalar(conn, "SELECT count(*) FROM rag.chunk WHERE embedding IS NOT NULL") or 0)


def init_index(conn: psycopg.Connection, provider: EmbeddingProvider, *, force: bool = False,
               dim: int | None = None) -> dict:
    """Pin the vector dimension and (re)build the ANN index. Idempotent."""
    dim = dim or provider.probe_dim()
    if not 1 <= dim <= 16000:
        raise IndexStateError(f"model returned an unusable embedding dimension: {dim}")
    state, stored, have_dim = read_state(conn), embedded_count(conn), current_dim(conn)
    changed = state is not None and (state["embed_provider"] != provider.provider
                                    or state["embed_model"] != provider.model
                                    or state["embed_dim"] != dim)
    if changed and stored and not force:
        raise IndexStateError(
            f"the index holds {stored:,} vectors from {state['embed_provider']}/{state['embed_model']} "
            f"(dim {state['embed_dim']}), but the configuration now says {provider.provider}/{provider.model} "
            f"(dim {dim}). Vectors from different models are not comparable. Re-run with --force to discard "
            "the stored embeddings and re-embed."
        )
    with conn.transaction():
        if changed and stored:
            log.warning("discarding embeddings from a different model",
                        extra={"vectors": stored, "old_model": state["embed_model"], "new_model": provider.model})
            conn.execute("UPDATE rag.chunk SET embedding = NULL, embed_model = NULL, embedded_at = NULL")
            conn.execute("UPDATE rag.document SET indexed_at = NULL")
        if have_dim != dim:
            # The ANN index is tied to the column type, so it goes away and comes back.
            conn.execute(f"DROP INDEX IF EXISTS rag.{HNSW_INDEX}")
            conn.execute(f"ALTER TABLE rag.chunk ALTER COLUMN embedding TYPE vector({dim})")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS {HNSW_INDEX} ON rag.chunk "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
        conn.execute(
            """
            INSERT INTO rag.index_state (id, embed_provider, embed_model, embed_dim, updated_at)
            VALUES (1, %s, %s, %s, now())
            ON CONFLICT (id) DO UPDATE SET embed_provider = EXCLUDED.embed_provider,
                                           embed_model = EXCLUDED.embed_model,
                                           embed_dim = EXCLUDED.embed_dim,
                                           updated_at = now()
            """,
            (provider.provider, provider.model, dim),
        )
    log.info("index ready", extra={"provider": provider.provider, "model": provider.model, "dim": dim})
    return {"provider": provider.provider, "model": provider.model, "dim": dim,
            "vectors_discarded": stored if changed else 0}


def require_state(conn: psycopg.Connection, provider: EmbeddingProvider | None = None) -> dict:
    state = read_state(conn)
    if state is None:
        raise IndexStateError("no index yet - run \"crawlerrag init\" first.")
    if provider is not None and (state["embed_provider"] != provider.provider
                                 or state["embed_model"] != provider.model):
        raise IndexStateError(
            f"the index was built with {state['embed_provider']}/{state['embed_model']} but the configuration "
            f"says {provider.provider}/{provider.model}. Searching across two vector spaces returns noise; "
            "either restore the old setting or re-run \"crawlerrag init --force\" and re-embed."
        )
    return state


def _vector_literal(vector: Iterable[float]) -> str:
    """pgvector accepts its own text form; this avoids depending on the pgvector Python adapter."""
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def refresh_lexeme_stats(conn: psycopg.Connection) -> int:
    """Recount chunks per lexeme; the keyword search reads it to skip words found in most chunks.
    Run whenever the set of chunks changed (the crawler measured ~1.3 s for 36k chunks)."""
    with conn.transaction():
        conn.execute("TRUNCATE rag.lexeme_stat")
        return conn.execute("INSERT INTO rag.lexeme_stat (lexeme, ndoc) "
                            "SELECT word, ndoc FROM ts_stat('SELECT tsv FROM rag.chunk')").rowcount


def index_stats(conn: psycopg.Connection) -> list[dict]:
    """Per document type, counted over the current version of each document.

    ``versions`` is every row of rag.document including closed ones, so the gap between it and
    ``documents`` is how much history the index carries.
    """
    return conn.execute("""
        SELECT d.doc_type,
               count(*) FILTER (WHERE d.is_current)                            AS documents,
               count(*) FILTER (WHERE d.is_current AND d.is_active)            AS active,
               count(*)                                                        AS versions,
               count(*) FILTER (WHERE d.is_current AND d.indexed_at IS NOT NULL) AS indexed,
               coalesce(sum(c.chunks), 0)                                      AS chunks,
               coalesce(sum(c.embedded), 0)                                    AS embedded
          FROM rag.document d
          LEFT JOIN LATERAL (
              SELECT count(*) AS chunks, count(embedding) AS embedded
                FROM rag.chunk ch WHERE ch.doc_sk = d.doc_sk
          ) c ON true
         GROUP BY d.doc_type
         ORDER BY d.doc_type
    """).fetchall()
