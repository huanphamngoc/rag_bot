"""Hybrid retrieval: pgvector nearest neighbours fused with PostgreSQL full text search.

Neither half is enough on its own for this data. Vector search finds "toys that can
choke a child" when the recall says "small parts pose a choking hazard"; lexical search
is the only one that reliably finds ``NDC 0093-1234`` or ``Ibuprofen 800 mg``, because an
embedding blurs exactly the identifiers this dataset is full of. The two rankings are
merged with Reciprocal Rank Fusion, which needs no score calibration between them.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import psycopg

from crawlerrag.rag import tracing
from crawlerrag.rag.providers import EmbeddingProvider
from crawlerrag.rag.index import _vector_literal, require_state

log = logging.getLogger(__name__)

RRF_K = 60           # the usual damping constant: rank 1 scores 1/61, rank 10 scores 1/70
MAX_PER_DOC = 2      # one long document must not crowd out every other answer
MAX_LEXEME_DF = 0.10  # lexical search ignores words found in more than 10% of chunks ...
MAX_LEXEMES = 8       # ... and keeps at most the 8 rarest of the rest (see text_search)


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    doc_type: str
    source_id: str
    title: str
    url: str | None
    text: str
    metadata: dict[str, Any]
    score: float = 0.0
    vector_rank: int | None = None
    text_rank: int | None = None
    distance: float | None = None
    lexical_rank: float | None = None

    @property
    def matched_by(self) -> str:
        if self.vector_rank and self.text_rank:
            return "both"
        return "vector" if self.vector_rank else "text"


@dataclass
class Retrieval:
    question: str
    hits: list[Hit] = field(default_factory=list)
    doc_types: list[str] = field(default_factory=list)
    vector_candidates: int = 0
    text_candidates: int = 0


def _filter_sql(doc_types: Sequence[str] | None, filters: dict[str, Any] | None,
                params: dict[str, Any]) -> str:
    # rag.document keeps every version (SCD Type 2); retrieval only ever reads the current one.
    sql = " AND d.is_current AND d.is_active"
    if doc_types:
        sql += " AND d.doc_type = ANY(%(doc_types)s)"
        params["doc_types"] = list(doc_types)
    exact: dict[str, Any] = {}
    for n, (key, value) in enumerate(sorted((filters or {}).items())):
        # jsonb containment is type-exact, and parse_filters has to guess the type: "26649" becomes an
        # int, "true" a bool. The same value is often stored as a string - cpsc_recall.recall_number is
        # "26649", drug_recall.event_id is "99824" - so an int filter on them matched nothing at all and
        # said so with an empty result. Both forms are tried; jsonb_path_ops serves each from
        # document_metadata_idx, so the plan is a BitmapOr of two index scans.
        if isinstance(value, (int, bool)):
            as_given, as_text = f"filter{n}", f"filter{n}_text"
            sql += f" AND (d.metadata @> %({as_given})s::jsonb OR d.metadata @> %({as_text})s::jsonb)"
            params[as_given] = json.dumps({key: value})
            params[as_text] = json.dumps({key: "true" if value is True else
                                          "false" if value is False else str(value)})
        else:
            exact[key] = value
    if exact:
        sql += " AND d.metadata @> %(filters)s::jsonb"
        params["filters"] = json.dumps(exact)
    return sql


def parse_filters(pairs: Sequence[str] | None) -> dict[str, Any]:
    """``--filter state=CA --filter year=2024`` -> metadata containment, numbers kept numeric."""
    out: dict[str, Any] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(f"filter must be key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            raise ValueError(f"filter without a key: {pair!r}")
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
        else:
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = value
    return out


SELECT_COLUMNS = """
    c.chunk_id, d.doc_id, c.text, d.doc_type, d.source_id, d.title, d.url, d.metadata
"""


def _scope(args: dict) -> dict:
    return {"limit": args.get("limit"), "doc_types": args.get("doc_types"), "filters": args.get("filters")}


@tracing.traceable(run_type="retriever", name="vector_search", inputs=_scope, outputs=tracing.row_ids)
def vector_search(conn: psycopg.Connection, vector: list[float], *, limit: int,
                  doc_types: Sequence[str] | None = None, filters: dict[str, Any] | None = None,
                  ef_search: int = 100) -> list[dict]:
    params: dict[str, Any] = {"q": _vector_literal(vector), "limit": limit}
    where = _filter_sql(doc_types, filters, params)
    # These are session GUCs and SET takes no bind parameters, hence the validated ints.
    conn.execute(f"SET hnsw.ef_search = {max(limit, int(ef_search))}")
    # A restrictive filter (one doc type, state=CA, year=2023) is applied *after* the ANN scan, so a
    # plain scan can return 2 usable rows out of 40. Iterative scan keeps walking the graph until the
    # filter is satisfied; relaxed_order is fine because RRF re-ranks everything anyway.
    restrictive = bool(filters) or bool(doc_types)
    conn.execute(f"SET hnsw.iterative_scan = {'relaxed_order' if restrictive else 'off'}")
    if restrictive:
        conn.execute(f"SET hnsw.max_scan_tuples = {max(20000, limit * 500)}")
    return conn.execute(
        f"""
        SELECT {SELECT_COLUMNS}, c.embedding <=> %(q)s::vector AS distance
          FROM rag.chunk c JOIN rag.document d ON d.doc_sk = c.doc_sk
         WHERE c.embedding IS NOT NULL{where}
         ORDER BY c.embedding <=> %(q)s::vector
         LIMIT %(limit)s
        """,
        params,
    ).fetchall()


@tracing.traceable(run_type="retriever", name="text_search",
                   inputs=lambda a: {"question": a.get("question"), **_scope(a)}, outputs=tracing.row_ids)
def text_search(conn: psycopg.Connection, question: str, *, limit: int,
                doc_types: Sequence[str] | None = None,
                filters: dict[str, Any] | None = None) -> list[dict]:
    """Lexical half of the hybrid, with OR semantics on purpose.

    ``websearch_to_tsquery``/``plainto_tsquery`` AND every term together, so a natural
    question of eight words matches almost nothing and the lexical side contributes
    nothing to the fusion. Here the question is reduced to its lexemes (stemmed, stop
    words dropped by the same ``english`` configuration the index uses) and OR-ed;
    ``ts_rank_cd`` then ranks a chunk covering more of those lexemes higher.

    Only the ``MAX_LEXEMES`` rarest lexemes found in at most ``MAX_LEXEME_DF`` of the chunks
    are kept (counts in ``rag.lexeme_stat``). Words like "recall" or "product" are in ~90%
    of chunks: they add no signal, and OR-ing them matched 97% of the corpus and made
    ``ts_rank_cd`` take 22-39 s on a long rewritten follow-up (0.7 s after the cut). The
    lexical side exists for identifiers and rare terms; meaning is the vector side's job.
    """
    params: dict[str, Any] = {"q": question, "limit": limit, "max_df": MAX_LEXEME_DF,
                              "max_lexemes": MAX_LEXEMES}
    where = _filter_sql(doc_types, filters, params)
    return conn.execute(
        f"""
        WITH total AS (
            SELECT count(*) AS n FROM rag.chunk
        ), kept AS (
            SELECT lex
              FROM unnest(tsvector_to_array(to_tsvector('english', %(q)s))) AS lex
              LEFT JOIN rag.lexeme_stat s ON s.lexeme = lex, total
             WHERE coalesce(s.ndoc, 0) <= %(max_df)s * total.n
             ORDER BY coalesce(s.ndoc, 0), lex
             LIMIT %(max_lexemes)s
        ), lexemes AS (
            SELECT string_agg(quote_literal(lex), ' | ') AS expr FROM kept
        ), query AS (
            SELECT to_tsquery('english', expr) AS tsq FROM lexemes WHERE expr IS NOT NULL
        )
        SELECT {SELECT_COLUMNS}, ts_rank_cd(c.tsv, query.tsq) AS lexical_rank
          FROM rag.chunk c JOIN rag.document d ON d.doc_sk = c.doc_sk, query
         WHERE c.tsv @@ query.tsq{where}
         ORDER BY lexical_rank DESC, c.chunk_id
         LIMIT %(limit)s
        """,
        params,
    ).fetchall()


def _hit(row: dict) -> Hit:
    metadata = row["metadata"] or {}
    return Hit(chunk_id=row["chunk_id"], doc_id=row["doc_id"], doc_type=row["doc_type"],
               source_id=row["source_id"], title=row["title"], url=row["url"], text=row["text"],
               metadata=metadata if isinstance(metadata, dict) else json.loads(metadata))


def fuse(vector_rows: Sequence[dict], text_rows: Sequence[dict], *, top_k: int,
         max_per_doc: int = MAX_PER_DOC) -> list[Hit]:
    hits: dict[int, Hit] = {}
    for rank, row in enumerate(vector_rows, start=1):
        hit = hits.setdefault(row["chunk_id"], _hit(row))
        hit.vector_rank, hit.distance = rank, row.get("distance")
        hit.score += 1.0 / (RRF_K + rank)
    for rank, row in enumerate(text_rows, start=1):
        hit = hits.setdefault(row["chunk_id"], _hit(row))
        hit.text_rank, hit.lexical_rank = rank, row.get("lexical_rank")
        hit.score += 1.0 / (RRF_K + rank)

    ordered = sorted(hits.values(), key=lambda h: (-h.score, h.chunk_id))
    kept: list[Hit] = []
    per_doc: dict[str, int] = {}
    for hit in ordered:
        if per_doc.get(hit.doc_id, 0) >= max_per_doc:
            continue
        per_doc[hit.doc_id] = per_doc.get(hit.doc_id, 0) + 1
        kept.append(hit)
        if len(kept) >= top_k:
            break
    return kept


@tracing.traceable(run_type="embedding", name="embed_query",
                   inputs=lambda a: {"input": a.get("question")}, outputs=lambda v: {"dimensions": len(v)})
def _embed_query(provider: EmbeddingProvider, question: str) -> list[float]:
    tracing.annotate_model(provider)
    return provider.embed([question], query=True)[0]


def _search_inputs(args: dict) -> dict:
    return {"question": args.get("question"), "doc_types": args.get("doc_types"), "top_k": args.get("top_k"),
            "candidates": args.get("candidates"), "filters": args.get("filters")}


def _search_outputs(result: "Retrieval") -> list[dict]:
    # A bare list of documents is what MLflow renders as a retriever result; the candidate
    # counts go on the span as attributes (see search()).
    return tracing.documents(result.hits)


@tracing.traceable(run_type="retriever", name="hybrid_retrieve", inputs=_search_inputs, outputs=_search_outputs)
def search(conn: psycopg.Connection, settings, provider: EmbeddingProvider, question: str, *,
           doc_types: Sequence[str] | None = None, top_k: int | None = None,
           candidates: int | None = None, filters: dict[str, Any] | None = None) -> Retrieval:
    question = (question or "").strip()
    if not question:
        raise ValueError("empty question")
    require_state(conn, provider)
    top_k = top_k or settings.rag_top_k
    candidates = candidates or settings.rag_candidates
    vector = _embed_query(provider, question)
    vector_rows = vector_search(conn, vector, limit=candidates, doc_types=doc_types, filters=filters)
    text_rows = text_search(conn, question, limit=candidates, doc_types=doc_types, filters=filters)
    hits = fuse(vector_rows, text_rows, top_k=top_k)
    tracing.annotate({"vector_candidates": len(vector_rows), "text_candidates": len(text_rows)})
    log.info("retrieved", extra={"question_chars": len(question), "vector": len(vector_rows),
                                 "text": len(text_rows), "kept": len(hits)})
    return Retrieval(question=question, hits=hits, doc_types=list(doc_types or []),
                     vector_candidates=len(vector_rows), text_candidates=len(text_rows))
