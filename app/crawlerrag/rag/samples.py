"""Sample questions built from records that are really in the index.

A gate that says "ask something else" and stops there leaves the person guessing, and a hand-written
list of examples in a YAML file goes stale the moment a recall is superseded. So when the clarify step
asks again, the examples it offers are built from documents this index holds *right now*, chosen by the
words the person already typed: "insulin" comes back with the insulin recalls that are actually there.

Deliberately lexical only - no embedding call, so asking again costs nothing. It reuses the rare-lexeme
trick of :func:`crawlerrag.rag.retrieve.text_search` (words found in more than ``MAX_LEXEME_DF`` of the
chunks carry no signal), over the same ``chunk_tsv_idx`` the answers come from.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import psycopg

log = logging.getLogger(__name__)

MAX_LEXEME_DF = 0.10      # same cut as retrieve.text_search: "recall" is in ~90% of chunks
MAX_LEXEMES = 8
SCAN_LIMIT = 200          # chunks considered before collapsing to documents, so the cost is bounded

MATCHED_SQL = f"""
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
), matched AS (
    SELECT c.doc_sk, ts_rank_cd(c.tsv, query.tsq) AS rank
      FROM rag.chunk c, query
     WHERE c.tsv @@ query.tsq
     ORDER BY rank DESC, c.chunk_id
     LIMIT {SCAN_LIMIT}
), best AS (
    SELECT DISTINCT ON (d.doc_id) d.doc_id, d.doc_type, d.title, d.metadata, m.rank
      FROM matched m JOIN rag.document d ON d.doc_sk = m.doc_sk
     WHERE d.is_current AND d.is_active{{scope}}
     ORDER BY d.doc_id, m.rank DESC
)
SELECT doc_id, doc_type, title, metadata FROM best ORDER BY rank DESC LIMIT %(limit)s
"""

RECENT_SQL = """
SELECT d.doc_id, d.doc_type, d.title, d.metadata
  FROM rag.document d
 WHERE d.is_current AND d.is_active{scope}
 ORDER BY d.metadata->>'recall_date' DESC NULLS LAST, d.doc_id DESC
 LIMIT %(limit)s
"""

# A sample question has to be one that RETRIEVAL CAN ANSWER, which rules out the obvious design.
# Measured on the real index: a recall number inside a sentence finds its own document 1 time in 5.
# `to_tsvector('english', 'Why was drug recall D-0445-2024 issued?')` splits the identifier into
# '-0445', '-2024' and 'd', and drops 'd' as too common (18,303 chunks), so the lexical side goes
# looking for '-0445 | -2024 | issu' and matches half the corpus. The identifier works alone, but not
# glued into a question.
#
# So the subject comes from the title instead - a firm name, a product - which is exactly what both
# halves of the hybrid are good at. The identifier is still returned alongside, as context to show.
TEMPLATES = {
    "drug_recall": "Why did {subject} recall a drug?",
    "cpsc_recall": "What hazard did CPSC report for {subject}?",
}
FALLBACK = "What do the records say about {subject}?"

DRUG_PREFIX = "FDA drug recall "
CPSC_PREFIX = "CPSC consumer product recall: "
# CPSC headlines read "<firm> Recalls <product> Due to <hazard>; <extra>". Cutting at the hazard keeps
# the words that identify the record and drops the ones every recall shares.
CPSC_CUTS = (" Due to ", " Due To ", " due to ", ";")
UNKNOWN = "unknown firm"


def _scope(doc_types: Sequence[str] | None, params: dict) -> str:
    if not doc_types:
        return ""
    params["doc_types"] = list(doc_types)
    return " AND d.doc_type = ANY(%(doc_types)s)"


def _subject(row: dict) -> str | None:
    """The part of the title worth asking about: the firm for a drug recall, the headline for a CPSC one."""
    title = (row.get("title") or "").strip()
    if row.get("doc_type") == "drug_recall":
        firm = title.split(" - ", 1)[1].strip() if " - " in title else ""
        return firm or None if firm.lower() != UNKNOWN else None
    headline = title[len(CPSC_PREFIX):] if title.startswith(CPSC_PREFIX) else title
    for cut in CPSC_CUTS:
        headline = headline.split(cut, 1)[0]
    headline = headline.strip().rstrip(".,;")
    # "Galanz Americas Recalls Retro Refrigerators" -> "Retro Refrigerators recalled by Galanz Americas"
    if " Recalls " in headline:
        firm, _, product = headline.partition(" Recalls ")
        if firm.strip() and product.strip():
            return f"{product.strip()} recalled by {firm.strip()}"
    return headline or None


def _question(row: dict) -> str | None:
    subject = _subject(row)
    if not subject or len(subject) > 120:
        return None
    template = TEMPLATES.get(row["doc_type"], FALLBACK)
    return template.format(subject=subject)


def build(conn: psycopg.Connection, question: str, *, doc_types: Sequence[str] | None = None,
          limit: int = 4) -> list[dict[str, Any]]:
    """Up to ``limit`` answerable questions, about records matching ``question`` where any do.

    Never raises: a clarify answer without examples is worse than one with them, but an exception here
    would turn a helpful nudge into a 500.
    """
    params: dict[str, Any] = {"q": question or "", "limit": limit, "max_df": MAX_LEXEME_DF,
                              "max_lexemes": MAX_LEXEMES}
    rows: list[dict] = []
    try:
        if (question or "").strip():
            scope = _scope(doc_types, params)
            rows = conn.execute(MATCHED_SQL.format(scope=scope), params).fetchall()
        if not rows:
            plain: dict[str, Any] = {"limit": limit}
            rows = conn.execute(RECENT_SQL.format(scope=_scope(doc_types, plain)), plain).fetchall()
    except psycopg.Error:
        log.warning("could not build sample questions", exc_info=True)
        return []

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        text = _question(row)
        if text and text not in seen:
            seen.add(text)
            out.append({"question": text, "doc_id": row["doc_id"], "title": row["title"]})
    return out
