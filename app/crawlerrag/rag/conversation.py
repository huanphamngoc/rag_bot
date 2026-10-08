"""Multi-turn conversations: history, follow-up rewriting and persistence.

A follow-up such as "and what about the second firm?" retrieves nothing useful on its own - the
embedding of "second firm" is not near any recall. So every turn after the first runs in two steps:

1. **condense**: a small LLM call rewrites the follow-up into a standalone question using the
   recent turns ("Which drugs did Cantrell Drug Company recall, and why?"); retrieval uses that;
2. **answer**: the model sees the earlier *questions*, the fresh excerpts, the user's own words and
   the standalone form. Earlier answers are deliberately left out of this step: shown a wrong
   answer from the previous turn, the model repeated it word for word. Facts come only from the
   excerpts retrieved for this turn.

History is read back from ``rag.query_log`` (rows of the conversation, oldest first), so nothing
is stored twice and a conversation survives the process: ``crawlerrag ask --conversation <id>`` or
``crawlerrag chat --conversation <id>`` continue it later.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Sequence

import psycopg

from crawlerrag.rag import tracing
from crawlerrag.rag.providers import ChatProvider, ProviderError

log = logging.getLogger(__name__)

MAX_TURNS = 6           # turns of history shown to the model (and to the rewriter)
ANSWER_CHARS = 400      # past answers, shortened, are shown to the rewrite step only
TITLE_CHARS = 120

CONDENSE_SYSTEM = """You rewrite the user's latest message into one standalone search question \
for a database of US regulatory records (FDA drug recalls, CPSC product recalls, drug directory, \
healthcare organisations, insurance plans, branded foods).

Rules:
1. Resolve every reference to earlier turns ("it", "they", "them", "that firm", "the second one", \
"same year") into the explicit names, identifiers, dates or products it refers to.
2. Carry the subject of the conversation forward - the firm, product, state, year or recall \
number being discussed - even when the latest message does not repeat it.
3. Be concise: name the firm or product family once rather than listing every item from an \
earlier answer. At most about 30 words.
4. Keep recall numbers, NDC codes, firm and product names exactly as written.
5. If the latest message is already standalone, return it unchanged.
6. Keep the language of the latest message.
7. Output only the question - no preamble, no quotes, no answer."""


@dataclass(frozen=True)
class Turn:
    turn: int
    question: str
    answer: str


# ---------------------------------------------------------------- persistence
def create(conn: psycopg.Connection) -> str:
    row = conn.execute("INSERT INTO rag.conversation DEFAULT VALUES RETURNING conversation_id").fetchone()
    return str(row["conversation_id"])


def exists(conn: psycopg.Connection, conversation_id: str) -> bool:
    try:
        return conn.execute("SELECT 1 FROM rag.conversation WHERE conversation_id = %s::uuid",
                            (conversation_id,)).fetchone() is not None
    except psycopg.errors.InvalidTextRepresentation:     # not a uuid at all
        return False


def history(conn: psycopg.Connection, conversation_id: str, *, limit: int = MAX_TURNS) -> list[Turn]:
    """The last ``limit`` *answered* turns, oldest first.

    A turn the gate blocked is not history, even though it has a row and a visible reply. Its "answer"
    is a canned message from ``qualify.yaml``, which carries no fact to resolve a reference against -
    and, measured on the live index, actively does harm. Query 32:

        turn 8  "how to design a boom"            -> reject, and the refusal is stored as the answer
        turn 9  "how to create a boom really big" -> the rewriter was shown that refusal and copied it
                 out as the standalone question. The refusal names "recall", "FDA", "CPSC", "drug",
                 "product" and "consumer", so it sailed through the scope check, retrieved 8 unrelated
                 excerpts and cost 3,772 prompt tokens.

    The system had laundered its own refusal into a question that passed. Blocked turns are still
    logged - they are the audit trail - they are just not conversation.
    """
    rows = conn.execute(
        """
        SELECT turn, question, answer FROM rag.query_log
         WHERE conversation_id = %s::uuid AND answer IS NOT NULL AND error IS NULL
           AND coalesce(qualify_decision, 'pass') = 'pass'
         ORDER BY turn DESC
         LIMIT %s
        """,
        (conversation_id, limit),
    ).fetchall()
    return [Turn(r["turn"], r["question"], r["answer"]) for r in reversed(rows)]


def next_turn(conn: psycopg.Connection, conversation_id: str) -> int:
    row = conn.execute("SELECT coalesce(max(turn), 0) + 1 AS turn FROM rag.query_log "
                       "WHERE conversation_id = %s::uuid", (conversation_id,)).fetchone()
    return int(row["turn"])


def touch(conn: psycopg.Connection, conversation_id: str, first_question: str) -> None:
    conn.execute("UPDATE rag.conversation SET updated_at = now(), title = coalesce(title, %s) "
                 "WHERE conversation_id = %s::uuid", (first_question[:TITLE_CHARS], conversation_id))


# ---------------------------------------------------------------- prompting
def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def history_block(turns: Sequence[Turn], *, answer_chars: int = ANSWER_CHARS) -> str:
    """Questions and (shortened) answers - for the rewrite step, which must resolve "the second one"."""
    return "\n\n".join(f"User: {t.question}\nAssistant: {_shorten(t.answer, answer_chars)}" for t in turns)


def questions_block(turns: Sequence[Turn]) -> str:
    """Questions only - for the answering step (see answer.build_prompt)."""
    return "\n".join(f"{n}. {t.question}" for n, t in enumerate(turns, start=1))


_PREFIX = re.compile(r"^(standalone question|question|rewritten question)\s*:\s*", re.IGNORECASE)


def clean_rewrite(text: str, fallback: str) -> str:
    """Models sometimes add a label, quotes or a second line; keep the question only."""
    lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if not lines:
        return fallback
    candidate = _PREFIX.sub("", lines[0]).strip().strip('"\'“”‘’`').strip()
    if not candidate or len(candidate) > 1000:
        return fallback
    return candidate


ECHO_CHARS = 20         # below this, a rewrite matching an answer is a coincidence, not a copy


def _echoes_history(candidate: str, turns: Sequence[Turn]) -> bool:
    """True when the rewrite is a piece of something already said back to the user.

    The rewrite is model output, and the model is shown the previous answers so it can resolve "the
    second one". That is also how it can hand back one of those answers as the next question to search
    for - see the incident in :func:`history`. Excluding blocked turns removes the case that was
    measured; this removes the shape of it, for any text this interface has ever printed.
    """
    flat = " ".join((candidate or "").lower().split())
    if len(flat) < ECHO_CHARS:
        return False
    return any(flat in " ".join(t.answer.lower().split()) for t in turns)


@tracing.traceable(run_type="llm", name="condense_question",
                   inputs=lambda a: {"messages": [
                       {"role": "system", "content": CONDENSE_SYSTEM},
                       {"role": "user", "content": _condense_prompt(a.get("turns") or [], a.get("question") or "")}]},
                   outputs=tracing.llm_outputs)
def _rewrite(chat: ChatProvider, turns: Sequence[Turn], question: str):
    tracing.annotate_model(chat)
    # Rewriting is mechanical; hidden thinking only adds latency and, on Gemini 2.5, eats the output
    # budget (a rewrite was once cut off at 38 visible tokens after ~6 s of thinking).
    reply = chat.complete(CONDENSE_SYSTEM, _condense_prompt(turns, question), reasoning=False)
    tracing.record_usage(reply)
    return reply


def _condense_prompt(turns: Sequence[Turn], question: str) -> str:
    return (f"CONVERSATION SO FAR\n{history_block(turns)}\n\n"
            f"LATEST MESSAGE\n{question}\n\nSTANDALONE QUESTION:")


def condense(chat: ChatProvider, turns: Sequence[Turn], question: str) -> str:
    """Standalone form of ``question``; the original when there is no history.

    A failed rewrite must not cost the user the turn: retrieval then uses the previous question
    followed by the new one, which still carries the subject the follow-up leaves out.
    """
    if not turns:
        return question
    fallback = f"{turns[-1].question} {question}"
    try:
        reply = _rewrite(chat, turns, question)
    except ProviderError as exc:
        log.warning("question rewrite failed; retrieving with the previous question as context",
                    extra={"error": str(exc)[:300]})
        return fallback
    if reply.truncated:
        # half a question retrieves worse than the two raw questions together
        log.warning("question rewrite hit the output token limit; using the previous question as context")
        return fallback
    candidate = clean_rewrite(reply.text, question)
    if _echoes_history(candidate, turns):
        # What the user typed is at least what the user meant. The gate judges it as a follow-up, which
        # is what it is, and nothing the interface printed earlier gets searched for as a question.
        log.warning("question rewrite echoed an earlier answer; keeping the message as typed")
        return question
    return candidate
