"""Grounded answering: retrieve, prompt, cite, log.

Two guard rails matter more here than prompt polish:

* the model answers only from the retrieved excerpts and must say so when they are not
  enough - this data is regulatory (recalls, drug labels, insurance plans), so a fluent
  guess is worse than "not in the retrieved records";
* retrieved excerpts are a *sample*, never a complete aggregate. "How many recalls in
  2024" is a SQL question; the model is told to say that instead of counting its context.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import psycopg

from crawlerrag.rag import conversation, retrieve, tracing
from crawlerrag.rag.providers import (ChatProvider, ChatReply, EmbeddingProvider, ProviderError,
                                      complete_streaming)

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about United States public regulatory data that was \
crawled into a database: FDA drug recalls and the NDC drug directory, CPSC consumer product \
recalls, NPPES healthcare organisations, ACA Marketplace insurance plans, and USDA FoodData \
Central branded foods.

Rules you must follow:
1. Use only the numbered excerpts provided in the CONTEXT. Never add facts from your own \
knowledge, and never guess a value that is not written there.
2. Cite the excerpts you used inline as [1], [2], ... after the sentence they support.
3. If the excerpts do not contain the answer, say plainly that the retrieved records do not \
contain it, and name what would be needed. Do not speculate.
4. The excerpts are a small retrieved sample, not the whole database. If the question asks for \
a total, a count, an average or a ranking across the whole dataset, say that this needs a SQL \
query over the tables rather than retrieval, then answer only what the excerpts do support. \
The same holds for "any", "none", "all", "every" and "only": never conclude that no record (or \
every record) in the database matches. Say what the N retrieved excerpts show - for example "the \
8 retrieved recalls are all Class II" - and that records outside the excerpts may differ.
5. Quote identifiers (recall numbers, NDC codes, NPI numbers, plan IDs, dates) exactly as written.
6. Answer in the same language the question was asked in. Be concise and factual.
7. This is public regulatory record-keeping, not advice. Do not add medical, legal or \
financial recommendations.
8. An EARLIER QUESTIONS block, when present, only tells you what the conversation is about. \
Never treat an earlier answer as a source: every fact must come from the CONTEXT excerpts of \
this turn, and the citation numbers refer to those excerpts only."""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[retrieve.Hit] = field(default_factory=list)
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    duration_ms: int = 0
    query_id: int | None = None
    error: str | None = None
    trace_id: str | None = None     # MLflow trace ("tr-..."), when tracing is on
    conversation_id: str | None = None
    turn: int | None = None
    standalone_question: str | None = None   # what retrieval searched for, when it differs
    truncated: bool = False                  # the model hit RAG_CHAT_MAX_TOKENS mid-answer
    qualify_decision: str | None = None      # pass / needs_sql / clarify / reject
    qualify_rule: str | None = None           # which rule in rules/qualify.yaml decided
    visited: list[str] = field(default_factory=list)   # the chat graph nodes this turn ran
    # Set when the turn is paused at the clarify step: what to ask, what to offer, and the thread to
    # resume with the corrected question. `text` is empty in that case - nothing has been answered yet.
    pending_question: str | None = None
    samples: list[dict] = field(default_factory=list)
    thread_id: str | None = None

    @property
    def waiting_for_a_better_question(self) -> bool:
        return self.pending_question is not None

    @property
    def decision(self) -> str | None:
        return self.qualify_decision

    @property
    def reached_the_model(self) -> bool:
        return "generate" in self.visited


def context_block(hits: Sequence[retrieve.Hit], *, max_chars: int = 2000) -> str:
    parts = []
    for n, hit in enumerate(hits, start=1):
        head = f"[{n}] type={hit.doc_type} id={hit.doc_id}"
        if hit.url:
            head += f" url={hit.url}"
        body = hit.text if len(hit.text) <= max_chars else hit.text[:max_chars] + " ..."
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)


def build_prompt(question: str, hits: Sequence[retrieve.Hit], *,
                 history: Sequence[conversation.Turn] = (), standalone: str | None = None) -> str:
    """The answering model sees earlier *questions* only. Shown its earlier answers, it repeated a
    wrong one word for word on the next turn; the rewrite step has already resolved what the
    follow-up refers to, so the answers add nothing here but a way for errors to carry over."""
    head = (f"EARLIER QUESTIONS IN THIS CONVERSATION\n{conversation.questions_block(history)}\n\n"
            if history else "")
    asked = question if not standalone or standalone == question else \
        f"{question}\n(interpreted as: {standalone})"
    if not hits:
        return (f"{head}CONTEXT\n(no records were retrieved)\n\nQUESTION\n{asked}\n\n"
                "Tell the user that no matching records were retrieved from the database.")
    return f"{head}CONTEXT\n{context_block(hits)}\n\nQUESTION\n{asked}\n\nANSWER (cite as [1], [2], ...):"


def log_query(conn: psycopg.Connection, *, question: str, doc_types: Sequence[str] | None,
              top_k: int, hits: Sequence[retrieve.Hit], answer: str | None,
              embed_provider: str, embed_model: str, chat_provider: str | None, chat_model: str | None,
              prompt_tokens: int | None, output_tokens: int | None, duration_ms: int,
              error: str | None, trace_id: str | None = None, conversation_id: str | None = None,
              turn: int | None = None, standalone_question: str | None = None,
              qualify_decision: str | None = None, qualify_rule: str | None = None) -> int | None:
    retrieved = [{"rank": n, "chunk_id": h.chunk_id, "doc_id": h.doc_id,
                  "score": round(h.score, 6), "matched_by": h.matched_by}
                 for n, h in enumerate(hits, start=1)]
    try:
        row = conn.execute(
            """
            INSERT INTO rag.query_log (question, doc_types, top_k, retrieved, answer, embed_provider,
                                       embed_model, chat_provider, chat_model, prompt_tokens, output_tokens,
                                       duration_ms, error, conversation_id, turn, standalone_question,
                                       trace_id, qualify_decision, qualify_rule)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING query_id
            """,
            (question, list(doc_types) if doc_types else None, top_k,
             json.dumps(retrieved), answer, embed_provider, embed_model, chat_provider, chat_model,
             prompt_tokens, output_tokens, duration_ms, error, conversation_id, turn, standalone_question,
             trace_id, qualify_decision, qualify_rule),
        ).fetchone()
        return int(row["query_id"])
    except psycopg.Error:
        log.warning("could not write rag.query_log", exc_info=True)
        return None


_llm_outputs = tracing.llm_outputs


@tracing.traceable(run_type="llm", name="generate_answer",
                   inputs=lambda a: {"messages": [{"role": "system", "content": a.get("system")},
                                                  {"role": "user", "content": a.get("prompt")}]},
                   outputs=_llm_outputs)
def _generate(chat: ChatProvider, system: str, prompt: str, sink=None) -> ChatReply:
    """``sink`` is called with each piece of text as it arrives (the web page streams them to the
    browser). Without one this is the plain single-shot call it has always been, and the span is
    identical either way - the usage still comes from the finished reply."""
    tracing.annotate_model(chat)
    reply = chat.complete(system, prompt) if sink is None else \
        complete_streaming(chat, system, prompt, sink)
    tracing.record_usage(reply)
    return reply


def _ask_inputs(args: dict) -> dict:
    return {"question": args.get("question"), "doc_types": args.get("doc_types"), "top_k": args.get("top_k"),
            "candidates": args.get("candidates"), "filters": args.get("filters"),
            "conversation_id": args.get("conversation_id")}


def _ask_outputs(result: "Answer") -> dict:
    return {"answer": result.text, "error": result.error, "query_id": result.query_id, "turn": result.turn,
            "standalone_question": result.standalone_question,
            "pending_question": result.pending_question,
            "samples": [s.get("question") for s in result.samples],
            "sources": [{"rank": n, "doc_id": h.doc_id, "title": h.title, "url": h.url}
                        for n, h in enumerate(result.hits, start=1)]}


@tracing.traceable(run_type="chain", name="rag_ask", inputs=_ask_inputs, outputs=_ask_outputs)
def ask(conn: psycopg.Connection, settings, embedder: EmbeddingProvider, chat: ChatProvider,
        question: str, *, doc_types: Sequence[str] | None = None, top_k: int | None = None,
        candidates: int | None = None, filters: dict[str, Any] | None = None,
        conversation_id: str | None = None, ruleset=None, sink=None,
        thread_id: str | None = None) -> Answer:
    """One turn, run through the chat graph (``crawlerrag.rag.graph``).

    Everything still enters here - ``crawlerrag ask``, ``chat`` and the web page - so this is where the
    MLflow chain span and the trace tags live. The order of the work, and the branches that keep a
    question away from the model, are the graph's.
    """
    # Imported here: the graph's nodes import this module, so the dependency only runs one way at import.
    from crawlerrag.rag import graph as chat_graph
    from crawlerrag.rules import load_rules_cached

    # The session id groups the turns of one conversation in MLflow's chat-sessions view.
    tracing.update_trace(tags={"embed_provider": embedder.provider, "embed_model": embedder.model,
                               "chat_provider": chat.provider, "chat_model": chat.model},
                         session_id=conversation_id)
    deps = chat_graph.ChatDeps(conn=conn, settings=settings, embedder=embedder, chat=chat,
                               ruleset=ruleset or load_rules_cached(settings.rules_dir), sink=sink)
    reply = chat_graph.run_chat(deps, question, doc_types=doc_types, top_k=top_k, candidates=candidates,
                                filters=filters, conversation_id=conversation_id, thread_id=thread_id)
    # The trace list shows these previews; by default it would show the inputs as JSON.
    tracing.update_trace(tags={"query_id": reply.query_id, "turn": reply.turn,
                               "qualify_decision": reply.qualify_decision,
                               # Searchable in the trace list: "show me every question the rules refused".
                               "qualify_rule": reply.qualify_rule},
                         request_preview=reply.question,
                         response_preview=reply.text or reply.pending_question or reply.error or "")
    return reply


def _resume_outputs(result: "Answer") -> dict:
    return {**_ask_outputs(result), "question": result.question}


@tracing.traceable(run_type="chain", name="rag_clarified",
                   inputs=lambda a: {"question": a.get("question"), "thread_id": a.get("thread_id")},
                   outputs=_resume_outputs)
def resume(conn: psycopg.Connection, settings, embedder: EmbeddingProvider, chat: ChatProvider,
           *, thread_id: str, question: str, ruleset=None, sink=None) -> Answer:
    """Hand a corrected question to a turn paused at the clarify step.

    A separate trace from the paused one, because the paused turn produced no answer and the two can be
    minutes apart; the tags carry the thread so the pair can be found together.
    """
    from crawlerrag.rag import graph as chat_graph
    from crawlerrag.rules import load_rules_cached

    deps = chat_graph.ChatDeps(conn=conn, settings=settings, embedder=embedder, chat=chat,
                               ruleset=ruleset or load_rules_cached(settings.rules_dir), sink=sink)
    reply = chat_graph.resume_chat(deps, thread_id=thread_id, question=question)
    tracing.update_trace(tags={"query_id": reply.query_id, "turn": reply.turn,
                               "qualify_decision": reply.qualify_decision,
                               "qualify_rule": reply.qualify_rule, "clarify_thread": thread_id},
                         session_id=reply.conversation_id, request_preview=reply.question,
                         response_preview=reply.text or reply.pending_question or reply.error or "")
    return reply
