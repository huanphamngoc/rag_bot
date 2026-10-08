"""Web chat over the RAG layer: one page and a small JSON API around ``answer.ask``.

The same code path as ``crawlerrag ask`` / ``chat`` - hybrid retrieval, follow-up rewriting, citations,
``rag.query_log`` - only the input and output are HTTP. Runs under gunicorn in the ``web`` compose service:

    gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 "crawlerrag.rag.web:create_app()"

Guard rails, because every question is a paid model call and the page may be reachable by others:

* a daily question cap (``RAG_WEB_DAILY_LIMIT``) counted in ``rag.query_log``, so every instance
  shares it; 0 turns the cap off;
* input limits: question length, doc types from the indexed set only, a few metadata filters;
* nothing from the model or the data is ever parsed as HTML by the page (static/app.js builds text
  nodes), and a strict Content-Security-Policy allows only this origin's own script and style.
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from typing import Any, Callable

import psycopg
from flask import Flask, Response, jsonify, request, send_from_directory

from crawlerrag import db
from crawlerrag.config import get_settings
from crawlerrag.logging_setup import setup_logging
from crawlerrag.rag import answer as answer_mod
from crawlerrag.rag import conversation, tracing
from crawlerrag.rag import graph as chat_graph
from crawlerrag.rag import index as rag_index
from crawlerrag.rag.providers import ProviderError, chat_provider, embedding_provider

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 2000
SSE_RETRY_MS = 3000
THREAD_ID = re.compile(r"[0-9a-f]{32}")
MAX_FILTERS = 5
MAX_FILTER_CHARS = 200
SNIPPET_CHARS = 600
INFO_TTL_S = 300          # indexed doc types change only when the index is rebuilt
# Provider error texts (HTTP bodies, credential hints) go to the log and rag.query_log, not to the browser.
MODEL_FAILED = "The language model call failed; the question was logged. Try again."

LABELS = {
    "drug_recall": "FDA drug recalls",
    "cpsc_recall": "CPSC consumer product recalls",
    "drug_product": "NDC drug directory",
    "food_product": "Branded foods (FoodData Central)",
    "provider": "Healthcare organisations (NPPES)",
    "insurance_plan": "Marketplace insurance plans",
}

SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
                                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class BadRequest(ValueError):
    """Input the page should never send; answered with HTTP 400 and the message."""


def _safe_url(url: str | None) -> str | None:
    return url if url and url.startswith(("https://", "http://")) else None


def _source(n: int, hit) -> dict[str, Any]:
    text = " ".join((hit.text or "").split())
    return {"n": n, "doc_id": hit.doc_id, "doc_type": hit.doc_type, "title": hit.title, "url": _safe_url(hit.url),
            "matched_by": hit.matched_by, "score": round(hit.score, 4),
            "snippet": text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rstrip() + " ..."}


def parse_ask(payload: Any, indexed: set[str]) -> dict[str, Any]:
    """Validate the JSON body of POST /api/ask; raises BadRequest."""
    if not isinstance(payload, dict):
        raise BadRequest("expected a JSON object")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise BadRequest("question is empty")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise BadRequest(f"question is longer than {MAX_QUESTION_CHARS} characters")

    doc_types = payload.get("doc_types") or None
    if doc_types is not None:
        if not isinstance(doc_types, list) or not all(isinstance(d, str) for d in doc_types):
            raise BadRequest("doc_types must be a list of names")
        unknown = sorted(set(doc_types) - indexed)
        if unknown:
            raise BadRequest(f"not indexed: {', '.join(unknown)}")
        doc_types = list(dict.fromkeys(doc_types))

    filters = payload.get("filters") or None
    if filters is not None:
        if not isinstance(filters, dict) or len(filters) > MAX_FILTERS:
            raise BadRequest(f"filters must be an object with at most {MAX_FILTERS} keys")
        for key, value in filters.items():
            if not key or len(key) > 64 or not isinstance(value, (str, int, float, bool)) \
                    or len(str(value)) > MAX_FILTER_CHARS:
                raise BadRequest(f"bad filter {key!r}")

    conversation_id = payload.get("conversation_id") or None
    if conversation_id is not None and not isinstance(conversation_id, str):
        raise BadRequest("conversation_id must be a string")

    # A turn paused at the clarify step is answered by sending the corrected question with the thread it
    # is waiting on. Same endpoint, same validation: the correction is a question like any other.
    thread_id = payload.get("thread_id") or None
    if thread_id is not None and (not isinstance(thread_id, str) or not THREAD_ID.fullmatch(thread_id)):
        raise BadRequest("thread_id must be a hex string")
    return {"question": question, "doc_types": doc_types, "filters": filters,
            "conversation_id": conversation_id, "thread_id": thread_id}


def used_today(conn: psycopg.Connection) -> int:
    row = conn.execute("SELECT count(*) AS n FROM rag.query_log "
                       "WHERE asked_at >= date_trunc('day', now(), 'UTC')").fetchone()
    return int(row["n"])


def indexed_doc_types(conn: psycopg.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT doc_type, count(*) AS n FROM rag.document WHERE is_current AND is_active "
                        "GROUP BY doc_type ORDER BY doc_type").fetchall()
    return {r["doc_type"]: int(r["n"]) for r in rows}


class OverTheCap(Exception):
    """The daily question cap is used up. Raised where /api/ask would have returned 429."""


def sse(event: str, data: dict) -> str:
    """One Server-Sent Event. The blank line is the terminator, so it is not optional."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def turn_sources(conn: psycopg.Connection, retrieved: list[dict]) -> list[dict]:
    """Rebuild a past turn's source list from the doc ids stored in rag.query_log.retrieved."""
    doc_ids = [r["doc_id"] for r in retrieved if r.get("doc_id")]
    if not doc_ids:
        return []
    docs = {r["doc_id"]: r for r in conn.execute(
        "SELECT doc_id, doc_type, title, url FROM rag.document WHERE doc_id = ANY(%s) AND is_current",
        (doc_ids,)).fetchall()}
    out = []
    for r in retrieved:
        d = docs.get(r.get("doc_id"))
        out.append({"n": r.get("rank"), "doc_id": r.get("doc_id"), "doc_type": d["doc_type"] if d else None,
                    "title": d["title"] if d else r.get("doc_id"), "url": _safe_url(d["url"]) if d else None,
                    "matched_by": r.get("matched_by"), "score": r.get("score"), "snippet": None})
    return out


def create_app(settings=None, *, connect: Callable[[], psycopg.Connection] | None = None,
               embedder_factory: Callable = embedding_provider, chat_factory: Callable = chat_provider) -> Flask:
    settings = settings or get_settings()
    if connect is None:
        setup_logging(settings.log_level, settings.log_format)

        def connect() -> psycopg.Connection:
            return db.connect(settings.database_url, application_name="crawler-rag-web")

    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.json.sort_keys = False
    daily_limit = settings.rag_web_daily_limit
    lock = threading.Lock()
    models: dict[str, Any] = {}
    info_cache: dict[str, Any] = {}

    def providers():
        """One embedder and one chat client per process: the Vertex credentials are loaded once and the HTTP
        connection pools are reused by the gunicorn threads (httpcore's sync pool guards its state with a
        threading lock; two threads refreshing an expired token at once only fetch it twice)."""
        with lock:
            if not models:
                models["embedder"] = embedder_factory(settings)
                models["chat"] = chat_factory(settings)
            return models["embedder"], models["chat"]

    def indexed(conn) -> dict[str, int]:
        if info_cache.get("until", 0) < time.monotonic():
            info_cache["doc_types"] = indexed_doc_types(conn)
            info_cache["until"] = time.monotonic() + INFO_TTL_S
        return info_cache["doc_types"]

    @app.after_request
    def _headers(response: Response) -> Response:
        response.headers.update(SECURITY_HEADERS)
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(BadRequest)
    def _bad_request(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(OverTheCap)
    def _over_the_cap(exc):
        return jsonify(error=str(exc)), 429

    @app.errorhandler(rag_index.IndexStateError)
    def _no_index(exc):
        log.error("index not usable", extra={"error": str(exc)})
        return jsonify(error=f"The search index is not ready: {exc}"), 503

    @app.errorhandler(psycopg.OperationalError)
    def _db_down(exc):
        log.error("database unavailable", extra={"error": str(exc).strip()[:300]})
        return jsonify(error="The database is not reachable right now. Try again in a minute."), 503

    @app.get("/")
    def index():
        response = send_from_directory(app.static_folder, "index.html")
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/healthz")
    def healthz():
        return jsonify(status="ok")

    @app.get("/api/info")
    def info():
        with connect() as conn:
            state = rag_index.read_state(conn)
            counts = indexed(conn)
            used = used_today(conn)
        default = [d.strip() for d in settings.rag_doc_types.split(",") if d.strip() in counts]
        return jsonify(
            ready=state is not None and bool(counts),
            doc_types=[{"name": name, "label": LABELS.get(name, name), "documents": n,
                        "default": name in default} for name, n in counts.items()],
            embed_model=f"{settings.rag_embed_provider}/{settings.rag_embed_model}",
            chat_model=f"{settings.rag_chat_provider}/{settings.rag_chat_model}",
            daily_limit=daily_limit, used_today=used,
            max_question_chars=MAX_QUESTION_CHARS,
        )

    @app.post("/api/conversations")
    def new_conversation():
        with connect() as conn:
            return jsonify(conversation_id=conversation.create(conn)), 201

    @app.get("/api/conversations/<conversation_id>")
    def get_conversation(conversation_id: str):
        with connect() as conn:
            if not conversation.exists(conn, conversation_id):
                return jsonify(error="no such conversation"), 404
            rows = conn.execute(
                """
                SELECT turn, question, answer, error, retrieved, duration_ms, prompt_tokens, output_tokens,
                       query_id, standalone_question
                  FROM rag.query_log
                 WHERE conversation_id = %s::uuid
                 ORDER BY turn
                """, (conversation_id,)).fetchall()
            turns = [{"turn": r["turn"], "question": r["question"], "answer": r["answer"] or "",
                      "error": MODEL_FAILED if r["error"] else None,
                      "query_id": r["query_id"], "duration_ms": r["duration_ms"],
                      "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"],
                      "standalone_question": r["standalone_question"],
                      "sources": turn_sources(conn, r["retrieved"] or [])} for r in rows]
        return jsonify(conversation_id=conversation_id, turns=turns)

    def payload(reply) -> dict:
        return {
            "question": reply.question, "answer": reply.text,
            "error": MODEL_FAILED if reply.error else None,
            "truncated": reply.truncated, "query_id": reply.query_id,
            "conversation_id": reply.conversation_id, "turn": reply.turn,
            "standalone_question": reply.standalone_question, "duration_ms": reply.duration_ms,
            "prompt_tokens": reply.prompt_tokens, "output_tokens": reply.output_tokens,
            "sources": [_source(n, h) for n, h in enumerate(reply.hits, start=1)],
            "trace_url": tracing.trace_url(reply.trace_id) if reply.trace_id else None,
            # Set when the turn is waiting for a better question: what to ask, questions this index can
            # really answer, and the thread to send the correction back on.
            "pending_question": reply.pending_question,
            "samples": [{"question": s["question"], "about": s.get("title")} for s in reply.samples],
            "thread_id": reply.thread_id,
        }

    def check(conn, body) -> dict:
        """Everything that must be refused before a paid call: input, conversation, daily cap."""
        req = parse_ask(body, set(indexed(conn)))
        if req["conversation_id"] and not conversation.exists(conn, req["conversation_id"]):
            raise BadRequest("no such conversation")
        if daily_limit and used_today(conn) >= daily_limit:
            raise OverTheCap(f"The daily limit of {daily_limit} questions is used up. "
                             "It resets at 00:00 UTC.")
        return req

    def one_turn(conn, req, *, sink=None):
        """Ask, or hand a correction to a turn that is waiting for one."""
        embedder, chat = providers()
        if req.get("thread_id"):
            return answer_mod.resume(conn, settings, embedder, chat, thread_id=req["thread_id"],
                                     question=req["question"], sink=sink)
        return answer_mod.ask(conn, settings, embedder, chat, req["question"],
                              doc_types=req["doc_types"], filters=req["filters"],
                              conversation_id=req["conversation_id"], sink=sink)

    @app.post("/api/ask")
    def ask():
        with connect() as conn:
            req = check(conn, request.get_json(silent=True))
            reply = one_turn(conn, req)
        if reply.error:
            log.error("answer failed", extra={"query_id": reply.query_id, "error": reply.error[:500]})
        return jsonify(**payload(reply))

    @app.post("/api/ask/stream")
    def ask_stream():
        """The same turn as /api/ask, sent as Server-Sent Events: ``delta`` while the model writes,
        then one ``done`` with the sources, the token counts and the trace link.

        The graph is synchronous and pushes text into a sink, while the response has to pull, so the
        turn runs in a worker thread with a queue between them. The worker opens its own connection -
        a psycopg connection belongs to one thread - and the SSE body is produced after the request
        context is gone, which is why everything the generator needs is read out of the request first.
        """
        body = request.get_json(silent=True)
        with connect() as conn:
            check(conn, body)                       # refuse bad input before starting a worker

        pieces: queue.Queue = queue.Queue()
        outcome: dict = {}

        def work():
            try:
                with connect() as conn:
                    req = check(conn, body)
                    outcome["reply"] = one_turn(conn, req, sink=pieces.put)
            except Exception as exc:                # noqa: BLE001 - reported as an SSE error event
                outcome["error"] = exc
            finally:
                pieces.put(None)

        worker = threading.Thread(target=work, name="ask-stream", daemon=True)
        worker.start()

        def events():
            while True:
                piece = pieces.get()
                if piece is None:
                    break
                yield sse("delta", {"text": piece})
            worker.join()
            exc = outcome.get("error")
            if exc is not None:
                if isinstance(exc, BadRequest):
                    yield sse("error", {"error": str(exc)})
                elif isinstance(exc, OverTheCap):
                    yield sse("error", {"error": str(exc)})
                else:
                    # Provider bodies and credential hints stay on the server, as in /api/ask.
                    log.error("streamed answer failed", extra={"error": str(exc)[:500]})
                    yield sse("error", {"error": MODEL_FAILED})
                return
            reply = outcome["reply"]
            if reply.error:
                log.error("answer failed", extra={"query_id": reply.query_id,
                                                  "error": reply.error[:500]})
            # A paused turn gets its own event: there is no answer to render, and the page has to show
            # the examples and keep the thread so the correction can be sent back on it.
            yield sse("question" if reply.pending_question else "done", payload(reply))

        return Response(events(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.errorhandler(chat_graph.NoSuchTurn)
    def _no_such_turn(exc):
        # The pause expired, or the page was reloaded with a stale thread: ask again from the start.
        log.info("resume for an unknown thread", extra={"detail": str(exc)[:200]})
        return jsonify(error="That question is no longer waiting for an answer. Please ask it again."), 409

    @app.errorhandler(ProviderError)
    def _provider(exc):
        # query embedding failed (the chat step records its own failure in the answer instead)
        log.error("model provider failed", extra={"error": str(exc)[:500]})
        return jsonify(error="The embedding model call failed. Try again in a minute."), 502

    return app
