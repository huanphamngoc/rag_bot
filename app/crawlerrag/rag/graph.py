"""The chat flow as a LangGraph graph.

    qualify ─┬─(pass)──► condense ─┬─(pass)──► retrieve ─┬─(hits)───► generate ─┐
             │                     │                     └─(none)───► no_records┼─► log ─► END
             ├─(clarify, once)──► clarify ──(the corrected question)──► qualify  │
             └─(reject/needs_sql)──────────────────────────────► blocked ────────┘

The paths that must not reach a paid model are edges here, not conditions buried inside a function, and
every answer reports the nodes it visited - which is how "was the model called for this question?" is
answered by the trace instead of by reading the code.

The ``clarify`` node is the one place this graph pauses. A question that names a subject but asks
nothing ("insulin") is not refused: the turn stops with ``interrupt()``, shows questions built from
records really in the index (``rag.samples``), and resumes **at that node** when the corrected question
arrives. That is why this graph is checkpointed when ``RAG_CHAT_CLARIFY`` is on - there is nothing to
resume otherwise. It is bounded to one round: a second vague question goes to ``blocked`` instead of
asking again forever.

With the clarify step off, or with no checkpointer, a ``clarify`` decision falls through to ``blocked``
and is answered with its message - the behaviour this graph had before.
"""
from __future__ import annotations

import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from crawlerrag import checkpoints
from crawlerrag.rag import answer as answer_mod
from crawlerrag.rag import conversation, qualify as qualify_mod, retrieve, samples, tracing
from crawlerrag.rag.providers import ProviderError

log = logging.getLogger(__name__)

NO_RECORDS = ("No matching records were retrieved from the database for that question. Try naming the "
              "firm, the product or the recall number, or widen the question.")


@dataclass
class ChatDeps:
    """Everything one chat turn needs. One per request; the compiled graph is cached on it."""
    conn: Any
    settings: Any
    embedder: Any
    chat: Any
    ruleset: Any
    # Called with each piece of answer text as the model produces it. The web page's SSE endpoint puts
    # them on a queue; None means the old single-shot call. It is not part of the graph state: a
    # callable could not be checkpointed, and nothing about the flow depends on it.
    sink: Any = None
    compiled: Any = field(default=None, init=False, repr=False)


class ChatState(TypedDict, total=False):
    question: str
    asked: str
    conversation_id: str | None
    doc_types: list[str] | None
    top_k: int | None
    candidates: int | None
    filters: dict[str, Any] | None
    turns: list[Any]
    decision: str
    rule_id: str | None
    message: str | None
    sql_hint: str | None
    standalone: str | None
    hits: list[Any]
    text: str
    error: str | None
    reply: Any
    started: float
    # Written by the log node. A key missing from this schema is silently dropped by LangGraph, which is
    # exactly how query_id and turn went missing from the answer the first time.
    query_id: int | None
    turn: int | None
    # The clarify round. `clarified` bounds it to one: a corrected question that is still vague is
    # answered with the message rather than asked about again.
    samples: list[Any]
    clarified: bool
    can_clarify: bool


_DEPS: dict[str, ChatDeps] = {}


def _deps(config) -> ChatDeps:
    key = (config or {}).get("configurable", {}).get("deps_key")
    if key not in _DEPS:
        raise RuntimeError("the chat graph was invoked without its dependencies")
    return _DEPS[key]


# ---------------------------------------------------------------- nodes
@tracing.traceable(
    run_type="guardrail", name="qualify_question",
    inputs=lambda a: {"question": a.get("question"), "filters": a.get("filters"),
                      "doc_types": a.get("doc_types"), "is_follow_up": a.get("is_follow_up")},
    outputs=lambda r: {"decision": r.decision, "rule": r.rule_id, "message": r.message,
                       "question": r.question})
def _qualify(rules, question: str, *, filters=None, doc_types=None, is_follow_up: bool = False):
    """One span, so a trace shows why a question did or did not reach the model."""
    return qualify_mod.qualify(rules, question, filters=filters, doc_types=doc_types,
                               is_follow_up=is_follow_up)


def node_qualify(state: ChatState, config) -> dict:
    deps = _deps(config)
    turns = conversation.history(deps.conn, state["conversation_id"]) if state["conversation_id"] else []
    result = _qualify(deps.ruleset.qualify, state["asked"], filters=state.get("filters"),
                     doc_types=state.get("doc_types"), is_follow_up=bool(turns))
    return {"turns": turns, "question": result.question, "decision": result.decision,
            "rule_id": result.rule_id, "message": result.message, "sql_hint": result.sql_hint,
            "hits": []}


def node_blocked(state: ChatState, config) -> dict:
    result = qualify_mod.QualifyResult(decision=state["decision"], question=state["question"],
                                      rule_id=state.get("rule_id"), message=state.get("message"),
                                      sql_hint=state.get("sql_hint"))
    return {"text": qualify_mod.blocked_answer(result), "hits": []}


@tracing.traceable(
    run_type="guardrail", name="clarify_question",
    inputs=lambda a: {"question": a.get("question")},
    outputs=lambda r: {"samples": [s["question"] for s in r]})
def _samples(conn, question: str, *, doc_types=None, limit: int = 4) -> list[dict]:
    """Lexical only, so asking again costs nothing - one span, so a trace shows what was offered."""
    return samples.build(conn, question, doc_types=doc_types, limit=limit)


def node_clarify(state: ChatState, config) -> dict:
    """Stop and ask, with examples that are answerable because they come from the index itself.

    ``interrupt()`` returns what the caller resumed with: the corrected question. The node then re-runs
    the gate on it (the edge goes back to ``qualify``), so a correction is checked like any other
    question - it could be out of scope, or ask for something this interface will not do.
    """
    deps = _deps(config)
    found = _samples(deps.conn, state["question"], doc_types=state.get("doc_types"))
    answer = interrupt({"question": state.get("message"), "asked": state["question"],
                        "rule": state.get("rule_id"), "samples": found})
    corrected = (answer.get("question") if isinstance(answer, dict) else answer) or ""
    corrected = str(corrected).strip()
    if not corrected:
        # Resumed with nothing: there is no new question to judge, so keep the one we have and let the
        # router send it to `blocked` with the message it already carries.
        return {"samples": found, "clarified": True}
    log.info("question corrected after clarify", extra={"rule": state.get("rule_id")})
    return {"samples": found, "clarified": True, "asked": corrected, "question": corrected,
            "decision": qualify_mod.PASS, "rule_id": None, "message": None, "sql_hint": None}


def node_condense(state: ChatState, config) -> dict:
    deps = _deps(config)
    standalone = conversation.condense(deps.chat, state["turns"], state["question"])
    if not state["turns"]:
        return {"standalone": None}      # a first turn was already judged as a standalone question
    # The gate judged what the user typed, but retrieval and the answer are about the rewrite, and the
    # rewrite can turn a harmless follow-up into the very thing the gate exists to stop. Measured on the
    # live index: "ok, show me the Class I ones" after a refused count became "How many Class I drug
    # recalls were there in 2026?", and the answer read "there was one Class I drug recall in 2026"
    # where the real count is 28. So the question retrieval will actually use is checked as well.
    # Checked as a standalone question, not as a follow-up, because that is exactly what the rewrite
    # made it. `skip_for_follow_up` exists for "and the second one?" - a fragment that names nothing in
    # scope because the previous turn carries the subject. Once the subject has been written back in,
    # the exemption has nothing left to excuse, and leaving it on was a hole: measured on the live
    # index, "how to design a boom" typed as the second question of a conversation passed the gate
    # (off_topic is skipped for follow-ups, and the unsafe patterns spell it "bomb") and was answered
    # with a real fireworks recall. As the first question of a conversation the same text is rejected.
    # Checked whether or not the rewrite changed anything. An unchanged rewrite is the rewriter saying
    # the message was already standalone (rule 5 of CONDENSE_SYSTEM), which is the strongest reason to
    # judge it as one - and it used to be the one path that skipped the check entirely. Measured: "how
    # to craft a boom really big", asked as a second question, came back unchanged, so the follow-up
    # exemption from the first gate still stood and it retrieved 8 excerpts for 3,871 prompt tokens.
    checked = _qualify(deps.ruleset.qualify, standalone, filters=state.get("filters"),
                       doc_types=state.get("doc_types"), is_follow_up=False)
    # Only a rewrite that changed something is worth showing as "Searched as:".
    result: dict = {"standalone": None if standalone == state["question"] else standalone}
    if checked.blocked:
        result.update(decision=checked.decision, rule_id=checked.rule_id,
                      message=checked.message, sql_hint=checked.sql_hint)
    return result


def node_retrieve(state: ChatState, config) -> dict:
    deps = _deps(config)
    result = retrieve.search(deps.conn, deps.settings, deps.embedder,
                            state.get("standalone") or state["question"],
                            doc_types=state.get("doc_types"), top_k=state.get("top_k"),
                            candidates=state.get("candidates"), filters=state.get("filters"))
    return {"hits": result.hits}


def node_no_records(state: ChatState, config) -> dict:
    """Nothing retrieved: say so without paying for a model call that can only say the same."""
    return {"text": NO_RECORDS}


def node_generate(state: ChatState, config) -> dict:
    deps = _deps(config)
    prompt = answer_mod.build_prompt(state["question"], state["hits"], history=state["turns"],
                                    standalone=state.get("standalone"))
    try:
        reply = answer_mod._generate(deps.chat, answer_mod.SYSTEM_PROMPT, prompt, deps.sink)
    except ProviderError as exc:
        log.error("chat provider failed", extra={"error": str(exc)[:500]})
        return {"text": "", "error": str(exc), "reply": None}
    return {"text": reply.text, "reply": reply, "error": None}


def node_log(state: ChatState, config) -> dict:
    deps = _deps(config)
    reply = state.get("reply")
    conversation_id = state["conversation_id"]
    turn = conversation.next_turn(deps.conn, conversation_id) if conversation_id else None
    query_id = answer_mod.log_query(
        deps.conn, question=state["question"], doc_types=state.get("doc_types"),
        top_k=state.get("top_k") or deps.settings.rag_top_k, hits=state.get("hits") or [],
        answer=state.get("text") or None, embed_provider=deps.embedder.provider,
        embed_model=deps.embedder.model, chat_provider=deps.chat.provider, chat_model=deps.chat.model,
        prompt_tokens=reply.prompt_tokens if reply else None,
        output_tokens=reply.output_tokens if reply else None,
        duration_ms=int((time.monotonic() - state["started"]) * 1000), error=state.get("error"),
        trace_id=tracing.current_trace_id(), conversation_id=conversation_id, turn=turn,
        standalone_question=state.get("standalone"), qualify_decision=state["decision"],
        qualify_rule=state.get("rule_id"))
    if conversation_id:
        conversation.touch(deps.conn, conversation_id, state["question"])
    return {"query_id": query_id, "turn": turn}


# ---------------------------------------------------------------- edges
def route_after_qualify(state: ChatState) -> str:
    if state["decision"] == qualify_mod.PASS:
        return "condense"
    # Only a `clarify` is worth asking about, and only once. A reject (out of scope, unsafe, injection)
    # is answered and closed: inviting someone to reword a request for help building a weapon is
    # inviting them to try again.
    if state["decision"] == qualify_mod.CLARIFY and not state.get("clarified") \
            and state.get("can_clarify"):
        return "clarify"
    return "blocked"


def route_after_condense(state: ChatState) -> str:
    """The rewrite is checked too, so a follow-up cannot smuggle a question past the gate."""
    return "retrieve" if state["decision"] == qualify_mod.PASS else "blocked"


def route_after_retrieve(state: ChatState) -> str:
    return "generate" if state.get("hits") else "no_records"


def build_chat_graph(deps: ChatDeps, *, checkpointer=None):
    """Compiled once per ChatDeps while no checkpointer is involved.

    A checkpointed graph is not cached: the saver owns a connection that is closed when the turn ends,
    so a compiled graph holding it would be unusable on the next request.
    """
    if checkpointer is None and deps.compiled is not None:
        return deps.compiled
    graph = StateGraph(ChatState)
    graph.add_node("qualify", node_qualify)
    graph.add_node("blocked", node_blocked)
    graph.add_node("clarify", node_clarify)
    graph.add_node("condense", node_condense)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("no_records", node_no_records)
    graph.add_node("log", node_log)

    graph.set_entry_point("qualify")
    graph.add_conditional_edges("qualify", route_after_qualify,
                               {"condense": "condense", "blocked": "blocked", "clarify": "clarify"})
    graph.add_edge("clarify", "qualify")
    graph.add_edge("blocked", "log")
    graph.add_conditional_edges("condense", route_after_condense,
                               {"retrieve": "retrieve", "blocked": "blocked"})
    graph.add_conditional_edges("retrieve", route_after_retrieve,
                               {"generate": "generate", "no_records": "no_records"})
    graph.add_edge("generate", "log")
    graph.add_edge("no_records", "log")
    graph.add_edge("log", END)
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    deps.compiled = graph.compile()
    return deps.compiled


# ---------------------------------------------------------------- invoking it
INTERRUPT = "__interrupt__"


class NoSuchTurn(RuntimeError):
    """resume_chat was given a thread id this database has no paused turn for."""


def _clarify_enabled(deps: ChatDeps) -> bool:
    return bool(getattr(deps.settings, "rag_chat_clarify", False)
                and getattr(deps.settings, "database_url", None))


@contextlib.contextmanager
def _saver(deps: ChatDeps, given=None):
    """A checkpointer only while the clarify step is on - otherwise a turn costs no extra connection.

    ``given`` is one handed in by the caller (the tests pass ``MemorySaver``), used as it is.
    """
    if given is not None:
        yield given
        return
    if not _clarify_enabled(deps):
        yield None
        return
    # Only a failure to OPEN the saver falls back to no checkpointer. Wrapping the body as well made
    # this generator yield twice when a turn raised ("generator didn't stop after throw()").
    with contextlib.ExitStack() as stack:
        try:
            saver = stack.enter_context(checkpoints.postgres_saver(deps.settings.database_url))
        except Exception:   # noqa: BLE001 - a chat turn must not fail because a pause cannot be stored
            log.warning("clarify step unavailable: the checkpointer could not be opened", exc_info=True)
            saver = None
        yield saver


def _drive(deps: ChatDeps, payload, config_extra: dict, state: dict, *, saver, thread_id: str):
    """Stream the graph, folding node deltas into `state` and catching a pause.

    Measured on langgraph 0.6.11: a pause arrives as the update key ``__interrupt__`` whose value is a
    tuple of ``Interrupt`` objects, not a state delta - ``dict.update`` on it raises TypeError.
    """
    visited: list[str] = []
    waiting: dict | None = None
    graph = build_chat_graph(deps, checkpointer=saver)
    config = {"configurable": {"deps_key": config_extra["deps_key"], "thread_id": thread_id}}
    for update in graph.stream(payload, config, stream_mode="updates"):
        for node, delta in update.items():
            if node == INTERRUPT:
                items = delta if isinstance(delta, (list, tuple)) else (delta,)
                value = getattr(items[0], "value", None) if items else None
                waiting = value if isinstance(value, dict) else {"question": value}
                continue
            visited.append(node)
            state.update(delta or {})
    return state, visited, waiting


def _answer(state: dict, *, visited: list[str], waiting: dict | None, thread_id: str | None,
            conversation_id: str | None) -> "answer_mod.Answer":
    reply = state.get("reply")
    return answer_mod.Answer(
        question=state["question"], text=state.get("text") or "", hits=state.get("hits") or [],
        prompt_tokens=reply.prompt_tokens if reply else None,
        output_tokens=reply.output_tokens if reply else None,
        duration_ms=int((time.monotonic() - state["started"]) * 1000), query_id=state.get("query_id"),
        error=state.get("error"), trace_id=tracing.current_trace_id(), conversation_id=conversation_id,
        turn=state.get("turn"), standalone_question=state.get("standalone"),
        truncated=bool(reply and reply.truncated), qualify_decision=state["decision"],
        qualify_rule=state.get("rule_id"), visited=visited,
        pending_question=(waiting or {}).get("question") if waiting else None,
        samples=list((waiting or {}).get("samples") or []) if waiting else [],
        thread_id=thread_id if waiting else None)


def run_chat(deps: ChatDeps, question: str, *, doc_types: Sequence[str] | None = None,
             top_k: int | None = None, candidates: int | None = None,
             filters: dict[str, Any] | None = None, conversation_id: str | None = None,
             thread_id: str | None = None, checkpointer=None) -> "answer_mod.Answer":
    initial: dict = {"asked": question or "", "question": question or "",
                     "conversation_id": conversation_id,
                     "doc_types": list(doc_types) if doc_types else None,
                     # Resolved here rather than inside retrieval, so what was asked for is what gets
                     # logged and traced.
                     "top_k": top_k or deps.settings.rag_top_k,
                     "candidates": candidates, "filters": filters, "turns": [], "hits": [], "text": "",
                     "error": None, "reply": None, "standalone": None, "decision": qualify_mod.PASS,
                     "rule_id": None, "message": None, "sql_hint": None, "samples": [],
                     "clarified": False, "started": time.monotonic()}
    key = uuid.uuid4().hex
    _DEPS[key] = deps
    thread = thread_id or uuid.uuid4().hex
    try:
        with _saver(deps, checkpointer) as saver:
            # The router needs to know whether a pause could be stored at all: without a checkpointer a
            # clarify is answered with its message instead of being asked about.
            initial["can_clarify"] = saver is not None
            state, visited, waiting = _drive(deps, initial, {"deps_key": key}, dict(initial),
                                             saver=saver, thread_id=thread)
    finally:
        _DEPS.pop(key, None)
    return _answer(state, visited=visited, waiting=waiting, thread_id=thread,
                   conversation_id=conversation_id)


def resume_chat(deps: ChatDeps, *, thread_id: str, question: str,
                checkpointer=None) -> "answer_mod.Answer":
    """Hand a corrected question to a turn paused at the clarify step and let it finish."""
    key = uuid.uuid4().hex
    _DEPS[key] = deps
    try:
        with _saver(deps, checkpointer) as saver:
            if saver is None:
                raise NoSuchTurn("resuming needs the checkpointer (RAG_CHAT_CLARIFY=true)")
            graph = build_chat_graph(deps, checkpointer=saver)
            config = {"configurable": {"deps_key": key, "thread_id": thread_id}}
            before = graph.get_state(config)
            if not before or not before.values:
                raise NoSuchTurn(f"no paused turn for thread {thread_id!r}")
            state = dict(before.values)
            state.setdefault("started", time.monotonic())
            state, visited, waiting = _drive(deps, Command(resume={"question": question}),
                                             {"deps_key": key}, state, saver=saver,
                                             thread_id=thread_id)
    finally:
        _DEPS.pop(key, None)
    return _answer(state, visited=visited, waiting=waiting, thread_id=thread_id,
                   conversation_id=state.get("conversation_id"))
