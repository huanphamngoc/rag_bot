"""The ingestion flow as a LangGraph graph.

    catalog ─► validate ─┬─(errors)──────────────────────────────────────► finalize
                         └─► plan ─► extract ─┬─(nothing changed)─► next ─┐
                                              └─► quality ─┬─(error)─► next
                                                            └─► stage ─► next
                             next ─┬─(more doc types)─► plan
                                   └─► lexemes ─► approve ─┬─(ok/under the limit)─► embed ─► finalize
                                                           └─(refused)───────────────────────► finalize

Why a graph and not the straight-line function this replaced: the three ways a document type can end
early - the rules do not match the database, nothing changed, a quality rule failed - were ``if``
statements and early returns in the middle of a 60-line function. As edges they are visible, and every
run reports the nodes it actually visited, so "did this batch reach the model?" is answered by the
trace rather than by reading code.

Two deliberate constraints:

* **the state stays JSON-serialisable.** Documents, rows and connections never enter it; they live in
  the ``IngestOps`` object, which the graph reaches through a key in the config rather than through the
  state. That is what lets the Postgres checkpointer store the state at all.
* **the checkpointer is not the watermark.** It records where in the *flow* a run stopped, so a run
  that died while embedding resumes at that node. What gets read from the source is still decided by
  ``ingest.watermark``, which commits with the documents.
"""
from __future__ import annotations

import contextlib
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

log = logging.getLogger(__name__)

COUNTERS = ("keys", "documents", "inserted", "updated", "refreshed", "unchanged", "deactivated",
            "reactivated", "versions_created", "chunks_added", "chunks_removed", "chunks_moved",
            "vectors_carried", "to_change_id", "missing_keys")


class NoSuchRun(RuntimeError):
    """resume_ingest was given a thread id that this vector database has no checkpoint for."""


class IngestOps(Protocol):
    """Everything the graph does to a database. :class:`crawlerrag.ingest.pipeline.PostgresOps` is the
    implementation; the tests pass a recording fake."""

    def read_catalog(self) -> Any: ...
    def validate(self, catalog: Any) -> list[Any]: ...
    def begin_batch(self, doc_type: str, *, full: bool) -> dict: ...
    def extract(self, plan: dict) -> dict: ...
    def quality(self, plan: dict) -> list[Any]: ...
    def stage(self, plan: dict) -> dict: ...
    def finish_batch(self, plan: dict, result: dict) -> None: ...
    def fail_batch(self, plan: dict, error: str) -> None: ...
    def refresh_lexemes(self) -> int: ...
    def pending_embeddings(self) -> int: ...
    def pending_cost(self) -> dict: ...
    def embed(self, max_chunks: int | None) -> dict: ...
    def mark_indexed(self) -> int: ...


class IngestState(TypedDict, total=False):
    """Plain data only - the Postgres checkpointer has to be able to store it."""
    doc_types: list[str]
    cursor: int
    full: bool
    do_embed: bool
    max_chunks: int | None
    catalog_digest: str | None
    findings: list[dict]
    batches: list[dict]
    current: dict | None
    extract: dict | None
    stage: dict | None
    batch_status: str | None
    lexemes: int | None
    embed: dict | None
    # The approval gate in front of the embedding step. approval_chunks is the threshold the run was
    # started with (part of the state, so a resumed run keeps it); approved is None when the gate did
    # not apply at all, which is the common case.
    approval_chunks: int
    approval: dict | None
    approved: bool | None
    status: str


# ---------------------------------------------------------------- reaching the operations
_OPS: dict[str, IngestOps] = {}


def _ops(config) -> IngestOps:
    key = (config or {}).get("configurable", {}).get("ops_key")
    if key not in _OPS:
        raise RuntimeError("the ingest graph was invoked without its operations")
    return _OPS[key]


def _finding(item: Any) -> dict:
    return {"level": item.level, "doc_type": getattr(item, "doc_type", None), "message": item.message}


# ---------------------------------------------------------------- nodes
def node_catalog(state: IngestState, config) -> dict:
    """Read the source schema once per run: the rules are checked against it, not against memory."""
    catalog = _ops(config).read_catalog()
    return {"catalog_digest": getattr(catalog, "digest", None)}


def node_validate(state: IngestState, config) -> dict:
    findings = [_finding(f) for f in _ops(config).validate(None)]
    bad = [f for f in findings if f["level"] == "error"]
    if bad:
        for f in bad:
            log.error("rule does not match the database", extra={"doc_type": f["doc_type"],
                                                                "detail": f["message"]})
    return {"findings": findings, "status": "failed" if bad else "running"}


def node_plan(state: IngestState, config) -> dict:
    name = state["doc_types"][state["cursor"]]
    plan = _ops(config).begin_batch(name, full=bool(state.get("full")))
    return {"current": plan, "extract": None, "stage": None, "batch_status": None}


def node_extract(state: IngestState, config) -> dict:
    plan = state["current"]
    summary = _ops(config).extract(plan)
    # An incremental window with no changed keys is a finished batch, not a failure.
    nothing = plan["mode"] == "incremental" and not summary.get("keys")
    return {"extract": summary, "batch_status": "nothing" if nothing else None}


def node_quality(state: IngestState, config) -> dict:
    plan = state["current"]
    findings = [_finding(f) for f in _ops(config).quality(plan)]
    failed = [f for f in findings if f["level"] == "error"]
    if failed:
        _ops(config).fail_batch(plan, "; ".join(f["message"] for f in failed))
    return {"findings": list(state.get("findings") or []) + findings,
            "batch_status": "failed" if failed else None}


def node_stage(state: IngestState, config) -> dict:
    return {"stage": _ops(config).stage(state["current"]), "batch_status": "succeeded"}


def node_next_doc_type(state: IngestState, config) -> dict:
    """Close the batch of the doc type just handled and move the cursor."""
    plan, status = state["current"], state.get("batch_status") or "succeeded"
    result = {**(state.get("extract") or {}), **(state.get("stage") or {}), "status": status}
    if status != "failed":
        _ops(config).finish_batch(plan, result)
    summary = {"doc_type": plan["doc_type"], "batch_id": plan["batch_id"], "mode": plan["mode"],
               "reason": plan["reason"], "from_change_id": plan["from_change_id"], "status": status}
    summary.update({name: result[name] for name in COUNTERS if name in result})
    return {"batches": list(state.get("batches") or []) + [summary], "cursor": state["cursor"] + 1,
            "current": None, "extract": None, "stage": None, "batch_status": None}


def node_lexemes(state: IngestState, config) -> dict:
    """Only worth doing when chunks moved in or out: it recounts every lexeme in the index."""
    touched = any(b.get("chunks_added") or b.get("chunks_removed") for b in state.get("batches") or [])
    if not touched:
        return {"lexemes": None}
    return {"lexemes": _ops(config).refresh_lexemes()}


def node_approve_embed(state: IngestState, config) -> dict:
    """Pause in front of the only step of the run that costs money.

    Embedding is the one paid call in an ingestion run, and `ingest --loop` runs unattended, so a rule
    change that suddenly re-embeds the whole index would be paid for before anyone saw it. Above the
    threshold the run stops here with the cost in hand and waits to be resumed - the documents and the
    watermark are already committed, so nothing is lost by waiting, and the chunks stay pending.
    """
    limit = int(state.get("approval_chunks") or 0)
    if limit <= 0 or not state.get("do_embed"):
        return {"approved": None, "approval": None}
    cost = _ops(config).pending_cost()
    if int(cost.get("chunks") or 0) <= limit:
        return {"approved": None, "approval": None}
    payload = {"question": "embed these chunks?", "limit": limit,
               "chunks": cost["chunks"], "chars": cost["chars"],
               "doc_types": list(state.get("doc_types") or [])}
    log.info("waiting for approval before embedding", extra={"chunks": cost["chunks"],
                                                             "chars": cost["chars"], "limit": limit})
    answer = interrupt(payload)
    approved = bool(answer) if not isinstance(answer, dict) else bool(answer.get("approved"))
    return {"approved": approved, "approval": payload}


def node_embed(state: IngestState, config) -> dict:
    if not state.get("do_embed"):
        return {"embed": None}
    ops = _ops(config)
    if not ops.pending_embeddings():
        return {"embed": None}
    return {"embed": ops.embed(state.get("max_chunks"))}


def node_finalize(state: IngestState, config) -> dict:
    batches = state.get("batches") or []
    if state.get("status") == "failed" or any(b["status"] == "failed" for b in batches):
        status = "failed"
    elif state.get("approved") is False:
        # The documents are written and the watermark has moved; only the embedding was refused.
        status = "embedding_refused"
    elif batches and all(b["status"] == "nothing" for b in batches):
        status = "nothing"
    else:
        status = "succeeded"
    if status != "failed":
        _ops(config).mark_indexed()
    return {"status": status}


# ---------------------------------------------------------------- edges
def route_after_validate(state: IngestState) -> str:
    if state.get("status") == "failed":
        return "finalize"
    return "finalize" if not state["doc_types"] else "plan"


def route_after_extract(state: IngestState) -> str:
    return "nothing" if state.get("batch_status") == "nothing" else "quality"


def route_after_quality(state: IngestState) -> str:
    return "next" if state.get("batch_status") == "failed" else "stage"


def route_after_approval(state: IngestState) -> str:
    return "finalize" if state.get("approved") is False else "embed"


def route_next(state: IngestState) -> str:
    return "plan" if state["cursor"] < len(state["doc_types"]) else "lexemes"


def build_ingest_graph(checkpointer=None):
    graph = StateGraph(IngestState)
    graph.add_node("catalog", node_catalog)
    graph.add_node("validate", node_validate)
    graph.add_node("plan", node_plan)
    graph.add_node("extract", node_extract)
    graph.add_node("quality", node_quality)
    graph.add_node("stage", node_stage)
    graph.add_node("next_doc_type", node_next_doc_type)
    graph.add_node("lexemes", node_lexemes)
    graph.add_node("approve_embed", node_approve_embed)
    graph.add_node("embed", node_embed)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("catalog")
    graph.add_edge("catalog", "validate")
    graph.add_conditional_edges("validate", route_after_validate,
                               {"plan": "plan", "finalize": "finalize"})
    graph.add_edge("plan", "extract")
    graph.add_conditional_edges("extract", route_after_extract,
                               {"quality": "quality", "nothing": "next_doc_type"})
    graph.add_conditional_edges("quality", route_after_quality,
                               {"stage": "stage", "next": "next_doc_type"})
    graph.add_edge("stage", "next_doc_type")
    graph.add_conditional_edges("next_doc_type", route_next, {"plan": "plan", "lexemes": "lexemes"})
    graph.add_edge("lexemes", "approve_embed")
    graph.add_conditional_edges("approve_embed", route_after_approval,
                               {"embed": "embed", "finalize": "finalize"})
    graph.add_edge("embed", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------- the checkpointer
@contextlib.contextmanager
def postgres_checkpointer(settings):
    """A Postgres checkpointer whose tables live in the ``graph`` schema, or None when switched off.

    ``search_path`` is how the tables are kept out of ``rag`` and ``ingest``: the saver creates and
    queries them unqualified.
    """
    if settings is None or not getattr(settings, "ingest_graph_checkpoint", False):
        yield None
        return
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    with psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row,
                         options="-c search_path=graph,public") as conn:
        saver = PostgresSaver(conn)
        saver.setup()
        yield saver


@contextlib.contextmanager
def _saver(given, settings):
    """The checkpointer to use: one handed in (tests pass ``MemorySaver``), else a Postgres one."""
    if given is not None:
        yield given
        return
    with postgres_checkpointer(settings) as saver:
        yield saver


# ---------------------------------------------------------------- invoking it
@dataclass
class IngestRun:
    state: dict
    visited: list[str]
    thread_id: str
    # Set when the run stopped at the approval gate: what it would cost, to show and to resume with.
    waiting: dict | None = None

    @property
    def paused(self) -> bool:
        return self.waiting is not None


INTERRUPT = "__interrupt__"


def _stream(graph, payload, config, state: dict) -> tuple[dict, list[str], dict | None]:
    """Measured on langgraph 0.6.11: a pause arrives as the key ``__interrupt__`` whose value is a tuple
    of ``Interrupt`` objects, not a state delta - handing it to ``dict.update`` raises TypeError."""
    visited: list[str] = []
    waiting: dict | None = None
    for update in graph.stream(payload, config, stream_mode="updates"):
        for node, delta in update.items():
            if node == INTERRUPT:
                items = delta if isinstance(delta, (list, tuple)) else (delta,)
                value = getattr(items[0], "value", None) if items else None
                waiting = value if isinstance(value, dict) else {"value": value}
                continue
            visited.append(node)
            state.update(delta or {})
    return state, visited, waiting


class ApprovalNeedsCheckpointer(RuntimeError):
    """The approval gate was armed while the checkpointer was off: the pause could not be resumed."""


def run_ingest(ops: IngestOps, doc_types: Sequence[str], *, full: bool = False, embed: bool = True,
               max_chunks: int | None = None, thread_id: str | None = None,
               checkpointer_settings=None, approval_chunks: int = 0, checkpointer=None) -> IngestRun:
    if approval_chunks > 0 and checkpointer is None \
            and not getattr(checkpointer_settings, "ingest_graph_checkpoint", False):
        # Measured: interrupt() stops the run even with no checkpointer, but then there is no stored
        # state to resume from, so the pause would be a dead end. Refusing here is clearer.
        raise ApprovalNeedsCheckpointer(
            "INGEST_EMBED_APPROVAL_CHUNKS needs INGEST_GRAPH_CHECKPOINT=true: a pause that cannot be "
            "resumed would strand the run. Turn the checkpointer on, or set the threshold to 0.")
    initial: dict = {"doc_types": list(doc_types), "cursor": 0, "full": full, "do_embed": embed,
                     "max_chunks": max_chunks, "findings": [], "batches": [], "status": "running",
                     "catalog_digest": None, "current": None, "extract": None, "stage": None,
                     "batch_status": None, "lexemes": None, "embed": None,
                     "approval_chunks": approval_chunks, "approval": None, "approved": None}
    thread = thread_id or uuid.uuid4().hex
    key = uuid.uuid4().hex
    _OPS[key] = ops
    try:
        with _saver(checkpointer, checkpointer_settings) as saver:
            graph = build_ingest_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": thread, "ops_key": key}, "recursion_limit": 400}
            state, visited, waiting = _stream(graph, initial, config, dict(initial))
    finally:
        _OPS.pop(key, None)
    log.info("ingest run finished", extra={"status": state.get("status"), "nodes": len(visited),
                                          "thread_id": thread, "waiting": bool(waiting)})
    return IngestRun(state=state, visited=visited, thread_id=thread, waiting=waiting)


def resume_ingest(ops: IngestOps, *, thread_id: str, checkpointer_settings=None,
                  answer: bool | None = None, checkpointer=None) -> IngestRun:
    """Continue a run that stopped inside a node, from the last checkpoint it reached.

    ``answer`` is what to tell a waiting ``interrupt()``: True approves the embedding, False refuses it.
    Left as None the run simply carries on, which is how a run that *died* inside a node resumes.
    """
    key = uuid.uuid4().hex
    _OPS[key] = ops
    try:
        with _saver(checkpointer, checkpointer_settings) as saver:
            if saver is None:
                raise NoSuchRun("resuming needs the Postgres checkpointer "
                                "(INGEST_GRAPH_CHECKPOINT=true)")
            graph = build_ingest_graph(checkpointer=saver)
            config = {"configurable": {"thread_id": thread_id, "ops_key": key}, "recursion_limit": 400}
            before = graph.get_state(config)
            if not before or not before.values:
                raise NoSuchRun(f"no checkpointed run for thread {thread_id!r}")
            payload = None if answer is None else Command(resume=answer)
            state, visited, waiting = _stream(graph, payload, config, dict(before.values))
    finally:
        _OPS.pop(key, None)
    return IngestRun(state=state, visited=visited, thread_id=thread_id, waiting=waiting)
