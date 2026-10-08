"""Optional MLflow tracing for `ask` / `search` / `chat` and the web chat.

Off unless ``MLFLOW_TRACKING_URI`` points at an MLflow tracking server. This project ships one:
``docker compose --profile mlflow up -d`` runs MLflow 3.16.1 against its own database in the vector
Postgres, published on http://localhost:5001. With tracing off the decorators are pass-throughs and
``mlflow`` is not even imported. When on, one ``ask`` becomes one trace::

    rag_ask (CHAIN)
    ├── qualify_question (GUARDRAIL)  the rules in rules/qualify.yaml; no model involved
    ├── condense_question (LLM)       from the second turn of a conversation
    ├── hybrid_retrieve (RETRIEVER)
    │   ├── embed_query (EMBEDDING)
    │   ├── vector_search (RETRIEVER)
    │   └── text_search (RETRIEVER)
    └── generate_answer (LLM)         token usage + model/provider, which the server prices

A question the gate refuses produces a trace with the first two spans and **no** LLM or RETRIEVER span
at all - which is what makes "did this question cost anything?" answerable from the trace list, where
the ``qualify_decision`` tag is searchable.

Turns of one conversation share the MLflow session id (the conversation id).

What is sent is exactly what the ``inputs`` / ``outputs`` functions below return: the question,
filters, retrieved excerpts (public regulatory records), prompts and the answer. Function
arguments are never serialised wholesale - ``settings`` carries API keys and ``conn`` a database
handle - so every traced function names the fields it exposes.

Two things MLflow would otherwise do on its own are switched off here: usage telemetry to
mlflow-telemetry.io (``MLFLOW_DISABLE_TELEMETRY``, read when ``mlflow`` is imported) and HTTP
retries that take minutes when the server is down (``MLFLOW_HTTP_REQUEST_MAX_RETRIES`` /
``_TIMEOUT``). Both are defaults only; values already in the environment win.
"""
from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any, Callable
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "crawler-rag"
FEEDBACK_SOURCE = "rag-cli"     # who judged the answer; never a personal identifier
PREVIEW_CHARS = 1000

# MLflow 3.16 span types used here. "guardrail" is the gate in front of the model: a question the rules
# refuse never reaches an LLM span, and the trace shows exactly that.
SPAN_TYPES = {"chain": "CHAIN", "retriever": "RETRIEVER", "embedding": "EMBEDDING", "llm": "LLM",
              "guardrail": "GUARDRAIL"}

# The MLflow server prices LLM spans from model + provider + token usage, using the provider ids
# of its bundled model catalogue (LiteLLM's names).
PROVIDERS = {"vertex": "vertex_ai", "gemini": "gemini", "openai": "openai",
             "anthropic": "anthropic", "ollama": "ollama"}

_ENV_DEFAULTS = {
    "MLFLOW_DISABLE_TELEMETRY": "true",
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "2",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
}

_state: dict[str, Any] = {}     # decided once per process: {"ok": bool, "uri", "experiment", ...}


# ---------------------------------------------------------------- configuration
def _mlflow():
    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    import mlflow
    return mlflow


def enabled() -> bool:
    """True when traces are being recorded; decided on first use from ``MLFLOW_TRACKING_URI``."""
    if "ok" not in _state:
        uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
        _state["ok"] = bool(uri) and configure(
            uri, (os.environ.get("MLFLOW_EXPERIMENT_NAME") or "").strip() or DEFAULT_EXPERIMENT)
    return _state["ok"]


def configure(uri: str, experiment: str = DEFAULT_EXPERIMENT) -> bool:
    """Point tracing at an MLflow server; False (tracing stays off) when that is not possible.

    A stopped server costs one 3-second check here instead of the client's retry policy on the
    first trace, and the chatbot keeps working either way.
    """
    _state["ok"] = False
    if not uri.startswith(("http://", "https://")):
        # file:/sqlite: stores need the full mlflow package and would write inside the container
        log.warning("MLFLOW_TRACKING_URI must be the http(s) URL of an MLflow server; tracing is off",
                    extra={"uri": uri})
        return False
    try:
        mlflow = _mlflow()
    except ImportError:
        log.warning("MLFLOW_TRACKING_URI is set but mlflow-tracing is not installed; tracing is off")
        return False
    # MLflow logs at INFO on its own handler ("Flushing the async trace logging queue before program
    # exit..." after every answer); its warnings - a trace that failed to upload - still show.
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    try:
        httpx.get(f"{uri.rstrip('/')}/health", timeout=3).raise_for_status()
    except httpx.HTTPError as exc:
        log.warning("MLflow server is not reachable; tracing is off for this run",
                    extra={"uri": uri, "error": str(exc)[:200]})
        return False
    try:
        mlflow.set_tracking_uri(uri)
        experiment_id = mlflow.set_experiment(experiment).experiment_id
    except Exception as exc:      # MlflowException, HTTP errors: tracing must never stop an answer
        log.warning("could not open the MLflow experiment; tracing is off",
                    extra={"uri": uri, "experiment": experiment, "error": str(exc)[:300]})
        return False
    _state.update(ok=True, uri=uri, experiment=experiment, experiment_id=experiment_id)
    return True


def reset() -> None:
    """Forget the configuration, so the next call decides again (tests)."""
    if _state.get("ok"):
        flush()
    _state.clear()


def experiment_name() -> str | None:
    return _state.get("experiment") if enabled() else None


def _ui_base() -> str | None:
    """Where a browser reaches the UI. Inside compose the server is http://mlflow:5000, which only
    containers resolve, so compose passes MLFLOW_UI_URL (http://localhost:<port>)."""
    if not enabled():
        return None
    return (os.environ.get("MLFLOW_UI_URL") or _state["uri"]).rstrip("/")


def trace_url(trace_id: str) -> str | None:
    base = _ui_base()
    if base is None:
        return None
    return (f"{base}/#/experiments/{_state['experiment_id']}/traces"
            f"?selectedEvaluationId={quote(trace_id, safe='')}")


def session_url(session_id: str) -> str | None:
    base = _ui_base()
    if base is None:
        return None
    return f"{base}/#/experiments/{_state['experiment_id']}/chat-sessions/{quote(session_id, safe='')}"


# ---------------------------------------------------------------- spans
def _shape(fn: Callable[[Any], Any], value: Any) -> Any:
    try:
        return fn(value)
    except Exception:              # a shaping bug must cost the trace a field, not the user an answer
        log.debug("could not shape a span field", exc_info=True)
        return {"unavailable": "could not be serialised"}


def _arguments(signature: inspect.Signature, args: tuple, kwargs: dict) -> dict:
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return {}
    bound.apply_defaults()
    return dict(bound.arguments)


def traceable(*, run_type: str, name: str, inputs: Callable[[dict], Any],
              outputs: Callable[[Any], Any] | None = None) -> Callable:
    """Trace the decorated function as one span.

    ``inputs`` receives the call's arguments by name and returns what the span records; nothing
    else about the call is sent. ``outputs`` shapes the return value (default: not recorded).
    An exception propagates unchanged; leaving the span with it marks the span as failed (status
    ERROR plus an exception event, recorded by OpenTelemetry underneath MLflow).
    """
    span_type = SPAN_TYPES[run_type]

    def decorate(fn: Callable) -> Callable:
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            with _mlflow().start_span(name=name, span_type=span_type) as span:
                span.set_inputs(_shape(inputs, _arguments(signature, args, kwargs)))
                result = fn(*args, **kwargs)
                if outputs is not None:
                    span.set_outputs(_shape(outputs, result))
                return result

        return wrapper

    return decorate


def _current_span():
    return _mlflow().get_current_active_span() if enabled() else None


def annotate(attributes: dict[str, Any]) -> None:
    """Set attributes on the span currently executing (no-op outside a trace)."""
    span = _current_span()
    if span is not None:
        span.set_attributes({k: v for k, v in attributes.items() if v is not None})


def annotate_model(provider) -> None:
    """Model and provider of an embedding/LLM span; with token usage the server prices LLM spans."""
    if not enabled():
        return
    from mlflow.tracing.constant import SpanAttributeKey
    annotate({SpanAttributeKey.MODEL: provider.model,
              SpanAttributeKey.MODEL_PROVIDER: PROVIDERS.get(provider.provider, provider.provider),
              "temperature": getattr(provider, "temperature", None),
              "max_tokens": getattr(provider, "max_tokens", None)})


def record_usage(reply) -> None:
    """Token usage of an LLM span. ``output_tokens`` already includes Gemini thinking tokens,
    which are billed as output - the cost shown is what the API charges."""
    if not enabled():
        return
    from mlflow.tracing.constant import SpanAttributeKey
    usage = {"input_tokens": reply.prompt_tokens or 0, "output_tokens": reply.output_tokens or 0}
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    annotate({SpanAttributeKey.CHAT_USAGE: usage})


def update_trace(*, tags: dict[str, Any] | None = None, session_id: str | None = None,
                 request_preview: str | None = None, response_preview: str | None = None) -> None:
    """Trace-level fields: searchable tags, the session (conversation) id and the two previews
    the trace list shows. No-op outside a trace."""
    if _current_span() is None:
        return
    kwargs: dict[str, Any] = {}
    if tags:
        kwargs["tags"] = {k: str(v) for k, v in tags.items() if v is not None}
    if session_id:
        kwargs["session_id"] = session_id
    if request_preview is not None:
        kwargs["request_preview"] = request_preview[:PREVIEW_CHARS]
    if response_preview is not None:
        kwargs["response_preview"] = response_preview[:PREVIEW_CHARS]
    if kwargs:
        _mlflow().update_current_trace(**kwargs)


def current_trace_id() -> str | None:
    return _mlflow().get_active_trace_id() if enabled() else None


def flush() -> None:
    """Block until queued traces are sent (tests, reset).

    The CLI does not call this. MLflow batches spans (5 per batch or every 5 s) and uploads them
    from a background thread, and its exit hooks send whatever is left when the process ends. A
    flush costs 1-2 s even with nothing queued: its consumer thread polls with a 1 s timeout.
    """
    if not _state.get("ok"):
        return
    try:
        _mlflow().flush_trace_async_logging()
    except Exception:
        log.warning("MLflow flush failed; the trace may be incomplete", exc_info=True)


def send_feedback(trace_id: str, *, good: bool, comment: str | None = None, name: str = "user_score") -> None:
    """Record a human judgement on a trace (an MLflow assessment of type feedback)."""
    if not enabled():
        raise RuntimeError("MLflow tracing is not configured")
    mlflow = _mlflow()
    from mlflow.entities import AssessmentSource, AssessmentSourceType
    mlflow.log_feedback(trace_id=trace_id, name=name, value=good, rationale=comment,
                        source=AssessmentSource(source_type=AssessmentSourceType.HUMAN,
                                                source_id=FEEDBACK_SOURCE))


# ---------------------------------------------------------------- shaping helpers
def llm_outputs(reply) -> dict:
    """An OpenAI-style chat completion: the shape the MLflow UI renders as a chat."""
    usage: dict = {"prompt_tokens": reply.prompt_tokens or 0, "completion_tokens": reply.output_tokens or 0}
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    if getattr(reply, "reasoning_tokens", None):
        usage["completion_tokens_details"] = {"reasoning_tokens": reply.reasoning_tokens}
    finish = "length" if getattr(reply, "truncated", False) else "stop"
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": reply.text},
                         "finish_reason": finish}],
            "usage": usage}


def documents(hits) -> list[dict]:
    """Retrieved chunks as MLflow retriever documents (page_content / metadata / id)."""
    return [{"page_content": h.text, "id": h.doc_id,
             "metadata": {"rank": n, "doc_id": h.doc_id, "doc_type": h.doc_type, "title": h.title,
                          "url": h.url, "chunk_id": h.chunk_id, "score": round(h.score, 6),
                          "matched_by": h.matched_by, "vector_rank": h.vector_rank, "text_rank": h.text_rank}}
            for n, h in enumerate(hits, start=1)]


def row_ids(rows) -> dict:
    return {"rows": len(rows), "chunk_ids": [r["chunk_id"] for r in rows]}
