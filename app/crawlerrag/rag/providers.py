"""Pluggable embedding / chat back ends, spoken over plain HTTP.

One small retrying JSON client instead of a vendor SDK per provider, because the
whole point here is that the model is a configuration value: Gemini, any
OpenAI-compatible endpoint (OpenAI, DeepSeek, OpenRouter, Together, vLLM,
LM Studio...), Anthropic, or a local Ollama.

Deliberate differences from the crawler's HTTP client (it fetches publisher sites):

* no crawler identity headers - the User-Agent/From pair identifies this crawler
  to *data publishers*; a model API has no business receiving a contact address;
* no robots.txt handling - these are APIs we hold credentials for, not sites we crawl;
* API keys never reach a log record (only provider + model are logged).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, Sequence

import httpx

from crawlerrag.http import parse_retry_after

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

# Where each provider lives when RAG_*_BASE_URL is not set.
DEFAULT_BASE_URL = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://localhost:11434",
}

# Environment variables accepted for each provider's key, in order, when
# RAG_EMBED_API_KEY / RAG_CHAT_API_KEY are not set.
KEY_ENV_VARS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    "vertex": (),          # OAuth via Application Default Credentials, no API key
}

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def vertex_base_url(project: str, location: str) -> str:
    """Publisher root for Vertex AI; ``:predict`` / ``:generateContent`` are appended to it."""
    if not project:
        raise ProviderError("vertex needs a project: set RAG_VERTEX_PROJECT (or GCP_PROJECT).")
    location = (location or "us-central1").strip()
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/publishers/google"


class _HttpxAuthTransport:
    """The transport google-auth needs to refresh a token, implemented with httpx.

    ``google.auth.transport.requests`` would do this, but it pulls in ``requests`` purely to
    refresh a token - httpx is already a dependency, and the interface is three attributes.
    """

    class _Response:
        def __init__(self, response: httpx.Response) -> None:
            self.status = response.status_code
            self.headers = response.headers
            self.data = response.content

    def __call__(self, url, method="GET", body=None, headers=None, timeout=None, **kwargs):
        response = httpx.request(method, url, content=body, headers=headers,
                                 timeout=timeout or 60.0, follow_redirects=True)
        return self._Response(response)


def google_token_provider(scope: str = VERTEX_SCOPE) -> Callable[[], str]:
    """A callable returning a fresh OAuth access token from Application Default Credentials.

    Tokens last an hour and indexing can run longer, so the token is resolved per request and
    refreshed by google-auth when it is about to expire - never frozen into a header.
    """
    try:
        import google.auth
    except ImportError as exc:   # pragma: no cover - dependency is in requirements.txt
        raise ProviderError("vertex needs the google-auth package (it is in app/requirements.txt).") from exc
    try:
        credentials, _ = google.auth.default(scopes=[scope])
    except Exception as exc:
        raise ProviderError(
            "vertex could not load Application Default Credentials: "
            f"{type(exc).__name__}: {exc}. Run \"gcloud auth application-default login\" on the host and "
            "point GCLOUD_CONFIG (in .env) at the gcloud config directory, or set "
            "GOOGLE_APPLICATION_CREDENTIALS to a service account key file."
        ) from exc
    transport = _HttpxAuthTransport()

    def token() -> str:
        if not credentials.valid:
            try:
                credentials.refresh(transport)
            except Exception as exc:
                # Stale or revoked credentials are the common case here (a deleted account, a
                # refresh token older than the session), and the fix is always the same command.
                raise ProviderError(
                    f"vertex could not refresh the Google credentials: {type(exc).__name__}: {exc}. "
                    "Run \"gcloud auth application-default login\" on the host to renew them."
                ) from exc
        return credentials.token

    return token


class ProviderError(RuntimeError):
    """The model API refused the request, or kept failing after every attempt.

    ``status`` is the HTTP status of the last answer, when there was one (429 = quota)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def resolve_api_key(provider: str, configured: str | None) -> str | None:
    if configured:
        return configured
    for name in KEY_ENV_VARS.get(provider, ()):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def base_url_for(provider: str, configured: str | None) -> str:
    return (configured or DEFAULT_BASE_URL[provider]).rstrip("/")


@dataclass
class JsonApi:
    """POST JSON, retry the statuses that mean "try again", give up loudly."""

    base_url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 120.0
    max_attempts: int = 5
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    transport: httpx.BaseTransport | None = None
    sleep: Any = time.sleep
    # Vertex AI authenticates with an OAuth token that expires after an hour, which is shorter than a
    # long indexing run - so the credential is resolved per request instead of frozen into a header.
    token_provider: Callable[[], str] | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Accept": "application/json", "Content-Type": "application/json", **self.headers},
            timeout=httpx.Timeout(self.timeout_s, connect=15.0),
            transport=self.transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JsonApi":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def post(self, path: str, payload: dict, *, params: dict | None = None, context: str = "") -> dict:
        last: str | None = None
        last_status: int | None = None
        headers = {"Authorization": f"Bearer {self.token_provider()}"} if self.token_provider else None
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._client.post(path, json=payload, params=params, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
                log.warning("model api transport error",
                            extra={"context": context, "attempt": attempt, "error": last})
                if attempt == self.max_attempts:
                    break
                self._backoff(attempt, None)
                continue
            if resp.status_code in RETRYABLE_STATUS:
                last, last_status = f"HTTP {resp.status_code}: {resp.text[:500]}", resp.status_code
                log.warning("model api retryable status",
                            extra={"context": context, "attempt": attempt, "status_code": resp.status_code})
                if attempt == self.max_attempts:
                    break
                self._backoff(attempt, parse_retry_after(resp.headers.get("Retry-After")))
                continue
            if resp.status_code >= 400:
                # 400/401/403/404 are configuration problems (bad key, unknown model): failing fast
                # is more useful than five identical refusals.
                raise ProviderError(f"{context}: HTTP {resp.status_code} - {resp.text[:800]}", status=resp.status_code)
            try:
                return resp.json()
            except ValueError as exc:
                last = f"invalid JSON: {exc}"
                if attempt == self.max_attempts:
                    break
                self._backoff(attempt, None)
        raise ProviderError(f"{context}: giving up after {self.max_attempts} attempts - {last}", status=last_status)

    def stream_sse(self, path: str, payload: dict, *, params: dict | None = None,
                   context: str = "") -> Iterator[dict]:
        """POST and yield each ``data:`` object of a Server-Sent Events response.

        Deliberately **not** retried. ``post`` can retry because a failed attempt produced nothing;
        a stream that breaks halfway has already handed text to the caller, and retrying would repeat
        it. A failure here is reported and the caller falls back or gives up.
        """
        headers = {"Authorization": f"Bearer {self.token_provider()}"} if self.token_provider else None
        try:
            with self._client.stream("POST", path, json=payload, params=params, headers=headers) as resp:
                if resp.status_code >= 400:
                    body = resp.read()[:800].decode("utf-8", "replace")
                    raise ProviderError(f"{context}: HTTP {resp.status_code} - {body}",
                                        status=resp.status_code)
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if not chunk or chunk == "[DONE]":
                        continue
                    try:
                        yield json.loads(chunk)
                    except ValueError as exc:
                        raise ProviderError(f"{context}: invalid JSON in the stream - {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{context}: {type(exc).__name__}: {exc}") from exc

    def _backoff(self, attempt: int, retry_after: float | None) -> None:
        delay = min(self.backoff_max_s, self.backoff_base_s * (2 ** (attempt - 1)))
        delay *= 0.5 + random.random() / 2
        if retry_after is not None:
            delay = max(delay, min(retry_after, 300.0))
        self.sleep(delay)


# ---------------------------------------------------------------- embeddings
class EmbeddingProvider(Protocol):
    provider: str
    model: str
    batch_size: int

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]: ...
    def probe_dim(self) -> int: ...
    def close(self) -> None: ...


@dataclass
class _BaseEmbeddings:
    model: str
    api: JsonApi
    dim: int | None = None          # requested output dimensionality, when the model supports truncation
    batch_size: int = 64

    def close(self) -> None:
        self.api.close()

    def probe_dim(self) -> int:
        """Ask the live model for one vector - the authoritative dimension, never guessed."""
        return len(self.embed(["dimension probe"])[0])


class GeminiEmbeddings(_BaseEmbeddings):
    provider = "gemini"

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        requests = []
        for text in texts:
            req: dict[str, Any] = {"model": f"models/{self.model}",
                                   "content": {"parts": [{"text": text}]},
                                   "taskType": task}
            if self.dim:
                req["outputDimensionality"] = self.dim
            requests.append(req)
        data = self.api.post(f"/models/{self.model}:batchEmbedContents", {"requests": requests},
                             context=f"gemini embed ({self.model})")
        vectors = [e.get("values") or [] for e in data.get("embeddings", [])]
        if len(vectors) != len(texts):
            raise ProviderError(f"gemini embed: asked for {len(texts)} vectors, got {len(vectors)}")
        return vectors


class OpenAIEmbeddings(_BaseEmbeddings):
    """Also covers every OpenAI-compatible endpoint (set RAG_EMBED_BASE_URL)."""

    provider = "openai"

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": list(texts)}
        if self.dim:
            payload["dimensions"] = self.dim
        data = self.api.post("/embeddings", payload, context=f"openai embed ({self.model})")
        items = data.get("data") or []
        if len(items) != len(texts):
            raise ProviderError(f"openai embed: asked for {len(texts)} vectors, got {len(items)}")
        # The API may return the batch out of order; "index" is what maps a vector back to its text.
        items = sorted(items, key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in items]


class OllamaEmbeddings(_BaseEmbeddings):
    provider = "ollama"

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        data = self.api.post("/api/embed", {"model": self.model, "input": list(texts)},
                             context=f"ollama embed ({self.model})")
        vectors = data.get("embeddings") or []
        if len(vectors) != len(texts):
            raise ProviderError(f"ollama embed: asked for {len(texts)} vectors, got {len(vectors)}")
        return vectors


class VertexEmbeddings(_BaseEmbeddings):
    """Vertex AI speaks the prediction API, not the AI Studio embedding API.

    Same Gemini models, different envelope: instances[]/predictions[] instead of
    requests[]/embeddings[], ``task_type`` in snake_case, and an OAuth token instead of a key.
    """

    provider = "vertex"

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
        payload: dict[str, Any] = {"instances": [{"content": t, "task_type": task} for t in texts]}
        if self.dim:
            payload["parameters"] = {"outputDimensionality": self.dim}
        data = self.api.post(f"/models/{self.model}:predict", payload,
                             context=f"vertex embed ({self.model})")
        predictions = data.get("predictions", [])
        vectors = [(p.get("embeddings") or {}).get("values") or [] for p in predictions]
        if len(vectors) != len(texts):
            raise ProviderError(f"vertex embed: asked for {len(texts)} vectors, got {len(vectors)}")
        # Vertex reports, per text, the tokens billed and whether the text was cut at the model's input limit.
        stats = [(p.get("embeddings") or {}).get("statistics") or {} for p in predictions]
        self.last_usage = {"tokens": sum(int(s.get("token_count") or 0) for s in stats),
                           "truncated": sum(1 for s in stats if s.get("truncated"))}
        return vectors


EMBEDDING_CLASSES = {"gemini": GeminiEmbeddings, "openai": OpenAIEmbeddings, "ollama": OllamaEmbeddings,
                     "vertex": VertexEmbeddings}


# ---------------------------------------------------------------- chat
@dataclass(frozen=True)
class ChatReply:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None        # everything billed as output, reasoning included
    reasoning_tokens: int | None = None     # the hidden part of output_tokens, when the API reports it
    truncated: bool = False                 # stopped by the output token limit, text is incomplete


class ChatProvider(Protocol):
    provider: str
    model: str

    def complete(self, system: str, prompt: str, *, reasoning: bool = True) -> ChatReply:
        """``reasoning=False`` asks for no hidden thinking where the model allows it (short,
        mechanical calls such as rewriting a follow-up question)."""
    def close(self) -> None: ...


def complete_streaming(chat: "ChatProvider", system: str, prompt: str,
                       on_delta: Callable[[str], None], *, reasoning: bool = True) -> ChatReply:
    """Stream where the provider can, otherwise hand the answer over in one piece.

    Only the Gemini/Vertex family implements ``stream``; ollama and the OpenAI-compatible endpoints
    fall back here, so the page behaves the same for all of them - just without the typing effect.
    """
    streamer = getattr(chat, "stream", None)
    if streamer is None:
        reply = chat.complete(system, prompt, reasoning=reasoning)
        if reply.text:
            on_delta(reply.text)
        return reply
    return streamer(system, prompt, on_delta=on_delta, reasoning=reasoning)


@dataclass
class _BaseChat:
    model: str
    api: JsonApi
    max_tokens: int = 1024
    temperature: float = 0.1
    thinking_budget: int | None = None

    def close(self) -> None:
        self.api.close()


# Gemini 2.5 Flash / Flash-Lite accept thinkingBudget 0; 2.5 Pro cannot switch thinking off.
_THINKING_CAN_BE_OFF = re.compile(r"gemini-2\.5-flash", re.IGNORECASE)


class GeminiChat(_BaseChat):
    provider = "gemini"

    def _thinking_budget(self, reasoning: bool) -> int | None:
        if not reasoning and _THINKING_CAN_BE_OFF.search(self.model):
            return 0
        return self.thinking_budget

    def _payload(self, system: str, prompt: str, reasoning: bool) -> dict:
        config: dict = {"temperature": self.temperature, "maxOutputTokens": self.max_tokens}
        budget = self._thinking_budget(reasoning)
        if budget is not None:
            config["thinkingConfig"] = {"thinkingBudget": budget}
        return {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }

    def complete(self, system: str, prompt: str, *, reasoning: bool = True) -> ChatReply:
        payload = self._payload(system, prompt, reasoning)
        data = self.api.post(f"/models/{self.model}:generateContent", payload,
                             context=f"gemini chat ({self.model})")
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError(f"gemini chat: no candidate returned ({data.get('promptFeedback')})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        usage = data.get("usageMetadata") or {}
        visible, thoughts = usage.get("candidatesTokenCount"), usage.get("thoughtsTokenCount")
        output = None if visible is None and thoughts is None else (visible or 0) + (thoughts or 0)
        # thought summaries (if ever requested) come back as parts flagged "thought": not answer text
        return ChatReply("".join(p.get("text", "") for p in parts if not p.get("thought")).strip(),
                         usage.get("promptTokenCount"), output, thoughts,
                         truncated=candidates[0].get("finishReason") == "MAX_TOKENS")


    def stream(self, system: str, prompt: str, *, on_delta: Callable[[str], None],
               reasoning: bool = True) -> ChatReply:
        """``:streamGenerateContent?alt=sse``, verified against Vertex AI on 2026-10-06.

        Measured shape: every event carries ``candidates[0].content.parts[*].text``, and the usage
        counts arrive **only in the last event** - so the tokens (and the cost MLflow computes from
        them) are whatever the final event reported.
        """
        payload = self._payload(system, prompt, reasoning)
        text: list[str] = []
        usage: dict = {}
        finish: str | None = None
        for event in self.api.stream_sse(f"/models/{self.model}:streamGenerateContent", payload,
                                         params={"alt": "sse"},
                                         context=f"{self.provider} chat stream ({self.model})"):
            usage = event.get("usageMetadata") or usage
            for candidate in event.get("candidates") or []:
                finish = candidate.get("finishReason") or finish
                for part in (candidate.get("content") or {}).get("parts") or []:
                    # thought summaries are not answer text, same as in complete()
                    piece = part.get("text") or "" if not part.get("thought") else ""
                    if piece:
                        text.append(piece)
                        on_delta(piece)
        visible, thoughts = usage.get("candidatesTokenCount"), usage.get("thoughtsTokenCount")
        output = None if visible is None and thoughts is None else (visible or 0) + (thoughts or 0)
        return ChatReply("".join(text).strip(), usage.get("promptTokenCount"), output, thoughts,
                         truncated=finish == "MAX_TOKENS")


class OpenAIChat(_BaseChat):
    provider = "openai"

    def complete(self, system: str, prompt: str, *, reasoning: bool = True) -> ChatReply:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = self.api.post("/chat/completions", payload, context=f"openai chat ({self.model})")
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("openai chat: no choice returned")
        usage = data.get("usage") or {}
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        return ChatReply((choices[0].get("message") or {}).get("content", "").strip(),
                         usage.get("prompt_tokens"), usage.get("completion_tokens"), reasoning_tokens,
                         truncated=choices[0].get("finish_reason") == "length")


class AnthropicChat(_BaseChat):
    provider = "anthropic"

    def complete(self, system: str, prompt: str, *, reasoning: bool = True) -> ChatReply:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = self.api.post("/messages", payload, context=f"anthropic chat ({self.model})")
        blocks = [b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text"]
        usage = data.get("usage") or {}
        return ChatReply("".join(blocks).strip(), usage.get("input_tokens"), usage.get("output_tokens"),
                         truncated=data.get("stop_reason") == "max_tokens")


class OllamaChat(_BaseChat):
    provider = "ollama"

    def complete(self, system: str, prompt: str, *, reasoning: bool = True) -> ChatReply:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        data = self.api.post("/api/chat", payload, context=f"ollama chat ({self.model})")
        return ChatReply((data.get("message") or {}).get("content", "").strip(),
                         data.get("prompt_eval_count"), data.get("eval_count"),
                         truncated=data.get("done_reason") == "length")


class VertexChat(GeminiChat):
    """``:generateContent`` takes the same body on Vertex; only the host and the credential change."""

    provider = "vertex"


CHAT_CLASSES = {"gemini": GeminiChat, "openai": OpenAIChat, "anthropic": AnthropicChat, "ollama": OllamaChat,
                "vertex": VertexChat}


# ---------------------------------------------------------------- wiring
def auth_headers(provider: str, key: str | None, *, what: str) -> dict[str, str]:
    """The header carrying the credential each provider expects."""
    if provider in ("ollama", "vertex"):
        return {}                     # ollama needs none; vertex signs every request with an OAuth token
    if not key:
        names = " / ".join(KEY_ENV_VARS.get(provider, ())) or "-"
        raise ProviderError(f"{provider} needs an API key: set RAG_{what}_API_KEY (or {names}).")
    if provider == "gemini":
        return {"x-goog-api-key": key}
    if provider == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    return {"Authorization": f"Bearer {key}"}


def _api(settings, provider: str, base_url: str | None, api_key: str | None, *, what: str,
         transport: httpx.BaseTransport | None) -> JsonApi:
    """One JSON client, with the credential style and address the provider needs."""
    if provider == "vertex":
        project = settings.rag_vertex_project or getattr(settings, "gcp_project", None)
        return JsonApi(base_url=base_url or vertex_base_url(project, settings.rag_vertex_location),
                       token_provider=google_token_provider(), timeout_s=settings.rag_api_timeout_s,
                       max_attempts=settings.rag_max_attempts, transport=transport)
    return JsonApi(base_url=base_url_for(provider, base_url),
                   headers=auth_headers(provider, resolve_api_key(provider, api_key), what=what),
                   timeout_s=settings.rag_api_timeout_s, max_attempts=settings.rag_max_attempts,
                   transport=transport)


def embedding_provider(settings, *, transport: httpx.BaseTransport | None = None) -> EmbeddingProvider:
    provider = settings.rag_embed_provider
    api = _api(settings, provider, settings.rag_embed_base_url, settings.rag_embed_api_key,
               what="EMBED", transport=transport)
    log.info("embedding provider ready", extra={"provider": provider, "model": settings.rag_embed_model,
                                                "endpoint": api.base_url})
    return EMBEDDING_CLASSES[provider](model=settings.rag_embed_model, api=api, dim=settings.rag_embed_dim,
                                       batch_size=settings.rag_embed_batch)


def chat_provider(settings, *, transport: httpx.BaseTransport | None = None) -> ChatProvider:
    provider = settings.rag_chat_provider
    api = _api(settings, provider, settings.rag_chat_base_url, settings.rag_chat_api_key,
               what="CHAT", transport=transport)
    log.info("chat provider ready", extra={"provider": provider, "model": settings.rag_chat_model,
                                           "endpoint": api.base_url})
    return CHAT_CLASSES[provider](model=settings.rag_chat_model, api=api,
                                  max_tokens=settings.rag_chat_max_tokens, temperature=settings.rag_chat_temperature,
                                  thinking_budget=getattr(settings, "rag_chat_thinking_budget", None))
