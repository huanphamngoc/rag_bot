"""Every model back end: request shape, response parsing, retries, credential handling."""
import httpx
import pytest

from crawlerrag.config import Settings
from crawlerrag.rag import providers


def settings(**kwargs) -> Settings:
    base = {"database_url": "postgresql://x/y", "rag_embed_api_key": "k", "rag_chat_api_key": "k"}
    return Settings(**{**base, **kwargs})


class Recorder:
    """Collects the requests a provider makes and replays canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            status, payload = self.responses.pop(0) if self.responses else (200, {})
            return httpx.Response(status, json=payload)
        return httpx.MockTransport(handler)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]


# ---------------------------------------------------------------- embeddings
def test_gemini_embed_request_and_parse():
    rec = Recorder([(200, {"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]})])
    provider = providers.embedding_provider(
        settings(rag_embed_provider="gemini", rag_embed_model="text-embedding-004", rag_embed_dim=2),
        transport=rec.transport())
    assert provider.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    body = rec.last.read().decode()
    assert "batchEmbedContents" in str(rec.last.url)
    assert '"taskType":"RETRIEVAL_DOCUMENT"' in body
    assert '"outputDimensionality":2' in body
    assert rec.last.headers["x-goog-api-key"] == "k"


def test_gemini_embed_marks_a_query_differently():
    rec = Recorder([(200, {"embeddings": [{"values": [1.0]}]})])
    provider = providers.embedding_provider(settings(rag_embed_provider="gemini"), transport=rec.transport())
    provider.embed(["question"], query=True)
    assert '"taskType":"RETRIEVAL_QUERY"' in rec.last.read().decode()


def test_openai_embed_reorders_by_index():
    rec = Recorder([(200, {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]})])
    provider = providers.embedding_provider(settings(rag_embed_provider="openai"), transport=rec.transport())
    assert provider.embed(["first", "second"]) == [[1.0], [2.0]]
    assert rec.last.headers["authorization"] == "Bearer k"


def test_ollama_embed_needs_no_key():
    rec = Recorder([(200, {"embeddings": [[0.5]]})])
    provider = providers.embedding_provider(
        Settings(database_url="postgresql://x/y", rag_embed_provider="ollama"), transport=rec.transport())
    assert provider.embed(["a"]) == [[0.5]]
    assert "authorization" not in rec.last.headers


def test_probe_dim_uses_the_live_model():
    rec = Recorder([(200, {"data": [{"index": 0, "embedding": [0.0] * 768}]})])
    provider = providers.embedding_provider(settings(rag_embed_provider="openai"), transport=rec.transport())
    assert provider.probe_dim() == 768


def test_short_vector_batch_is_an_error():
    rec = Recorder([(200, {"data": [{"index": 0, "embedding": [1.0]}]})])
    provider = providers.embedding_provider(settings(rag_embed_provider="openai"), transport=rec.transport())
    with pytest.raises(providers.ProviderError, match="asked for 2 vectors"):
        provider.embed(["a", "b"])


# ---------------------------------------------------------------- chat
def test_gemini_chat_parses_parts_and_usage():
    rec = Recorder([(200, {"candidates": [{"content": {"parts": [{"text": "Hel"}, {"text": "lo"}]}}],
                           "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 2}})])
    chat = providers.chat_provider(settings(rag_chat_provider="gemini"), transport=rec.transport())
    reply = chat.complete("sys", "prompt")
    assert (reply.text, reply.prompt_tokens, reply.output_tokens) == ("Hello", 11, 2)
    assert '"system_instruction"' in rec.last.read().decode()


def test_gemini_thinking_tokens_count_as_output_and_truncation_is_flagged():
    rec = Recorder([(200, {"candidates": [{"content": {"parts": [{"text": "plan", "thought": True},
                                                                  {"text": "why was Fent"}]},
                                           "finishReason": "MAX_TOKENS"}],
                           "usageMetadata": {"promptTokenCount": 339, "candidatesTokenCount": 38,
                                             "thoughtsTokenCount": 986}})])
    chat = providers.chat_provider(settings(rag_chat_provider="gemini", rag_chat_model="gemini-2.5-flash"),
                                   transport=rec.transport())
    reply = chat.complete("sys", "prompt")
    assert reply.text == "why was Fent"                      # thought parts are not answer text
    assert (reply.output_tokens, reply.reasoning_tokens) == (1024, 986)   # billed output = visible + thinking
    assert reply.truncated


@pytest.mark.parametrize("model, budget, reasoning, expected", [
    ("gemini-2.5-flash", None, False, 0),          # mechanical call: thinking off
    ("gemini-2.5-flash-lite", None, False, 0),
    ("gemini-2.5-pro", None, False, None),         # Pro cannot turn thinking off: leave the default
    ("gemini-2.5-flash", None, True, None),        # answers: model default unless configured
    ("gemini-2.5-flash", 256, True, 256),
])
def test_gemini_thinking_budget_in_request(model, budget, reasoning, expected):
    rec = Recorder([(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})])
    chat = providers.chat_provider(settings(rag_chat_provider="gemini", rag_chat_model=model,
                                            rag_chat_thinking_budget=budget), transport=rec.transport())
    chat.complete("sys", "prompt", reasoning=reasoning)
    import json
    config = json.loads(rec.last.read())["generationConfig"]
    assert config.get("thinkingConfig", {}).get("thinkingBudget") == expected


def test_openai_reports_reasoning_tokens_and_length_cutoff():
    rec = Recorder([(200, {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 90,
                                     "completion_tokens_details": {"reasoning_tokens": 64}}})])
    chat = providers.chat_provider(settings(rag_chat_provider="openai"), transport=rec.transport())
    reply = chat.complete("sys", "prompt")
    assert (reply.output_tokens, reply.reasoning_tokens, reply.truncated) == (90, 64, True)


def test_anthropic_and_ollama_flag_truncation():
    rec = Recorder([(200, {"content": [{"type": "text", "text": "a"}], "stop_reason": "max_tokens"})])
    assert providers.chat_provider(settings(rag_chat_provider="anthropic"),
                                   transport=rec.transport()).complete("s", "p").truncated
    rec = Recorder([(200, {"message": {"content": "a"}, "done_reason": "length"})])
    assert providers.chat_provider(Settings(database_url="postgresql://x/y", rag_chat_provider="ollama"),
                                   transport=rec.transport()).complete("s", "p").truncated


def test_gemini_chat_without_candidates_is_an_error():
    rec = Recorder([(200, {"promptFeedback": {"blockReason": "SAFETY"}})])
    chat = providers.chat_provider(settings(rag_chat_provider="gemini"), transport=rec.transport())
    with pytest.raises(providers.ProviderError, match="no candidate"):
        chat.complete("sys", "prompt")


def test_openai_chat_parses_message():
    rec = Recorder([(200, {"choices": [{"message": {"content": " hi "}}],
                           "usage": {"prompt_tokens": 5, "completion_tokens": 1}})])
    chat = providers.chat_provider(settings(rag_chat_provider="openai"), transport=rec.transport())
    reply = chat.complete("sys", "prompt")
    assert (reply.text, reply.prompt_tokens) == ("hi", 5)


def test_anthropic_chat_sends_version_and_joins_text_blocks():
    rec = Recorder([(200, {"content": [{"type": "text", "text": "a"}, {"type": "thinking", "text": "x"},
                                       {"type": "text", "text": "b"}],
                           "usage": {"input_tokens": 3, "output_tokens": 4}})])
    chat = providers.chat_provider(settings(rag_chat_provider="anthropic"), transport=rec.transport())
    assert chat.complete("sys", "p").text == "ab"
    assert rec.last.headers["anthropic-version"] == "2023-06-01"
    assert rec.last.headers["x-api-key"] == "k"


def test_ollama_chat_disables_streaming():
    rec = Recorder([(200, {"message": {"content": "ok"}, "prompt_eval_count": 7, "eval_count": 2})])
    chat = providers.chat_provider(
        Settings(database_url="postgresql://x/y", rag_chat_provider="ollama"), transport=rec.transport())
    assert chat.complete("sys", "p").text == "ok"
    assert '"stream":false' in rec.last.read().decode()


# ---------------------------------------------------------------- transport behaviour
def test_retryable_status_is_retried_then_succeeds():
    rec = Recorder([(503, {}), (429, {}), (200, {"data": [{"index": 0, "embedding": [1.0]}]})])
    provider = providers.embedding_provider(settings(rag_embed_provider="openai"), transport=rec.transport())
    provider.api.sleep = lambda _: None
    assert provider.embed(["a"]) == [[1.0]]
    assert len(rec.requests) == 3


def test_client_error_fails_fast_without_retrying():
    rec = Recorder([(401, {"error": {"message": "bad key"}})])
    provider = providers.embedding_provider(settings(rag_embed_provider="openai"), transport=rec.transport())
    with pytest.raises(providers.ProviderError, match="HTTP 401"):
        provider.embed(["a"])
    assert len(rec.requests) == 1


def test_gives_up_after_max_attempts():
    rec = Recorder([(503, {})] * 3)
    provider = providers.embedding_provider(settings(rag_embed_provider="openai", rag_max_attempts=3),
                                            transport=rec.transport())
    provider.api.sleep = lambda _: None
    with pytest.raises(providers.ProviderError, match="giving up after 3 attempts"):
        provider.embed(["a"])
    assert len(rec.requests) == 3


def test_the_error_carries_the_last_http_status():
    rec = Recorder([(429, {"error": {"code": 429}})] * 2 + [(403, {})])
    provider = providers.embedding_provider(settings(rag_embed_provider="openai", rag_max_attempts=2),
                                            transport=rec.transport())
    provider.api.sleep = lambda _: None
    with pytest.raises(providers.ProviderError) as quota:
        provider.embed(["a"])
    assert quota.value.status == 429
    with pytest.raises(providers.ProviderError) as refused:
        provider.embed(["a"])
    assert refused.value.status == 403


def test_retry_after_header_is_honoured():
    slept: list[float] = []

    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "7"}, json={})

    api = providers.JsonApi(base_url="https://example.invalid", max_attempts=2,
                            transport=httpx.MockTransport(handler), sleep=slept.append)
    with pytest.raises(providers.ProviderError):
        api.post("/x", {}, context="t")
    assert slept and slept[0] >= 7


# ---------------------------------------------------------------- credentials
def test_missing_key_names_the_variables_to_set(monkeypatch):
    # Hermetic on purpose: the crawler container is started with .env loaded, so a real key may be
    # sitting in the environment and would otherwise satisfy the lookup under test.
    for name in ("RAG_EMBED_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(providers.ProviderError, match="RAG_EMBED_API_KEY"):
        providers.embedding_provider(Settings(database_url="postgresql://x/y", rag_embed_provider="gemini",
                                              rag_embed_api_key=None))


def test_key_falls_back_to_provider_env_var(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert providers.resolve_api_key("gemini", None) == "from-env"
    assert providers.resolve_api_key("gemini", "explicit") == "explicit"
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "second-choice")
    assert providers.resolve_api_key("gemini", None) == "second-choice"


def test_ollama_needs_no_credential():
    assert providers.auth_headers("ollama", None, what="EMBED") == {}


def test_base_url_override_wins():
    assert providers.base_url_for("openai", "http://localhost:1234/v1/") == "http://localhost:1234/v1"
    assert providers.base_url_for("openai", None) == providers.DEFAULT_BASE_URL["openai"]


def test_api_key_never_appears_in_a_log_record(caplog):
    rec = Recorder([(503, {}), (200, {"data": [{"index": 0, "embedding": [1.0]}]})])
    provider = providers.embedding_provider(settings(rag_embed_provider="openai", rag_embed_api_key="s3cret"),
                                            transport=rec.transport())
    provider.api.sleep = lambda _: None
    with caplog.at_level("DEBUG"):
        provider.embed(["a"])
    assert "s3cret" not in caplog.text


# ---------------------------------------------------------------- Vertex AI
def vertex_settings(**kwargs) -> Settings:
    return Settings(database_url="postgresql://x/y", rag_embed_provider="vertex", rag_chat_provider="vertex",
                    rag_vertex_project="p1", rag_embed_model="gemini-embedding-001",
                    rag_chat_model="gemini-2.5-flash", **kwargs)


@pytest.fixture
def fake_adc(monkeypatch):
    """Skip the real ADC lookup: this asserts on the request we send, not on Google's auth flow."""
    monkeypatch.setattr(providers, "google_token_provider", lambda *a, **k: lambda: "tok-123")


def test_vertex_base_url_is_regional_by_default():
    assert providers.vertex_base_url("p1", "us-central1") == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p1/locations/us-central1/publishers/google")


def test_vertex_base_url_has_a_global_form():
    assert providers.vertex_base_url("p1", "global") == (
        "https://aiplatform.googleapis.com/v1/projects/p1/locations/global/publishers/google")


def test_vertex_without_a_project_says_what_to_set():
    with pytest.raises(providers.ProviderError, match="RAG_VERTEX_PROJECT"):
        providers.vertex_base_url("", "us-central1")


def test_vertex_embed_uses_the_prediction_envelope(fake_adc):
    rec = Recorder([(200, {"predictions": [{"embeddings": {"values": [0.1, 0.2]}},
                                           {"embeddings": {"values": [0.3, 0.4]}}]})])
    provider = providers.embedding_provider(vertex_settings(rag_embed_dim=1536), transport=rec.transport())
    assert provider.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    body = rec.last.read().decode()
    assert str(rec.last.url).endswith("/models/gemini-embedding-001:predict")
    assert '"instances"' in body and '"task_type":"RETRIEVAL_DOCUMENT"' in body
    assert '"outputDimensionality":1536' in body


def test_vertex_embed_keeps_the_token_count_and_truncation_it_reports(fake_adc):
    # response shape seen from Vertex AI on 2026-10-04 (gemini-embedding-001, :predict)
    rec = Recorder([(200, {"predictions": [
        {"embeddings": {"statistics": {"truncated": False, "token_count": 3}, "values": [0.1]}},
        {"embeddings": {"statistics": {"truncated": True, "token_count": 2048}, "values": [0.2]}}],
        "metadata": {"billableCharacterCount": 77}})])
    provider = providers.embedding_provider(vertex_settings(), transport=rec.transport())
    provider.embed(["a", "b"])
    assert provider.last_usage == {"tokens": 2051, "truncated": 1}


def test_vertex_embed_marks_a_query(fake_adc):
    rec = Recorder([(200, {"predictions": [{"embeddings": {"values": [1.0]}}]})])
    provider = providers.embedding_provider(vertex_settings(), transport=rec.transport())
    provider.embed(["q"], query=True)
    assert '"task_type":"RETRIEVAL_QUERY"' in rec.last.read().decode()


def test_vertex_sends_a_bearer_token_and_no_api_key(fake_adc):
    rec = Recorder([(200, {"predictions": [{"embeddings": {"values": [1.0]}}]})])
    provider = providers.embedding_provider(vertex_settings(), transport=rec.transport())
    provider.embed(["a"])
    assert rec.last.headers["authorization"] == "Bearer tok-123"
    assert "x-goog-api-key" not in rec.last.headers


def test_vertex_short_prediction_batch_is_an_error(fake_adc):
    rec = Recorder([(200, {"predictions": [{"embeddings": {"values": [1.0]}}]})])
    provider = providers.embedding_provider(vertex_settings(), transport=rec.transport())
    with pytest.raises(providers.ProviderError, match="asked for 2 vectors"):
        provider.embed(["a", "b"])


def test_vertex_chat_reuses_the_gemini_body_on_the_vertex_url(fake_adc):
    rec = Recorder([(200, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                           "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1}})])
    chat = providers.chat_provider(vertex_settings(), transport=rec.transport())
    assert chat.complete("sys", "p").text == "ok"
    assert str(rec.last.url).endswith("/models/gemini-2.5-flash:generateContent")
    assert '"system_instruction"' in rec.last.read().decode()
    assert rec.last.headers["authorization"] == "Bearer tok-123"


def test_vertex_token_is_resolved_per_request_not_cached(monkeypatch):
    tokens = iter(["t1", "t2"])
    monkeypatch.setattr(providers, "google_token_provider", lambda *a, **k: lambda: next(tokens))
    rec = Recorder([(200, {"predictions": [{"embeddings": {"values": [1.0]}}]}),
                    (200, {"predictions": [{"embeddings": {"values": [1.0]}}]})])
    provider = providers.embedding_provider(vertex_settings(), transport=rec.transport())
    provider.embed(["a"])
    provider.embed(["b"])
    assert [r.headers["authorization"] for r in rec.requests] == ["Bearer t1", "Bearer t2"]


def test_httpx_auth_transport_exposes_what_google_auth_reads():
    def handler(request):
        return httpx.Response(200, json={"access_token": "x"})

    # google-auth reads .status / .data off the object the transport returns.
    transport = providers._HttpxAuthTransport()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkey = httpx.request
        try:
            httpx.request = lambda method, url, **kw: client.request(method, url, **kw)
            response = transport("https://oauth2.example/token", method="POST", body=b"x")
        finally:
            httpx.request = monkey
    assert response.status == 200
    assert b"access_token" in response.data


# ---------------------------------------------------------------- streaming (SSE)
# The real shape, measured against Vertex AI on 2026-10-06: `:streamGenerateContent?alt=sse` answers
# 200 text/event-stream, every event carries candidates[0].content.parts[*].text, and the usage counts
# arrive only in the LAST event.
def sse_body(*events: str) -> bytes:
    return "".join(f"data: {e}\n\n" for e in events).encode()


def streaming(body: bytes, *, status: int = 200, provider="gemini", **kw):
    """A chat provider whose transport replays one SSE response.

    Driven as "gemini" because VertexChat only changes the host and the credential - ``stream`` itself
    is GeminiChat's, which is the code under test here."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})

    chat = providers.chat_provider(settings(rag_chat_provider=provider, **kw),
                                   transport=httpx.MockTransport(handler))
    return chat, seen


def test_the_deltas_arrive_in_order_and_join_into_the_answer():
    chat, seen = streaming(sse_body(
        '{"candidates":[{"content":{"parts":[{"text":"Pfizer recalled "}]}}]}',
        '{"candidates":[{"content":{"parts":[{"text":"D-0853-2026"}]}}]}',
        '{"candidates":[{"content":{"parts":[{"text":" [1]."}]},"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":3797,"candidatesTokenCount":700}}'))
    pieces: list[str] = []

    reply = chat.stream("system", "prompt", on_delta=pieces.append)

    assert pieces == ["Pfizer recalled ", "D-0853-2026", " [1]."]
    assert reply.text == "Pfizer recalled D-0853-2026 [1]."
    assert "streamGenerateContent" in str(seen["request"].url)
    assert "alt=sse" in str(seen["request"].url)


def test_the_token_counts_come_from_the_last_event():
    """Earlier events carry no counts at all, so MLflow's cost would be zero if we read the first."""
    chat, _ = streaming(sse_body(
        '{"candidates":[{"content":{"parts":[{"text":"a"}]}}],"usageMetadata":{"trafficType":"ON_DEMAND"}}',
        '{"candidates":[{"content":{"parts":[{"text":"b"}]},"finishReason":"STOP"}],'
        '"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":7,"thoughtsTokenCount":2}}'))

    reply = chat.stream("system", "prompt", on_delta=lambda _: None)

    assert reply.prompt_tokens == 10
    assert reply.output_tokens == 9           # visible + thoughts, as in complete()
    assert reply.reasoning_tokens == 2


def test_a_thought_part_is_not_streamed_as_answer_text():
    chat, _ = streaming(sse_body(
        '{"candidates":[{"content":{"parts":[{"text":"thinking...","thought":true}]}}]}',
        '{"candidates":[{"content":{"parts":[{"text":"The answer."}]},"finishReason":"STOP"}]}'))
    pieces: list[str] = []

    reply = chat.stream("system", "prompt", on_delta=pieces.append)

    assert pieces == ["The answer."]
    assert reply.text == "The answer."


def test_a_stream_cut_off_at_the_token_limit_is_flagged():
    chat, _ = streaming(sse_body(
        '{"candidates":[{"content":{"parts":[{"text":"half an ans"}]},"finishReason":"MAX_TOKENS"}]}'))

    assert chat.stream("system", "p", on_delta=lambda _: None).truncated


def test_blank_lines_and_a_done_marker_are_skipped():
    chat, _ = streaming(b'\n\ndata: {"candidates":[{"content":{"parts":[{"text":"x"}]}}]}\n\ndata: [DONE]\n\n')
    pieces: list[str] = []

    assert chat.stream("system", "p", on_delta=pieces.append).text == "x"
    assert pieces == ["x"]


def test_a_refused_stream_raises_without_leaking_the_url():
    chat, _ = streaming(b'{"error": "nope"}', status=403)

    with pytest.raises(providers.ProviderError) as exc:
        chat.stream("system", "p", on_delta=lambda _: None)
    assert exc.value.status == 403
    assert "googleapis.com" not in str(exc.value)


def test_broken_json_in_the_stream_is_an_error_not_a_silent_truncation():
    chat, _ = streaming(b'data: {"candidates":[{"content":{"parts":[{"text":"a"}]}}]}\n\ndata: {oops\n\n')
    pieces: list[str] = []

    with pytest.raises(providers.ProviderError):
        chat.stream("system", "p", on_delta=pieces.append)
    assert pieces == ["a"]          # what already arrived was already handed over


def test_a_stream_is_not_retried():
    """post() retries because a failed attempt produced nothing; a broken stream has already emitted
    text, so retrying would repeat it."""
    attempts = []

    def handler(request):
        attempts.append(request)
        return httpx.Response(503, content=b"busy")

    chat = providers.chat_provider(settings(rag_chat_provider="gemini"),
                                   transport=httpx.MockTransport(handler))
    with pytest.raises(providers.ProviderError):
        chat.stream("system", "p", on_delta=lambda _: None)
    assert len(attempts) == 1


def test_vertex_streams_from_its_own_publisher_path(fake_adc):
    """The same method, the host and credential of Vertex. This is the combination measured live."""
    chat, seen = streaming(sse_body('{"candidates":[{"content":{"parts":[{"text":"x"}]}}]}'),
                           provider="vertex", rag_vertex_project="p1")

    assert chat.stream("system", "p", on_delta=lambda _: None).text == "x"
    url = str(seen["request"].url)
    assert "aiplatform.googleapis.com" in url and "publishers/google" in url
    assert ":streamGenerateContent" in url and "alt=sse" in url


# ---------------------------------------------------------------- the fallback
class OnlyComplete:
    provider, model, temperature, max_tokens = "fake", "fake-chat", 0.1, 1024

    def complete(self, system, prompt, *, reasoning=True):
        return providers.ChatReply("one piece", 10, 5)


def test_a_provider_without_streaming_hands_the_answer_over_in_one_piece():
    """ollama and the OpenAI-compatible endpoints have no stream(); the page still works."""
    pieces: list[str] = []

    reply = providers.complete_streaming(OnlyComplete(), "s", "p", pieces.append)

    assert pieces == ["one piece"]
    assert reply.text == "one piece"


def test_the_fallback_emits_nothing_when_the_model_returned_nothing():
    class Empty(OnlyComplete):
        def complete(self, system, prompt, *, reasoning=True):
            return providers.ChatReply("", 10, 0)

    pieces: list[str] = []
    assert providers.complete_streaming(Empty(), "s", "p", pieces.append).text == ""
    assert pieces == []


def test_streaming_is_used_when_the_provider_has_it():
    chat, _ = streaming(sse_body('{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}'))
    pieces: list[str] = []

    providers.complete_streaming(chat, "s", "p", pieces.append)

    assert pieces == ["hi"]
