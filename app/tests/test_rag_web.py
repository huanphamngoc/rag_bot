"""Web chat (crawlerrag.rag.web): routes, input validation, the daily cap, and what reaches the browser.

A fake connection answers the statements the routes issue; ``answer.ask`` itself is covered by
test_rag_conversation.py and is replaced here, so no model or database is involved.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flask")

from crawlerrag.rag import answer, index, retrieve, web  # noqa: E402

STATIC = Path(web.__file__).parent / "static"


class FakeConn:
    def __init__(self, *, used=0, counts=None, state=True):
        self.used = used
        self.counts = {"drug_recall": 17937, "cpsc_recall": 10002} if counts is None else counts
        self.state = state
        self.conversations: set[str] = set()
        self.log_rows: list[dict] = []
        self.sql: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append(sql)
        rows: list[dict] = []
        if "FROM rag.index_state" in sql:
            rows = [{"embed_provider": "vertex", "embed_model": "gemini-embedding-001", "embed_dim": 1536}] \
                if self.state else []
        elif "GROUP BY doc_type" in sql:
            rows = [{"doc_type": k, "n": v} for k, v in self.counts.items()]
        elif "date_trunc('day'" in sql:
            rows = [{"n": self.used}]
        elif "INSERT INTO rag.conversation" in sql:
            cid = str(uuid.uuid4())
            self.conversations.add(cid)
            rows = [{"conversation_id": cid}]
        elif "SELECT 1 FROM rag.conversation" in sql:
            rows = [{"?column?": 1}] if params[0] in self.conversations else []
        elif "FROM rag.query_log" in sql:
            rows = self.log_rows
        elif "FROM rag.document WHERE doc_id = ANY" in sql:
            rows = [{"doc_id": d, "doc_type": "drug_recall", "title": f"Recall {d}", "url": "https://example.gov/r"}
                    for d in params[0]]
        return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: rows[0] if rows else None)


def _settings(**over):
    base = dict(rag_web_daily_limit=300, rag_doc_types="drug_recall,cpsc_recall", rag_embed_provider="vertex",
                rag_embed_model="gemini-embedding-001", rag_chat_provider="vertex", rag_chat_model="gemini-2.5-flash",
                log_level="INFO", log_format="json")
    base.update(over)
    return SimpleNamespace(**base)


def _hit(n, url="https://www.accessdata.fda.gov/x", text="Cantrell Drug Company recalled syringes. " * 30):
    return retrieve.Hit(chunk_id=n, doc_id=f"drug_recall:D-{n:04d}-2017", doc_type="drug_recall",
                        source_id="openfda_enforcement", title=f"Recall {n}", url=url, text=text, metadata={},
                        score=0.03, vector_rank=n, text_rank=n)


@pytest.fixture
def made():
    """(app client, fake connection, calls) with answer.ask and the model factories replaced."""
    def build(conn=None, settings=None, reply=None, monkeypatch=None):
        conn = conn or FakeConn()
        calls = {"ask": [], "embed": 0, "chat": 0}

        def fake_ask(c, s, embedder, chat, question, **kw):
            calls["ask"].append((question, kw))
            return reply or answer.Answer(question=question, text="Cantrell recalled syringes [1][2].",
                                          hits=[_hit(1), _hit(2, url="javascript:alert(1)")], prompt_tokens=2796,
                                          output_tokens=632, duration_ms=4200, query_id=7,
                                          conversation_id=kw.get("conversation_id"), turn=1)

        def embed_factory(s):
            calls["embed"] += 1
            return object()

        def chat_factory(s):
            calls["chat"] += 1
            return object()

        monkeypatch.setattr(web.answer_mod, "ask", fake_ask)
        app = web.create_app(settings or _settings(), connect=lambda: conn,
                             embedder_factory=embed_factory, chat_factory=chat_factory)
        return app.test_client(), conn, calls
    return build


def test_page_and_assets_are_served_with_a_strict_policy(made, monkeypatch):
    client, _, _ = made(monkeypatch=monkeypatch)
    page = client.get("/")
    assert page.status_code == 200 and b"Recall Chatbot" in page.data
    csp = page.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/healthz").get_json() == {"status": "ok"}


def test_page_never_parses_model_or_data_text_as_html():
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in script, sink
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in page and "style=" not in page      # nothing inline: the CSP forbids it


def test_info_lists_indexed_types_defaults_and_usage(made, monkeypatch):
    conn = FakeConn(used=12, counts={"cpsc_recall": 10002, "drug_recall": 17937, "insurance_plan": 5})
    client, _, _ = made(conn=conn, monkeypatch=monkeypatch)
    data = client.get("/api/info").get_json()
    assert data["ready"] is True
    assert [(d["name"], d["default"]) for d in data["doc_types"]] == \
        [("cpsc_recall", True), ("drug_recall", True), ("insurance_plan", False)]
    assert data["doc_types"][0]["label"] == "CPSC consumer product recalls"
    assert (data["used_today"], data["daily_limit"]) == (12, 300)
    assert data["chat_model"] == "vertex/gemini-2.5-flash"


def test_info_reports_an_empty_index(made, monkeypatch):
    client, _, _ = made(conn=FakeConn(counts={}, state=False), monkeypatch=monkeypatch)
    assert client.get("/api/info").get_json()["ready"] is False


@pytest.mark.parametrize("body, message", [
    ([], "JSON object"),
    ({"question": "   "}, "empty"),
    ({"question": "x" * 2001}, "longer than"),
    ({"question": "q", "doc_types": ["food_product"]}, "not indexed"),
    ({"question": "q", "doc_types": "drug_recall"}, "list"),
    ({"question": "q", "filters": {str(i): i for i in range(6)}}, "at most"),
    ({"question": "q", "filters": {"year": {"$gt": 1}}}, "bad filter"),
    ({"question": "q", "conversation_id": "not-a-known-one"}, "no such conversation"),
])
def test_ask_rejects_bad_input_before_any_model_call(made, monkeypatch, body, message):
    client, _, calls = made(monkeypatch=monkeypatch)
    response = client.post("/api/ask", json=body)
    assert response.status_code == 400
    assert message in response.get_json()["error"]
    assert calls["ask"] == [] and calls["embed"] == 0


def test_ask_answers_with_sources_and_drops_unsafe_links(made, monkeypatch):
    client, conn, calls = made(monkeypatch=monkeypatch)
    cid = client.post("/api/conversations").get_json()["conversation_id"]
    response = client.post("/api/ask", json={"question": "Why did Cantrell recall?", "conversation_id": cid,
                                             "doc_types": ["drug_recall"], "filters": {"year": 2017}})
    assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
    data = response.get_json()
    assert data["answer"] == "Cantrell recalled syringes [1][2]." and data["error"] is None
    assert [s["n"] for s in data["sources"]] == [1, 2]
    assert data["sources"][0]["url"].startswith("https://")
    assert data["sources"][1]["url"] is None                      # javascript: never becomes a link
    assert data["sources"][0]["matched_by"] == "both"
    assert len(data["sources"][0]["snippet"]) <= web.SNIPPET_CHARS + 4
    question, kw = calls["ask"][0]
    assert question == "Why did Cantrell recall?"
    assert kw == {"doc_types": ["drug_recall"], "filters": {"year": 2017}, "conversation_id": cid,
                  "sink": None}      # the JSON endpoint shares one_turn() with the streaming one


def test_models_are_created_once_per_process(made, monkeypatch):
    client, _, calls = made(monkeypatch=monkeypatch)
    for _ in range(3):
        assert client.post("/api/ask", json={"question": "q"}).status_code == 200
    assert (calls["embed"], calls["chat"], len(calls["ask"])) == (1, 1, 3)


def test_daily_cap_stops_paid_calls(made, monkeypatch):
    client, _, calls = made(conn=FakeConn(used=300), monkeypatch=monkeypatch)
    response = client.post("/api/ask", json={"question": "q"})
    assert response.status_code == 429 and "00:00 UTC" in response.get_json()["error"]
    assert calls["ask"] == [] and calls["embed"] == 0


def test_cap_of_zero_means_no_cap(made, monkeypatch):
    client, conn, _ = made(conn=FakeConn(used=10_000), settings=_settings(rag_web_daily_limit=0),
                           monkeypatch=monkeypatch)
    assert client.post("/api/ask", json={"question": "q"}).status_code == 200
    assert not any("date_trunc" in s for s in conn.sql)


def test_provider_error_text_stays_on_the_server(made, monkeypatch):
    secretish = "HTTP 403 from https://us-central1-aiplatform.googleapis.com/v1/projects/p: PERMISSION_DENIED"
    failed = answer.Answer(question="q", text="", error=secretish, query_id=9)
    client, _, _ = made(reply=failed, monkeypatch=monkeypatch)
    response = client.post("/api/ask", json={"question": "q"})
    assert response.status_code == 200
    assert response.get_json()["error"] == web.MODEL_FAILED
    assert b"PERMISSION_DENIED" not in response.data


def test_index_problems_and_a_down_database_are_503(made, monkeypatch):
    import psycopg

    client, _, _ = made(monkeypatch=monkeypatch)

    def no_index(*a, **kw):
        raise index.IndexStateError("no index yet")
    monkeypatch.setattr(web.answer_mod, "ask", no_index)
    assert client.post("/api/ask", json={"question": "q"}).status_code == 503

    def down():
        raise psycopg.OperationalError("connection refused")
    app = web.create_app(_settings(), connect=down, embedder_factory=lambda s: None, chat_factory=lambda s: None)
    response = app.test_client().get("/api/info")
    assert response.status_code == 503 and "connection refused" not in response.get_json()["error"]


def test_conversation_history_is_rebuilt_from_the_query_log(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)
    assert client.get(f"/api/conversations/{uuid.uuid4()}").status_code == 404
    assert client.get("/api/conversations/not-a-uuid").status_code == 404
    cid = client.post("/api/conversations").get_json()["conversation_id"]
    conn.log_rows = [{"turn": 1, "question": "Why?", "answer": "Because [1].", "error": None, "query_id": 3,
                      "duration_ms": 3000, "prompt_tokens": 10, "output_tokens": 5, "standalone_question": None,
                      "retrieved": [{"rank": 1, "chunk_id": 4, "doc_id": "drug_recall:D-1", "score": 0.03,
                                     "matched_by": "both"}]},
                     {"turn": 2, "question": "And?", "answer": None, "error": "HTTP 500 raw provider body",
                      "query_id": 4, "duration_ms": 10, "prompt_tokens": None, "output_tokens": None,
                      "standalone_question": "And why?", "retrieved": []}]
    data = client.get(f"/api/conversations/{cid}").get_json()
    assert [t["turn"] for t in data["turns"]] == [1, 2]
    assert data["turns"][0]["sources"][0]["title"] == "Recall drug_recall:D-1"
    assert data["turns"][1]["error"] == web.MODEL_FAILED


# ---------------------------------------------------------------- streaming (/api/ask/stream)
# The page posts the question and reads Server-Sent Events, so the answer appears as the model writes
# it and the conversation carries on in the same session. The graph pushes text into a sink while the
# response has to pull, so the turn runs in a worker thread with a queue between them.
def events(response):
    """Parse an SSE body into [(event, data), ...]."""
    import json as _json
    out = []
    for block in response.get_data(as_text=True).split("\n\n"):
        if not block.strip():
            continue
        name, payload = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload.append(line[5:].strip())
        out.append((name, _json.loads("\n".join(payload))))
    return out


def streaming_ask(pieces, **answer_kw):
    """A fake answer.ask that feeds `pieces` to the sink, like the real streaming provider does."""
    def fake_ask(c, s, embedder, chat, question, *, sink=None, **kw):
        for piece in pieces:
            if sink:
                sink(piece)
        return answer.Answer(question=question, text="".join(pieces),
                             hits=[_hit(1)], prompt_tokens=100, output_tokens=20, duration_ms=1234,
                             query_id=11, conversation_id=kw.get("conversation_id"), turn=2,
                             **answer_kw)
    return fake_ask


def test_the_stream_sends_the_answer_in_pieces_then_one_done(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)
    monkeypatch.setattr(web.answer_mod, "ask", streaming_ask(["Cantrell ", "recalled ", "syringes [1]."]))

    response = client.post("/api/ask/stream", json={"question": "what was recalled?"})

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/event-stream")
    got = events(response)
    assert [d["text"] for name, d in got if name == "delta"] == ["Cantrell ", "recalled ", "syringes [1]."]
    assert [name for name, _ in got][-1] == "done"


def test_the_done_event_carries_what_the_json_endpoint_returns(made, monkeypatch):
    """The page rebuilds the finished bubble from it, so it needs the sources and the trace link."""
    client, conn, _ = made(monkeypatch=monkeypatch)
    monkeypatch.setattr(web.answer_mod, "ask", streaming_ask(["answer"]))

    _, done = events(client.post("/api/ask/stream", json={"question": "what was recalled?"}))[-1]

    assert done["answer"] == "answer"
    assert done["query_id"] == 11 and done["turn"] == 2
    assert [s["doc_id"] for s in done["sources"]] == ["drug_recall:D-0001-2017"]
    assert set(done) == {"question", "answer", "error", "truncated", "query_id", "conversation_id",
                         "turn", "standalone_question", "duration_ms", "prompt_tokens",
                         "output_tokens", "sources", "trace_url",
                         # empty unless the turn is waiting for a clearer question
                         "pending_question", "samples", "thread_id"}
    assert done["pending_question"] is None and done["samples"] == [] and done["thread_id"] is None


def test_a_blocked_question_streams_no_text_and_still_explains_itself(made, monkeypatch):
    """The gate answers for free, so there is nothing to stream - the whole reply is in `done`."""
    client, conn, _ = made(monkeypatch=monkeypatch)

    def blocked(c, s, embedder, chat, question, *, sink=None, **kw):
        return answer.Answer(question=question, text="That is a SQL question.", query_id=12,
                             qualify_decision="needs_sql", qualify_rule="aggregate")

    monkeypatch.setattr(web.answer_mod, "ask", blocked)

    got = events(client.post("/api/ask/stream", json={"question": "how many recalls in 2026?"}))

    assert [name for name, _ in got] == ["done"]
    assert got[0][1]["answer"] == "That is a SQL question."


def test_the_stream_keeps_the_conversation_so_the_next_question_follows_on(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)
    monkeypatch.setattr(web.answer_mod, "ask", streaming_ask(["ok"]))
    cid = client.post("/api/conversations", json={}).get_json()["conversation_id"]

    _, done = events(client.post("/api/ask/stream",
                                 json={"question": "and the second one?", "conversation_id": cid}))[-1]

    assert done["conversation_id"] == cid


def test_bad_input_is_refused_before_a_worker_is_started(made, monkeypatch):
    client, conn, calls = made(monkeypatch=monkeypatch)

    response = client.post("/api/ask/stream", json={"question": "  "})

    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert calls["ask"] == []


def test_an_unknown_conversation_is_refused_before_a_worker_is_started(made, monkeypatch):
    client, conn, calls = made(monkeypatch=monkeypatch)

    response = client.post("/api/ask/stream",
                           json={"question": "what was recalled?", "conversation_id": str(uuid.uuid4())})

    assert response.status_code == 400
    assert calls["ask"] == []


def test_the_daily_cap_stops_the_stream_too(made, monkeypatch):
    client, conn, calls = made(conn=FakeConn(used=300), monkeypatch=monkeypatch)

    response = client.post("/api/ask/stream", json={"question": "what was recalled?"})

    assert response.status_code == 429
    assert calls["ask"] == []


def test_a_model_failure_mid_stream_becomes_an_error_event_without_the_provider_text(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)

    def boom(c, s, embedder, chat, question, *, sink=None, **kw):
        if sink:
            sink("the beginning of an ")
        raise retrieve.ProviderError("HTTP 500 from https://aiplatform.googleapis.com secret-key")

    monkeypatch.setattr(web.answer_mod, "ask", boom)

    got = events(client.post("/api/ask/stream", json={"question": "what was recalled?"}))

    assert [name for name, _ in got] == ["delta", "error"]
    assert got[-1][1]["error"] == web.MODEL_FAILED
    assert "googleapis.com" not in got[-1][1]["error"]


def test_the_stream_is_never_cached(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)
    monkeypatch.setattr(web.answer_mod, "ask", streaming_ask(["x"]))

    response = client.post("/api/ask/stream", json={"question": "what was recalled?"})

    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Accel-Buffering"] == "no"


# ---------------------------------------------------------------- the clarify round over HTTP
# A paused turn is its own SSE event: there is no answer to render, and the page needs the examples and
# the thread so the correction goes back to the turn that is waiting instead of starting a new one.
THREAD = "a" * 32


def waiting_ask(samples=(("Why did Eli Lilly & Company recall a drug?", "FDA drug recall D-0445-2024"),)):
    def fake_ask(c, s, embedder, chat, question, *, sink=None, **kw):
        return answer.Answer(question=question, text="", query_id=31,
                             qualify_decision="clarify", qualify_rule="limit:min_words",
                             pending_question="That is a subject, not a question yet.",
                             samples=[{"question": q, "title": t} for q, t in samples],
                             thread_id=THREAD)
    return fake_ask


def test_a_vague_question_streams_a_question_event_instead_of_an_answer(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)
    monkeypatch.setattr(web.answer_mod, "ask", waiting_ask())

    got = events(client.post("/api/ask/stream", json={"question": "insulin"}))

    assert [name for name, _ in got] == ["question"]
    data = got[0][1]
    assert data["thread_id"] == THREAD
    assert [s["question"] for s in data["samples"]] == ["Why did Eli Lilly & Company recall a drug?"]
    assert data["samples"][0]["about"] == "FDA drug recall D-0445-2024"
    assert data["answer"] == ""


def test_the_json_endpoint_reports_the_pause_too(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)
    monkeypatch.setattr(web.answer_mod, "ask", waiting_ask())

    data = client.post("/api/ask", json={"question": "insulin"}).get_json()

    assert data["pending_question"] and data["thread_id"] == THREAD


def test_a_correction_is_sent_to_the_waiting_turn_not_asked_fresh(made, monkeypatch):
    client, conn, calls = made(monkeypatch=monkeypatch)
    resumed: list = []

    def fake_resume(c, s, embedder, chat, *, thread_id, question, sink=None, **kw):
        resumed.append((thread_id, question))
        return answer.Answer(question=question, text="Because of sterility [1].", hits=[_hit(1)],
                             query_id=32, qualify_decision="pass")

    monkeypatch.setattr(web.answer_mod, "resume", fake_resume)

    data = client.post("/api/ask", json={"question": "Why did Eli Lilly recall a drug?",
                                        "thread_id": THREAD}).get_json()

    assert resumed == [(THREAD, "Why did Eli Lilly recall a drug?")]
    assert calls["ask"] == []                      # the paused turn finished; no new one was started
    assert data["answer"].startswith("Because of sterility")


def test_a_correction_can_be_streamed_as_well(made, monkeypatch):
    client, conn, _ = made(monkeypatch=monkeypatch)

    def fake_resume(c, s, embedder, chat, *, thread_id, question, sink=None, **kw):
        if sink:
            sink("Because of ")
            sink("sterility [1].")
        return answer.Answer(question=question, text="Because of sterility [1].", hits=[_hit(1)],
                             query_id=33, qualify_decision="pass")

    monkeypatch.setattr(web.answer_mod, "resume", fake_resume)

    got = events(client.post("/api/ask/stream", json={"question": "Why did Eli Lilly recall?",
                                                      "thread_id": THREAD}))

    assert [name for name, _ in got] == ["delta", "delta", "done"]


@pytest.mark.parametrize("thread", ["not-hex", "a" * 31, "A" * 32, 1234])
def test_a_malformed_thread_is_refused(made, monkeypatch, thread):
    client, conn, calls = made(monkeypatch=monkeypatch)

    response = client.post("/api/ask", json={"question": "Why did Eli Lilly recall?", "thread_id": thread})

    assert response.status_code == 400
    assert calls["ask"] == []


def test_a_thread_that_is_no_longer_waiting_is_a_conflict(made, monkeypatch):
    """The page was reloaded, or the pause is long gone: say so instead of a 500."""
    from crawlerrag.rag import graph as chat_graph
    client, conn, _ = made(monkeypatch=monkeypatch)

    def gone(c, s, embedder, chat, *, thread_id, question, sink=None, **kw):
        raise chat_graph.NoSuchTurn(f"no paused turn for thread {thread_id!r}")

    monkeypatch.setattr(web.answer_mod, "resume", gone)

    response = client.post("/api/ask", json={"question": "anything", "thread_id": THREAD})

    assert response.status_code == 409
    assert THREAD not in response.get_json()["error"]


def test_the_daily_cap_applies_to_a_correction_too(made, monkeypatch):
    client, conn, _ = made(conn=FakeConn(used=300), monkeypatch=monkeypatch)
    response = client.post("/api/ask", json={"question": "Why did Eli Lilly recall?", "thread_id": THREAD})
    assert response.status_code == 429

# ---------------------------------------------------------------- which sources are shown
# Retrieval always hands the model RAG_TOP_K excerpts and the answer usually leans on one. Measured
# over 14 real answers in rag.query_log: 8 retrieved every time, and the answer cited 1 of them seven
# times, 0 three times, 2 once, 6 once, 8 twice. So the page shows what was cited and folds the rest
# away - it never drops them, because the retrieved set is how a wrong answer is explained.
@pytest.mark.parametrize("text, expected", [
    ("Because of a fire hazard [1].", {1}),
    ("Two firms were involved [1, 3].", {1, 3}),
    ("Both say so [2][4].", {2, 4}),
    ("Spaces are allowed [1 , 2].", {1, 2}),
    ("No citation at all.", set()),
    ("", set()),
    (None, set()),
])
def test_cited_reads_the_numbers_out_of_an_answer(text, expected):
    assert answer.cited(text) == expected


@pytest.mark.parametrize("text, count, expected", [
    ("Only the first [1].", 4, {1}),
    ("The first and the third [1][3].", 4, {1, 3}),
    # An answer that cites nothing still rests on what was retrieved; an empty list beside it would
    # hide the only evidence there is. 3 of the 14 measured answers were like this.
    ("No citations here.", 4, {1, 2, 3, 4}),
    (None, 3, {1, 2, 3}),
    # The model occasionally invents a number past the end. It marks nothing on its own, so the answer
    # falls back to all of them rather than to an empty list.
    ("As source [9] says.", 4, {1, 2, 3, 4}),
    ("Sources [2] and [9].", 4, {2}),
])
def test_mark_cited_decides_what_the_page_shows(text, count, expected):
    assert web._mark_cited(text, count) == expected


def _four_hits_citing_one_and_three():
    return answer.Answer(question="Why was it recalled?", text="A fire hazard [1], and a fall [3].",
                         hits=[_hit(n) for n in (1, 2, 3, 4)], prompt_tokens=100, output_tokens=20,
                         duration_ms=900, query_id=11)


def test_the_payload_marks_only_the_cited_sources(made, monkeypatch):
    client, _, _ = made(reply=_four_hits_citing_one_and_three(), monkeypatch=monkeypatch)
    body = client.post("/api/ask", json={"question": "Why was it recalled?"}).get_json()

    assert [s["n"] for s in body["sources"] if s["cited"]] == [1, 3]
    assert [s["n"] for s in body["sources"] if not s["cited"]] == [2, 4]


def test_the_uncited_sources_are_still_sent(made, monkeypatch):
    """Folded away on the page, not dropped from the payload."""
    client, _, _ = made(reply=_four_hits_citing_one_and_three(), monkeypatch=monkeypatch)
    body = client.post("/api/ask", json={"question": "Why was it recalled?"}).get_json()

    assert [s["n"] for s in body["sources"]] == [1, 2, 3, 4]


def test_the_page_folds_the_uncited_ones_away_and_keeps_their_numbers():
    """`value` on the <li> is what keeps "[3]" pointing at an item that still reads 3 once the two
    before it are hidden."""
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "s.cited !== false" in script and "s.cited === false" in script
    assert "value: s.n" in script
    assert "not cited in the answer" in script

