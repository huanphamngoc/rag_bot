"""Multi-turn RAG: history, follow-up rewriting, persistence and the `chat` loop.

A small stateful fake database answers exactly the statements the code issues, so a whole
conversation can be played turn by turn without Postgres or a model.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from crawlerrag import commands
from crawlerrag.rag import answer, conversation, retrieve
from crawlerrag.rag.providers import ChatReply, ProviderError
from tests.conftest import permissive_ruleset

RULES_DIR = str(Path(__file__).parents[1] / "rules")


class FakeDB:
    def __init__(self):
        self.conversations: dict[str, dict] = {}
        self.log: list[dict] = []
        self.searched: list[str] = []          # question text each lexical search ran with

    def execute(self, sql, params=None):
        rows: list[dict] = []
        if "INSERT INTO rag.conversation" in sql:
            cid = str(uuid.uuid4())
            self.conversations[cid] = {"title": None}
            rows = [{"conversation_id": cid}]
        elif "SELECT 1 FROM rag.conversation" in sql:
            rows = [{"?column?": 1}] if params[0] in self.conversations else []
        elif "UPDATE rag.conversation" in sql:
            title, cid = params
            self.conversations[cid]["title"] = self.conversations[cid]["title"] or title
        elif "max(turn)" in sql:
            turns = [r["turn"] for r in self.log if r["conversation_id"] == params[0]]
            rows = [{"turn": max(turns, default=0) + 1}]
        elif "SELECT turn, question, answer FROM rag.query_log" in sql:
            cid, limit = params
            done = [r for r in self.log if r["conversation_id"] == cid and r["answer"] and not r["error"]
                    and (r.get("qualify_decision") or "pass") == "pass"]
            rows = sorted(done, key=lambda r: -r["turn"])[:limit]
        elif "INSERT INTO rag.query_log" in sql:
            # ... trace_id, qualify_decision, qualify_rule are the last three parameters
            (question, _dt, _k, _ret, ans, *_mid, error, cid, turn, standalone,
             _trace, _decision, _rule) = params
            self.log.append({"query_id": len(self.log) + 1, "question": question, "answer": ans, "error": error,
                             "conversation_id": cid, "turn": turn, "standalone_question": standalone,
                             "qualify_decision": _decision})
            rows = [{"query_id": len(self.log)}]
        elif "<=>" in sql:
            rows = [_row(1, "drug_recall:D-0195-2017", "Cantrell Drug Company recalled hydromorphone syringes")]
        elif "ts_rank_cd" in sql:
            self.searched.append(params["q"])
            rows = [_row(1, "drug_recall:D-0195-2017", "Cantrell Drug Company recalled hydromorphone syringes")]
        return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: rows[0] if rows else None)


def _row(chunk_id, doc_id, text):
    return {"chunk_id": chunk_id, "doc_id": doc_id, "doc_type": "drug_recall", "source_id": "openfda",
            "title": doc_id, "url": None, "text": text, "metadata": {}, "distance": 0.1, "lexical_rank": 0.3}


class FakeEmbedder:
    provider, model = "vertex", "gemini-embedding-001"

    def embed(self, texts, *, query=False):
        return [[0.1, 0.2] for _ in texts]

    def close(self):
        pass


class FakeChat:
    """Answers the rewrite prompt and the answer prompt differently and records both."""
    provider, model, temperature, max_tokens = "vertex", "gemini-2.5-flash", 0.1, 1024

    def __init__(self, rewrite="Why did Cantrell Drug Company recall hydromorphone?", fail_rewrite=False):
        self.rewrite, self.fail_rewrite = rewrite, fail_rewrite
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, prompt, *, reasoning=True):
        self.calls.append((system, prompt))
        if system == conversation.CONDENSE_SYSTEM:
            if self.fail_rewrite:
                raise ProviderError("vertex chat: HTTP 503")
            return ChatReply(self.rewrite, 50, 12)
        return ChatReply(f"Answer {len(self.calls)} [1].", 900, 20)

    def close(self):
        pass


SETTINGS = SimpleNamespace(rag_top_k=4, rag_candidates=10, rules_dir=RULES_DIR)
PERMISSIVE = permissive_ruleset()


@pytest.fixture(autouse=True)
def no_index_state(monkeypatch):
    monkeypatch.setattr(retrieve, "require_state", lambda conn, provider=None: {})


def _ask(db, chat, question, cid):
    return answer.ask(db, SETTINGS, FakeEmbedder(), chat, question, conversation_id=cid,
                      ruleset=PERMISSIVE)


# ---------------------------------------------------------------- turns
def test_first_turn_needs_no_rewrite_and_starts_the_history():
    db, chat = FakeDB(), FakeChat()
    cid = conversation.create(db)
    first = _ask(db, chat, "Which drugs did Cantrell Drug Company recall?", cid)

    assert len(chat.calls) == 1                                  # answer only, no rewrite call
    assert "EARLIER QUESTIONS" not in chat.calls[0][1]
    assert db.searched == ["Which drugs did Cantrell Drug Company recall?"]
    assert (first.turn, first.standalone_question) == (1, None)
    assert db.log[0]["conversation_id"] == cid and db.log[0]["turn"] == 1
    assert db.conversations[cid]["title"] == "Which drugs did Cantrell Drug Company recall?"


def test_follow_up_is_rewritten_for_retrieval_and_answered_with_history():
    db, chat = FakeDB(), FakeChat()
    cid = conversation.create(db)
    _ask(db, chat, "Which drugs did Cantrell Drug Company recall?", cid)
    second = _ask(db, chat, "why?", cid)

    rewrite_system, rewrite_prompt = chat.calls[1]
    assert rewrite_system == conversation.CONDENSE_SYSTEM
    assert "User: Which drugs did Cantrell Drug Company recall?" in rewrite_prompt
    assert "LATEST MESSAGE\nwhy?" in rewrite_prompt

    # retrieval used the standalone question, not "why?"
    assert db.searched[-1] == "Why did Cantrell Drug Company recall hydromorphone?"

    answer_prompt = chat.calls[2][1]
    assert answer_prompt.startswith(
        "EARLIER QUESTIONS IN THIS CONVERSATION\n1. Which drugs did Cantrell Drug Company recall?")
    assert "Answer 1" not in answer_prompt          # the previous answer is not shown to the answering step
    assert "QUESTION\nwhy?\n(interpreted as: Why did Cantrell Drug Company recall hydromorphone?)" in answer_prompt
    assert second.turn == 2 and second.standalone_question.startswith("Why did Cantrell")
    assert db.log[1]["standalone_question"] == second.standalone_question
    assert db.log[1]["question"] == "why?"                       # the user's own words are what is logged


def test_failed_rewrite_falls_back_to_previous_question_plus_follow_up():
    db, chat = FakeDB(), FakeChat(fail_rewrite=True)
    cid = conversation.create(db)
    _ask(db, chat, "Which drugs did Cantrell Drug Company recall?", cid)
    second = _ask(db, chat, "why?", cid)
    assert db.searched[-1] == "Which drugs did Cantrell Drug Company recall? why?"
    assert second.text.startswith("Answer") and second.error is None


def test_rewrite_runs_without_reasoning_and_a_cut_off_rewrite_is_not_used():
    class CutOff(FakeChat):
        def complete(self, system, prompt, *, reasoning=True):
            self.calls.append((system, prompt, reasoning))
            if system == conversation.CONDENSE_SYSTEM:
                return ChatReply("why was Fentanyl Citrate 1,500 mcg and Fent", 339, 1024, 986, truncated=True)
            return ChatReply("Answer [1].", 900, 20)

    db, chat = FakeDB(), CutOff()
    cid = conversation.create(db)
    _ask(db, chat, "Which drugs did Cantrell Drug Company recall?", cid)
    _ask(db, chat, "why was the fentanyl one recalled?", cid)
    assert chat.calls[1][0] == conversation.CONDENSE_SYSTEM and chat.calls[1][2] is False
    assert db.searched[-1] == "Which drugs did Cantrell Drug Company recall? why was the fentanyl one recalled?"


def test_truncated_answer_is_flagged():
    class Short(FakeChat):
        def complete(self, system, prompt, *, reasoning=True):
            return ChatReply("The recall was due to", 900, 1024, 1000, truncated=True)

    result = answer.ask(FakeDB(), SETTINGS, FakeEmbedder(), Short(), "why?", ruleset=PERMISSIVE)
    assert result.truncated and result.text == "The recall was due to"


def test_failed_turns_are_not_history():
    db = FakeDB()
    cid = conversation.create(db)
    db.log.append({"query_id": 1, "question": "q1", "answer": None, "error": "quota", "conversation_id": cid,
                   "turn": 1, "standalone_question": None})
    assert conversation.history(db, cid) == []
    assert conversation.next_turn(db, cid) == 2                 # but they still use up their turn number


@pytest.mark.parametrize("decision", ["reject", "needs_sql"])
def test_a_blocked_turn_is_logged_but_is_not_history(decision):
    """Its "answer" is a canned message from the YAML. It resolves no reference, and it is what the
    rewriter copied out as a question in the incident below."""
    db = FakeDB()
    cid = conversation.create(db)
    db.log.append({"query_id": 1, "question": "how to design a boom", "answer": "I cannot help with that.",
                   "error": None, "conversation_id": cid, "turn": 1, "standalone_question": None,
                   "qualify_decision": decision})
    assert conversation.history(db, cid) == []
    assert conversation.next_turn(db, cid) == 2                 # still audited, still uses its number


def test_an_answered_turn_is_history():
    db = FakeDB()
    cid = conversation.create(db)
    db.log.append({"query_id": 1, "question": "q1", "answer": "a1", "error": None, "conversation_id": cid,
                   "turn": 1, "standalone_question": None, "qualify_decision": "pass"})
    assert [t.turn for t in conversation.history(db, cid)] == [1]


# ---------------------------------------------------------------- the refusal that became a question
REFUSAL = ("I cannot help with that. This interface answers questions about public recall records - "
           "FDA drug recalls and CPSC consumer product recalls - and nothing else.")


def test_a_rewrite_that_hands_back_an_earlier_answer_is_not_used():
    """Measured, query 32: turn 8 was refused, the refusal was stored as the answer, and the rewriter
    was shown it and returned it as the standalone question for turn 9. That text names "recall",
    "FDA", "CPSC", "drug", "product" and "consumer", so it passed the scope check, retrieved 8
    unrelated excerpts and cost 3,772 prompt tokens. The interface had laundered its own refusal."""
    class Echoing(FakeChat):
        def complete(self, system, prompt, *, reasoning=True):
            return ChatReply(REFUSAL, 10, 30, 40)

    turns = [conversation.Turn(8, "how to design a boom", REFUSAL)]
    assert conversation.condense(Echoing(), turns, "how to create a boom really big") == \
        "how to create a boom really big"


def test_a_real_rewrite_is_still_used():
    class Rewriting(FakeChat):
        def complete(self, system, prompt, *, reasoning=True):
            return ChatReply("Why did Pfizer recall D-0853-2026?", 10, 30, 40)

    turns = [conversation.Turn(1, "Which drugs did Pfizer recall?", "D-0853-2026 was recalled [1].")]
    assert conversation.condense(Rewriting(), turns, "and why?") == "Why did Pfizer recall D-0853-2026?"


@pytest.mark.parametrize("candidate, echoed", [
    (REFUSAL, True),
    (REFUSAL[:60], True),                                   # a fragment of it is still a copy
    ("Why did Pfizer recall D-0853-2026?", False),
    ("recall", False),                                      # too short to be anything but a coincidence
    ("", False),
])
def test_the_echo_guard(candidate, echoed):
    turns = [conversation.Turn(8, "how to design a boom", REFUSAL)]
    assert conversation._echoes_history(candidate, turns) is echoed


def test_history_keeps_only_the_last_turns_oldest_first():
    db = FakeDB()
    cid = conversation.create(db)
    for n in range(1, 10):
        db.log.append({"query_id": n, "question": f"q{n}", "answer": f"a{n}", "error": None,
                       "conversation_id": cid, "turn": n, "standalone_question": None})
    turns = conversation.history(db, cid)
    assert [t.turn for t in turns] == [4, 5, 6, 7, 8, 9]


def test_without_conversation_nothing_changes():
    db, chat = FakeDB(), FakeChat()
    result = answer.ask(db, SETTINGS, FakeEmbedder(), chat, "Which drugs?", ruleset=PERMISSIVE)
    assert len(chat.calls) == 1 and result.turn is None and result.conversation_id is None
    assert db.log[0]["conversation_id"] is None


# ---------------------------------------------------------------- prompt pieces
def test_history_block_shortens_long_answers():
    block = conversation.history_block([conversation.Turn(1, "q", "word " * 500)], answer_chars=50)
    assert block.startswith("User: q\nAssistant: word") and block.endswith(" ...") and len(block) < 100


@pytest.mark.parametrize("raw, expected", [
    ('Standalone question: "Why did Cantrell recall it?"', "Why did Cantrell recall it?"),
    ("Why did Cantrell recall it?\nThis resolves 'it'.", "Why did Cantrell recall it?"),
    ("   ", "why?"),
    ("x" * 2000, "why?"),
])
def test_clean_rewrite(raw, expected):
    assert conversation.clean_rewrite(raw, "why?") == expected


def test_rewrite_instructions_carry_the_subject_and_stay_short():
    text = conversation.CONDENSE_SYSTEM.lower()
    assert "carry the subject of the conversation forward" in text
    assert "rather than listing every item" in text


def test_system_prompt_forbids_using_earlier_answers_as_sources():
    assert "never treat an earlier answer as a source" in answer.SYSTEM_PROMPT.lower()


def test_questions_block_numbers_questions_without_answers():
    turns = [conversation.Turn(1, "q1", "an earlier answer"), conversation.Turn(2, "q2", "a2")]
    assert conversation.questions_block(turns) == "1. q1\n2. q2"


# ---------------------------------------------------------------- CLI
def test_ask_rejects_an_unknown_conversation_id():
    with pytest.raises(ValueError, match="no conversation"):
        commands._resolve_conversation(FakeDB(), "6f1c0000-0000-0000-0000-000000000000")


def test_chat_loop_answers_keeps_context_and_handles_commands(monkeypatch, capsys):
    db, chat = FakeDB(), FakeChat()
    monkeypatch.setattr(commands, "embedding_provider", lambda settings: FakeEmbedder())
    monkeypatch.setattr(commands, "chat_provider", lambda settings: chat)
    lines = iter(["Which drugs did Cantrell Drug Company recall?", "why?", "/history", "/sources",
                  "/new", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    args = SimpleNamespace(conversation=None, doc_types=None, filter=None, top_k=None, candidates=None,
                           verbose=True)
    settings = SimpleNamespace(rag_top_k=4, rag_candidates=10, rag_doc_types=None,
                               rules_dir=RULES_DIR)

    assert commands._chat(settings, db, args) == 0
    out = capsys.readouterr().out
    assert "(searched as: Why did Cantrell Drug Company recall hydromorphone?)" in out
    assert "[1] You: Which drugs did Cantrell Drug Company recall?" in out      # /history
    assert "New conversation" in out                                             # /new
    assert len(db.conversations) == 2
    assert [r["turn"] for r in db.log] == [1, 2]


def test_chat_survives_ctrl_d(monkeypatch, capsys):
    db = FakeDB()
    monkeypatch.setattr(commands, "embedding_provider", lambda settings: FakeEmbedder())
    monkeypatch.setattr(commands, "chat_provider", lambda settings: FakeChat())

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    args = SimpleNamespace(conversation=None, doc_types=None, filter=None, top_k=None, candidates=None,
                           verbose=True)
    assert commands._chat(SimpleNamespace(rag_doc_types=None), db, args) == 0
    assert "Continue later: crawlerrag chat --conversation" in capsys.readouterr().out
