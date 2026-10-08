"""What counts as conversation history, against real Postgres.

test_rag_conversation.py covers the same rules against a FakeDB that imitates the query. This file is
here because the rule *is* a WHERE clause: the unit test can only prove the Python agrees with itself.

The rule was written after a measured incident on the live index (query 32). Turn 8, "how to design a
boom", was refused, and the refusal from qualify.yaml was stored as that turn's answer. Turn 9, "how to
create a boom really big", was shown that refusal by the rewrite step, which handed it back as the
standalone question to search for. The refusal names "recall", "FDA", "CPSC", "drug", "product" and
"consumer", so it passed the scope check that exists to stop exactly this, retrieved 8 unrelated
excerpts and cost 3,772 prompt tokens. The interface had laundered its own refusal into a question.
"""
from __future__ import annotations

import pytest

from crawlerrag.rag import answer as answer_mod
from crawlerrag.rag import conversation

REFUSAL = ("I cannot help with that. This interface answers questions about public recall records - "
           "FDA drug recalls and CPSC consumer product recalls - and nothing else.")


def log(conn, cid, turn, question, answer, *, decision="pass", rule=None, error=None):
    return answer_mod.log_query(
        conn, question=question, doc_types=None, top_k=5, hits=[], answer=answer,
        embed_provider="fake", embed_model="fake", chat_provider="fake", chat_model="fake",
        prompt_tokens=None, output_tokens=None, duration_ms=1, error=error,
        conversation_id=cid, turn=turn, qualify_decision=decision, qualify_rule=rule)


@pytest.fixture
def conv(env):
    conn = env["vector"]
    return conn, conversation.create(conn)


def test_an_answered_turn_is_history(conv):
    conn, cid = conv
    log(conn, cid, 1, "Which drugs did Pfizer recall?", "D-0853-2026 was recalled [1].")
    assert [t.question for t in conversation.history(conn, cid)] == ["Which drugs did Pfizer recall?"]


@pytest.mark.parametrize("decision, rule", [("reject", "unsafe"), ("reject", "off_topic"),
                                            ("needs_sql", "aggregate")])
def test_a_blocked_turn_is_not_history(conv, decision, rule):
    conn, cid = conv
    log(conn, cid, 1, "how to design a boom", REFUSAL, decision=decision, rule=rule)
    assert conversation.history(conn, cid) == []


def test_a_blocked_turn_is_still_in_the_log(conv):
    """It is the audit trail - what was asked and what the gate did with it. It is just not history."""
    conn, cid = conv
    log(conn, cid, 1, "how to design a boom", REFUSAL, decision="reject", rule="unsafe")
    rows = conn.execute("SELECT question, qualify_rule FROM rag.query_log WHERE conversation_id = %s::uuid",
                        (cid,)).fetchall()
    assert [(r["question"], r["qualify_rule"]) for r in rows] == [("how to design a boom", "unsafe")]


def test_a_blocked_turn_still_uses_up_its_turn_number(conv):
    conn, cid = conv
    log(conn, cid, 1, "how to design a boom", REFUSAL, decision="reject", rule="unsafe")
    assert conversation.next_turn(conn, cid) == 2


def test_the_refusal_cannot_reach_the_rewriter(conv):
    """The incident end to end: a refused turn followed by a question, and the history the rewrite step
    would be given. Before the fix this held the refusal, which is what came back out of it."""
    conn, cid = conv
    log(conn, cid, 1, "Which drugs did Pfizer recall?", "D-0853-2026 was recalled [1].")
    log(conn, cid, 2, "how to design a boom", REFUSAL, decision="reject", rule="unsafe")

    turns = conversation.history(conn, cid)
    assert [t.turn for t in turns] == [1]
    assert not any(REFUSAL[:40] in t.answer for t in turns)
    assert not conversation._echoes_history(REFUSAL, turns)


def test_a_failed_turn_is_not_history_either(conv):
    """A provider error leaves no answer; it was already excluded and still is."""
    conn, cid = conv
    log(conn, cid, 1, "Which drugs did Pfizer recall?", None, error="quota exhausted")
    assert conversation.history(conn, cid) == []


def test_turns_logged_before_this_column_existed_still_count(conv):
    """`qualify_decision` is nullable and older rows have none. coalesce() keeps them as history rather
    than silently emptying every conversation that predates the column."""
    conn, cid = conv
    log(conn, cid, 1, "Which drugs did Pfizer recall?", "D-0853-2026 [1].", decision=None)
    assert [t.turn for t in conversation.history(conn, cid)] == [1]
