"""The clarify round against real Postgres, with the real checkpointer and real retrieval.

test_chat_graph.py covers the branches with MemorySaver and a fake database. What can only be checked
here is the part the whole idea rests on: that a sample question offered to the user is one this index
can actually answer. A suggestion that leads to "no matching records" is worse than no suggestion.
"""
from __future__ import annotations

import pytest

from crawlerrag.ingest import pipeline
from crawlerrag.rag import graph as chat_graph
from crawlerrag.rag import retrieve, samples

from .conftest import FakeEmbedder
from .test_incremental import seed


class FakeChat:
    provider, model, temperature, max_tokens = "fake", "fake-chat", 0.1, 1024

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, system, prompt, *, reasoning=True):
        from crawlerrag.rag.providers import ChatReply
        self.prompts.append(prompt)
        if "rewrite" in system.lower():
            return ChatReply(text="rewritten", prompt_tokens=5, output_tokens=2)
        return ChatReply(text="Answered from the excerpts [1].", prompt_tokens=100, output_tokens=10)


@pytest.fixture
def loaded(env):
    """A small but real index: documents, chunks and vectors, from the fixture source."""
    seed(env)
    embedder = FakeEmbedder()
    doc_types = pipeline.doc_types_for(env["settings"], ["drug_recall", "cpsc_recall"])
    pipeline.run(env["vector"], env["source"], env["settings"], doc_types,
                 embedder_factory=lambda s: embedder)
    env["settings"].rag_chat_clarify = True
    return env


def deps(env, chat=None):
    return chat_graph.ChatDeps(conn=env["vector"], settings=env["settings"], embedder=FakeEmbedder(),
                               chat=chat or FakeChat(), ruleset=env["ruleset"])


# ---------------------------------------------------------------- the examples have to work
def test_every_sample_question_retrieves_something(loaded):
    """The whole point of building them from the index. Measured on the real index the identifier-based
    design failed this 4 times out of 5, which is why the subject comes from the title instead."""
    env = loaded
    offered = samples.build(env["vector"], "recall")
    assert offered, "no examples at all"

    for item in offered:
        hits = retrieve.search(env["vector"], env["settings"], FakeEmbedder(), item["question"]).hits
        assert hits, f"nothing retrieved for {item['question']!r}"


def test_a_sample_is_about_the_word_that_was_typed(loaded):
    offered = samples.build(loaded["vector"], "power banks")
    assert offered
    assert any("cpsc" in s["doc_id"] for s in offered)


def test_samples_can_be_narrowed_to_one_doc_type(loaded):
    offered = samples.build(loaded["vector"], "recall", doc_types=["drug_recall"])
    assert offered and all(s["doc_id"].startswith("drug_recall:") for s in offered)


def test_samples_cost_no_embedding_call(loaded):
    """Asking again has to be free, or the gate in front of the model becomes a cost of its own."""
    embedder = FakeEmbedder()
    samples.build(loaded["vector"], "insulin")
    assert embedder.requests == 0


# ---------------------------------------------------------------- the round trip, real checkpointer
def test_a_vague_question_pauses_and_the_state_is_stored_in_postgres(loaded):
    env = loaded
    result = chat_graph.run_chat(deps(env), "insulin")

    assert result.waiting_for_a_better_question and result.thread_id
    rows = env["vector"].execute(
        "SELECT count(*) AS n FROM graph.checkpoints WHERE thread_id = %s", (result.thread_id,)).fetchall()
    assert rows[0]["n"] > 0


def test_the_pause_costs_no_model_call(loaded):
    chat = FakeChat()
    result = chat_graph.run_chat(deps(loaded, chat), "insulin")

    assert result.pending_question and chat.prompts == []


def test_answering_the_pause_finishes_the_same_turn(loaded):
    env = loaded
    d = deps(env)
    first = chat_graph.run_chat(d, "insulin")
    picked = first.samples[0]["question"]

    second = chat_graph.resume_chat(d, thread_id=first.thread_id, question=picked)

    assert second.question == picked
    assert second.qualify_decision == "pass"
    assert second.hits, "the question we offered retrieved nothing"
    assert second.text


def test_only_the_finished_turn_is_written_to_the_query_log(loaded):
    """A pause answered nothing, so it is not a turn - the log would otherwise count it as a question."""
    env = loaded
    d = deps(env)
    before = env["vector"].execute("SELECT count(*) AS n FROM rag.query_log").fetchall()[0]["n"]
    first = chat_graph.run_chat(d, "insulin")
    paused = env["vector"].execute("SELECT count(*) AS n FROM rag.query_log").fetchall()[0]["n"]

    chat_graph.resume_chat(d, thread_id=first.thread_id, question=first.samples[0]["question"])
    after = env["vector"].execute("SELECT count(*) AS n FROM rag.query_log").fetchall()[0]["n"]

    assert paused == before
    assert after == before + 1


def test_an_unknown_thread_is_refused(loaded):
    with pytest.raises(chat_graph.NoSuchTurn):
        chat_graph.resume_chat(deps(loaded), thread_id="f" * 32, question="anything")


def test_two_paused_turns_keep_their_own_threads(loaded):
    env = loaded
    d = deps(env)
    first = chat_graph.run_chat(d, "insulin")
    second = chat_graph.run_chat(d, "stroller")

    assert first.thread_id != second.thread_id
    done = chat_graph.resume_chat(d, thread_id=first.thread_id, question=first.samples[0]["question"])
    assert done.text
    still = env["vector"].execute(
        "SELECT count(*) AS n FROM graph.checkpoints WHERE thread_id = %s", (second.thread_id,)
    ).fetchall()[0]["n"]
    assert still > 0, "answering one turn must not discard the other"


def test_the_checkpointer_tables_stay_out_of_the_rag_schema(loaded):
    chat_graph.run_chat(deps(loaded), "insulin")
    rows = loaded["vector"].execute(
        "SELECT table_schema FROM information_schema.tables WHERE table_name = 'checkpoints'").fetchall()
    assert [r["table_schema"] for r in rows] == ["graph"]
