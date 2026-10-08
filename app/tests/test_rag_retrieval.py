"""Fusion of the two retrievers, metadata filters, and the answer prompt."""
import json

import pytest

from crawlerrag.rag import answer as answer_mod
from crawlerrag.rag import retrieve


def row(chunk_id: int, *, doc_id: str | None = None, distance: float | None = None,
        lexical_rank: float | None = None) -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id or f"d{chunk_id}", "doc_type": "cpsc_recall",
            "source_id": "cpsc_recall", "title": f"title {chunk_id}", "url": None,
            "text": f"text {chunk_id}", "metadata": {"state": "CA"},
            "distance": distance, "lexical_rank": lexical_rank}


# ---------------------------------------------------------------- fusion
def test_rrf_ranks_a_document_found_by_both_retrievers_first():
    vector = [row(1), row(2), row(3)]
    text = [row(3), row(4)]
    hits = retrieve.fuse(vector, text, top_k=4)
    assert hits[0].chunk_id == 3
    assert hits[0].matched_by == "both"
    assert hits[0].score == pytest.approx(1 / (retrieve.RRF_K + 3) + 1 / (retrieve.RRF_K + 1))


def test_single_list_ranks_follow_input_order():
    hits = retrieve.fuse([row(7), row(8)], [], top_k=5)
    assert [h.chunk_id for h in hits] == [7, 8]
    assert [h.vector_rank for h in hits] == [1, 2]
    assert all(h.text_rank is None for h in hits)
    assert all(h.matched_by == "vector" for h in hits)


def test_text_only_hit_is_labelled_text():
    hits = retrieve.fuse([], [row(5, lexical_rank=0.4)], top_k=5)
    assert hits[0].matched_by == "text"
    assert hits[0].lexical_rank == 0.4
    assert hits[0].distance is None


def test_top_k_truncates():
    assert len(retrieve.fuse([row(i) for i in range(20)], [], top_k=3)) == 3


def test_one_document_cannot_occupy_the_whole_context():
    # Five chunks of the same document, ranked above everything else.
    vector = [row(i, doc_id="same") for i in range(1, 6)] + [row(99, doc_id="other")]
    hits = retrieve.fuse(vector, [], top_k=5)
    assert sum(1 for h in hits if h.doc_id == "same") == retrieve.MAX_PER_DOC
    assert any(h.doc_id == "other" for h in hits)


def test_distance_and_scores_are_carried_through():
    hits = retrieve.fuse([row(1, distance=0.25)], [row(1, lexical_rank=0.8)], top_k=1)
    assert hits[0].distance == 0.25 and hits[0].lexical_rank == 0.8


# ---------------------------------------------------------------- filters
def test_parse_filters_keeps_numbers_numeric():
    assert retrieve.parse_filters(["state=CA", "year=2024", "is_dental=true"]) == \
           {"state": "CA", "year": 2024, "is_dental": True}


def test_parse_filters_accepts_values_containing_equals():
    assert retrieve.parse_filters(["title=a=b"]) == {"title": "a=b"}


def test_parse_filters_rejects_malformed_input():
    with pytest.raises(ValueError, match="key=value"):
        retrieve.parse_filters(["state"])
    with pytest.raises(ValueError, match="without a key"):
        retrieve.parse_filters(["=CA"])


def test_parse_filters_of_nothing_is_empty():
    assert retrieve.parse_filters(None) == {} and retrieve.parse_filters([]) == {}


def test_filter_sql_binds_doc_types_and_metadata():
    params: dict = {}
    sql = retrieve._filter_sql(["cpsc_recall"], {"state": "CA"}, params)
    assert "d.is_active" in sql
    assert "d.doc_type = ANY(%(doc_types)s)" in sql
    assert "d.metadata @> %(filters)s::jsonb" in sql
    assert params["doc_types"] == ["cpsc_recall"]
    assert params["filters"] == '{"state": "CA"}'


def test_a_number_filter_also_looks_for_the_value_stored_as_a_string():
    """cpsc_recall.recall_number is the JSON string "26649"; --filter recall_number=26649 arrives as an
    int, and type-exact containment found nothing. Both forms have to be tried."""
    params: dict = {}
    sql = retrieve._filter_sql(None, {"recall_number": 26649}, params)
    assert "d.metadata @> %(filter0)s::jsonb OR d.metadata @> %(filter0_text)s::jsonb" in sql
    assert params["filter0"] == '{"recall_number": 26649}'
    assert params["filter0_text"] == '{"recall_number": "26649"}'


def test_a_boolean_filter_also_looks_for_the_value_stored_as_a_string():
    params: dict = {}
    retrieve._filter_sql(None, {"is_dental": True}, params)
    assert params["filter0"] == '{"is_dental": true}'
    assert params["filter0_text"] == '{"is_dental": "true"}'
    params = {}
    retrieve._filter_sql(None, {"is_dental": False}, params)
    assert params["filter0_text"] == '{"is_dental": "false"}'


def test_a_text_filter_stays_a_single_containment():
    """Only the values parse_filters had to guess the type of are looked up twice."""
    params: dict = {}
    sql = retrieve._filter_sql(None, {"state": "CA"}, params)
    assert sql.count("@>") == 1
    assert params == {"filters": '{"state": "CA"}'}


def test_mixed_filters_keep_their_own_parameters():
    params: dict = {}
    sql = retrieve._filter_sql(None, {"year": 2026, "state": "CA", "event_id": 99824}, params)
    assert sql.count("@>") == 5          # two ambiguous values, two forms each, plus the text filter
    assert params["filters"] == '{"state": "CA"}'
    # The number is the key's position in the sorted filters (event_id, state, year), so a parameter
    # name never carries user input into the SQL.
    assert params["filter0"] == '{"event_id": 99824}'
    assert params["filter0_text"] == '{"event_id": "99824"}'
    assert params["filter2"] == '{"year": 2026}'
    assert params["filter2_text"] == '{"year": "2026"}'


def test_every_filter_is_a_bound_parameter():
    """A filter value is never pasted into the SQL, whatever it contains."""
    params: dict = {}
    payload = "CA'; DROP TABLE rag.document; --"
    sql = retrieve._filter_sql(None, {"state": payload}, params)
    assert "DROP" not in sql
    assert json.loads(params["filters"]) == {"state": payload}


def test_filter_sql_without_restrictions_still_hides_soft_deleted_rows_and_old_versions():
    """rag.document keeps every version (SCD Type 2); only the current, active one is searchable."""
    params: dict = {}
    assert retrieve._filter_sql(None, None, params).strip() == "AND d.is_current AND d.is_active"
    assert params == {}


def test_text_search_keeps_only_the_rarest_lexemes():
    seen = {}

    class Conn:
        def execute(self, sql, params=None):
            seen["sql"], seen["params"] = sql, params
            return type("R", (), {"fetchall": lambda self: []})()

    retrieve.text_search(Conn(), "why was the fentanyl one recalled?", limit=40)
    assert seen["params"]["max_df"] == retrieve.MAX_LEXEME_DF == 0.10
    assert seen["params"]["max_lexemes"] == retrieve.MAX_LEXEMES == 8
    sql = " ".join(seen["sql"].split())
    assert "LEFT JOIN rag.lexeme_stat s ON s.lexeme = lex" in sql
    assert "WHERE coalesce(s.ndoc, 0) <= %(max_df)s * total.n" in sql
    assert "ORDER BY coalesce(s.ndoc, 0), lex LIMIT %(max_lexemes)s" in sql
    assert "ts_rank_cd(c.tsv, query.tsq)" in sql


# ---------------------------------------------------------------- prompt
def test_context_block_numbers_excerpts_and_shows_ids():
    hits = retrieve.fuse([row(1), row(2)], [], top_k=2)
    hits[0].url = "https://example.org/a"
    block = answer_mod.context_block(hits)
    assert block.startswith("[1] type=cpsc_recall id=d1 url=https://example.org/a")
    assert "[2] type=cpsc_recall id=d2" in block


def test_context_block_truncates_a_long_excerpt():
    hits = retrieve.fuse([row(1)], [], top_k=1)
    hits[0].text = "x" * 5000
    block = answer_mod.context_block(hits, max_chars=100)
    assert block.endswith(" ...")
    assert len(block) < 200


def test_prompt_contains_context_and_question():
    hits = retrieve.fuse([row(1)], [], top_k=1)
    prompt = answer_mod.build_prompt("what happened?", hits)
    assert "CONTEXT" in prompt and "QUESTION\nwhat happened?" in prompt
    assert "cite as [1]" in prompt


def test_prompt_without_hits_tells_the_model_to_say_so():
    prompt = answer_mod.build_prompt("anything", [])
    assert "no records were retrieved" in prompt


def test_system_prompt_states_the_two_guard_rails():
    text = answer_mod.SYSTEM_PROMPT.lower()
    assert "only the numbered excerpts" in text          # no outside knowledge
    assert "sql query" in text                            # aggregates are not a retrieval job
    assert "same language the question was asked in" in text
    # "were any of them Class I?" once got "No" from 8 Class II excerpts; the firm had 7 Class I recalls
    assert "never conclude that no record" in text
