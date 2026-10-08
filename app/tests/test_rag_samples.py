"""Sample questions built from the index, for the clarify step.

The design here was decided by measurement, not taste. The obvious version - "Why was drug recall
D-0445-2024 issued?" - finds its own document 1 time in 5, because `to_tsvector('english', ...)` splits
the identifier into '-0445', '-2024' and 'd', and 'd' is in 18,303 chunks so the rarity filter drops it;
the lexical side then goes looking for '-0445 | -2024 | issu'. So the subject comes from the title - a
firm, a product - which is what both halves of the hybrid actually match on.

A sample is therefore a question that *works*, about a subject that is in the index. It is not a pointer
to one document: "Why did Novo Nordisk Inc recall a drug?" returns Novo Nordisk recalls, which is the
point. test_rag_retrieval.py and the integration suite cover retrieval itself.
"""
from __future__ import annotations

from types import SimpleNamespace

import psycopg
import pytest

from crawlerrag.rag import samples


class FakeConn:
    """Returns the rows a test wants, or raises, for whichever of the two queries runs."""

    def __init__(self, rows=(), *, fail: bool = False):
        self.rows = list(rows)
        self.fail = fail
        self.sql: list[str] = []

    def execute(self, sql, params=None):
        self.sql.append(sql)
        if self.fail:
            raise psycopg.OperationalError("connection is bad")
        return SimpleNamespace(fetchall=lambda: self.rows, fetchone=lambda: self.rows[0] if self.rows else None)


def drug(number="D-0445-2024", firm="Eli Lilly & Company"):
    return {"doc_id": f"drug_recall:{number}", "doc_type": "drug_recall",
            "title": f"FDA drug recall {number} - {firm}",
            "metadata": {"recall_number": number}}


def cpsc(recall_id="10885", title="Galanz Americas Recalls Retro Refrigerators Due to Risk of Fire"):
    return {"doc_id": f"cpsc_recall:{recall_id}", "doc_type": "cpsc_recall",
            "title": f"CPSC consumer product recall: {title}",
            "metadata": {"recall_number": recall_id}}


# ---------------------------------------------------------------- the subject
def test_a_drug_recall_is_asked_about_by_its_firm():
    got = samples.build(FakeConn([drug()]), "insulin")
    assert got == [{"question": "Why did Eli Lilly & Company recall a drug?",
                    "doc_id": "drug_recall:D-0445-2024",
                    "title": "FDA drug recall D-0445-2024 - Eli Lilly & Company"}]


def test_a_cpsc_recall_is_asked_about_by_its_product_and_firm():
    got = samples.build(FakeConn([cpsc()]), "refrigerator")
    assert got[0]["question"] == ("What hazard did CPSC report for Retro Refrigerators recalled by "
                                 "Galanz Americas?")


def test_the_hazard_half_of_a_cpsc_headline_is_cut_off():
    """"Due to Risk of Fire" is in thousands of titles: it identifies nothing and makes the question long."""
    got = samples.build(FakeConn([cpsc(title="Acme Recalls Toasters Due to Burn Hazard")]), "toaster")
    assert "Due to" not in got[0]["question"]
    assert "Toasters recalled by Acme" in got[0]["question"]


def test_a_semicolon_also_ends_the_headline():
    row = cpsc(title="Acme Recalls Heaters Due to Fire Hazard; One Death Reported")
    assert "Death" not in samples.build(FakeConn([row]), "heater")[0]["question"]


def test_a_headline_that_does_not_say_recalls_is_used_whole():
    row = cpsc(title="CPSC, Firms Announce Recall of Jogging Strollers")
    got = samples.build(FakeConn([row]), "stroller")
    assert got[0]["question"] == ("What hazard did CPSC report for CPSC, Firms Announce Recall of "
                                 "Jogging Strollers?")


def test_a_recall_with_no_known_firm_is_skipped():
    """"FDA drug recall D-1 - unknown firm" would become "Why did unknown firm recall a drug?"."""
    assert samples.build(FakeConn([drug(firm="unknown firm")]), "insulin") == []


def test_a_drug_title_without_a_firm_at_all_is_skipped():
    row = {"doc_id": "drug_recall:D-1", "doc_type": "drug_recall",
           "title": "FDA drug recall D-1", "metadata": {}}
    assert samples.build(FakeConn([row]), "x") == []


def test_an_absurdly_long_subject_is_skipped():
    assert samples.build(FakeConn([drug(firm="A" * 200)]), "x") == []


def test_the_same_question_is_not_offered_twice():
    """One firm with three recalls would otherwise fill the list with one question repeated."""
    rows = [drug("D-1"), drug("D-2"), drug("D-3")]
    assert len(samples.build(FakeConn(rows), "insulin")) == 1


def test_a_mixed_result_keeps_both_kinds():
    got = samples.build(FakeConn([drug(), cpsc()]), "pfizer")
    assert [s["doc_id"].split(":")[0] for s in got] == ["drug_recall", "cpsc_recall"]


# ---------------------------------------------------------------- the queries
def test_the_lexical_query_runs_for_a_question():
    conn = FakeConn([drug()])
    samples.build(conn, "insulin")
    assert "ts_rank_cd" in conn.sql[0]
    assert "chunk" in conn.sql[0]


def test_nothing_typed_yet_falls_back_to_the_most_recent_records():
    conn = FakeConn([cpsc()])
    samples.build(conn, "")
    assert len(conn.sql) == 1
    assert "recall_date" in conn.sql[0]       # the fallback orders by date, not by relevance


def test_a_question_that_matches_nothing_falls_back_too():
    class Empty(FakeConn):
        def execute(self, sql, params=None):
            self.sql.append(sql)
            rows = [] if "ts_rank_cd" in sql else [cpsc()]
            return SimpleNamespace(fetchall=lambda: rows, fetchone=lambda: rows[0] if rows else None)

    conn = Empty()
    got = samples.build(conn, "xyzzy")
    assert len(conn.sql) == 2 and got


def test_doc_types_narrow_the_search():
    conn = FakeConn([drug()])
    samples.build(conn, "insulin", doc_types=["drug_recall"])
    assert "d.doc_type = ANY" in conn.sql[0]


def test_the_limit_is_passed_through():
    conn = FakeConn([drug()])
    samples.build(conn, "insulin", limit=2)
    assert "%(limit)s" in conn.sql[0]


# ---------------------------------------------------------------- failure
def test_a_database_problem_gives_no_examples_rather_than_an_error():
    """A clarify answer without examples is poorer; a 500 instead of one is worse."""
    assert samples.build(FakeConn(fail=True), "insulin") == []


@pytest.mark.parametrize("question", [None, "", "   "])
def test_an_empty_question_is_not_sent_to_the_lexical_query(question):
    conn = FakeConn([cpsc()])
    samples.build(conn, question)
    assert "ts_rank_cd" not in conn.sql[0]
