"""The YAML rules must build exactly the documents the Python builders built.

The golden test (``test_the_yaml_rules_rebuild_the_frozen_*``) is what makes this refactor safe to run
against the live vector database: identical title+body means an identical content hash, so `ingest`
sees "unchanged" and nothing is re-chunked or re-embedded. The 35,746 chunks already in the index cost
real money to embed.

The rest of the file pins the small value language the YAML uses: field / join / template / coalesce /
const, plus the iso / year / thousands formats.
"""
from __future__ import annotations

import datetime as dt
import hashlib

import pytest

from tests import reference_builders as ref


# ---------------------------------------------------------------- source rows, as the generated SQL returns them
def drug_row(**over):
    row = {"record_key": "D-0853-2026", "recall_number": "D-0853-2026", "event_id": "E-12345",
           "status": "Ongoing", "classification": "Class II", "product_type": "Drugs",
           "recalling_firm": "Pfizer Inc.", "city": "New York", "state": "NY", "country": "United States",
           "voluntary_mandated": "Voluntary: Firm initiated", "initial_firm_notification": "Letter",
           "distribution_pattern": "Nationwide", "product_description": "Hydromorphone 2 mg/mL injection",
           "product_quantity": "1,200 vials", "reason_for_recall": "Lack of Assurance of Sterility",
           "code_info": "Lot 2026-07", "recall_initiation_date": dt.date(2026, 9, 4),
           "report_date": dt.date(2026, 9, 18), "termination_date": None, "is_active": True,
           "product_ndcs": ["0069-0010", "0069-0011"]}
    row.update(over)
    return row


def cpsc_row(**over):
    row = {"record_key": "25-301", "recall_id": "25-301", "recall_number": "25301",
           "recall_date": dt.date(2025, 5, 2), "title": "Power banks recalled due to fire hazard",
           "description": "The lithium-ion battery can overheat.", "url": "https://www.cpsc.gov/Recalls/1",
           "injuries": ["None reported"], "remedies": ["Refund"], "units_total_approx": 1234567,
           "is_active": True, "hazards": ["The battery can overheat, posing fire and burn hazards."],
           "hazard_types": ["Fire", "Burn"], "products": ["AnkerPower 500"], "product_types": ["Power bank"],
           "manufacturers": ["Anker Innovations"], "other_companies": ["Anker Technology"],
           "retailers": ["Amazon.com", "Walmart"], "remedy_options": ["Refund", "Replacement"],
           "countries": ["China"]}
    row.update(over)
    return row


DRUG_CASES = {
    "typical": {},
    "no firm": {"recalling_firm": None},
    "blank firm": {"recalling_firm": "   "},
    "no city": {"city": None},
    "no city or state": {"city": None, "state": None},
    "no location at all": {"city": None, "state": None, "country": None},
    "no ndcs": {"product_ndcs": None},
    "empty ndcs": {"product_ndcs": []},
    "ndc list with a hole": {"product_ndcs": ["0069-0010", None]},
    "no dates": {"recall_initiation_date": None, "report_date": None},
    "terminated": {"termination_date": dt.date(2026, 10, 1)},
    "inactive": {"is_active": False},
    "everything optional missing": {"event_id": None, "status": None, "classification": None,
                                    "product_type": None, "product_description": None,
                                    "product_quantity": None, "reason_for_recall": None, "code_info": None,
                                    "distribution_pattern": None, "voluntary_mandated": None,
                                    "initial_firm_notification": None, "recall_initiation_date": None,
                                    "report_date": None, "state": None, "country": None},
}

CPSC_CASES = {
    "typical": {},
    "no title": {"title": None},
    "no title and no recall number": {"title": None, "recall_number": None},
    "blank title": {"title": "  "},
    "no url": {"url": None},
    "blank url": {"url": " "},
    "no units": {"units_total_approx": None},
    "zero units": {"units_total_approx": 0},
    "one unit": {"units_total_approx": 1},
    "no children": {"hazards": None, "hazard_types": None, "products": None, "product_types": None,
                    "manufacturers": None, "other_companies": None, "retailers": None,
                    "remedy_options": None, "countries": None},
    "empty children": {"hazards": [], "hazard_types": [], "products": [], "retailers": []},
    "hazard types with a hole": {"hazard_types": ["Fire", None]},
    "no date": {"recall_date": None},
    "inactive": {"is_active": False},
}


def _as_dict(doc):
    return {"doc_id": doc.doc_id, "doc_type": doc.doc_type, "source_id": doc.source_id, "title": doc.title,
            "body": doc.body, "url": doc.url, "metadata": doc.metadata, "is_active": doc.is_active}


@pytest.mark.parametrize("name", sorted(DRUG_CASES))
def test_the_yaml_rules_rebuild_the_frozen_drug_documents(ruleset, name):
    from crawlerrag.rules import build
    row = drug_row(**DRUG_CASES[name])
    assert _as_dict(build.build_document(ruleset.doc_types["drug_recall"], row)) == ref.drug_recall(row)


@pytest.mark.parametrize("name", sorted(CPSC_CASES))
def test_the_yaml_rules_rebuild_the_frozen_cpsc_documents(ruleset, name):
    from crawlerrag.rules import build
    row = cpsc_row(**CPSC_CASES[name])
    assert _as_dict(build.build_document(ruleset.doc_types["cpsc_recall"], row)) == ref.cpsc_recall(row)


def test_the_content_hash_of_a_rebuilt_document_is_unchanged(ruleset):
    """What protects the embeddings already paid for: same text in, same hash out."""
    from crawlerrag.rules import build
    row = drug_row()
    doc = build.build_document(ruleset.doc_types["drug_recall"], row)
    expected = ref.drug_recall(row)
    assert doc.content_hash == hashlib.sha256(
        f"{expected['title']}\n{expected['body']}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the value language
def spec(**kwargs):
    from crawlerrag.rules.models import ValueSpec
    return ValueSpec.model_validate(kwargs)


def test_a_field_renders_its_column():
    assert spec(field="a").text({"a": "x"}) == "x"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_an_empty_field_renders_nothing(value):
    assert spec(field="a").text({"a": value}) == ""


def test_a_join_skips_empty_parts():
    assert spec(join=["a", "b", "c"]).text({"a": "x", "b": None, "c": "z"}) == "x, z"


def test_a_join_of_only_empty_parts_renders_nothing():
    assert spec(join=["a", "b"]).text({"a": None, "b": " "}) == ""


def test_a_list_value_is_joined_with_the_separator():
    assert spec(field="a").text({"a": ["x", "y"]}) == "x, y"


def test_a_list_value_drops_empty_elements():
    assert spec(field="a").text({"a": ["x", None, " ", "y"]}) == "x, y"


def test_a_template_fills_columns():
    assert spec(template="{a} - {b}").text({"a": "x", "b": "y"}) == "x - y"


def test_a_template_renders_nothing_when_a_column_it_needs_is_empty():
    """This is how coalesce falls through to the next alternative."""
    assert spec(template="{a} - {b}").text({"a": "x", "b": None}) == ""


def test_coalesce_takes_the_first_alternative_that_renders():
    s = spec(coalesce=[{"field": "a"}, {"field": "b"}, {"const": "unknown"}])
    assert s.text({"a": None, "b": "y"}) == "y"
    assert s.text({"a": None, "b": None}) == "unknown"


def test_format_iso_renders_a_date():
    assert spec(field="d", format="iso").text({"d": dt.date(2026, 9, 4)}) == "2026-09-04"


def test_format_year_renders_the_year_of_a_date():
    assert spec(field="d", format="year").value({"d": dt.date(2026, 9, 4)}) == 2026


def test_format_year_of_no_date_is_nothing():
    assert spec(field="d", format="year").value({"d": None}) is None


def test_format_thousands_groups_digits():
    assert spec(field="n", format="thousands").text({"n": 1234567}) == "1,234,567"


def test_format_thousands_of_zero_renders_nothing():
    """The frozen builder printed no "Units affected" line for 0 or NULL."""
    assert spec(field="n", format="thousands").text({"n": 0}) == ""


def test_a_raw_value_keeps_its_python_type():
    """Metadata is jsonb used for filters: a year must stay an int, a list a list."""
    assert spec(field="n").value({"n": 1234567}) == 1234567
    assert spec(field="xs").value({"xs": ["a", "b"]}) == ["a", "b"]


def test_a_raw_list_value_drops_only_nulls():
    assert spec(field="xs").value({"xs": ["a", None, ""]}) == ["a", ""]


def test_a_spec_must_name_exactly_one_source():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        spec(field="a", const="b")
    with pytest.raises(ValidationError):
        spec()


def test_an_unknown_key_in_a_spec_is_rejected():
    """A typo must fail loudly at load time, not silently render an empty line."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        spec(feild="a")
