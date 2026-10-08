"""The document builders as they were BEFORE the YAML rules, frozen, for one purpose only.

``test_rules_build.py`` builds the same source rows through ``crawlerrag.rules`` and through these
functions and requires byte-identical title, body, url and metadata. That is what makes the move to
YAML safe: the vector database holds 35,746 chunks whose text was embedded for real money, and a
document whose text shifts by one character is re-chunked and re-embedded.

Copied verbatim from crawlerrag/ingest/documents.py at the commit before rules/ existed (which in
turn was a verbatim copy of the crawler's rag/documents.py). Never "fix" anything here: a difference
found by the golden test is a difference in the YAML, not here.
"""
from __future__ import annotations

from typing import Any


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _lines(*pairs: tuple[str, Any]) -> str:
    out = []
    for label, value in pairs:
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value if v is not None and str(v).strip())
        text = _clean(value)
        if text:
            out.append(f"{label}: {text}")
    return "\n".join(out)


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else _clean(value)


def _meta(**kwargs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)):
            value = [v for v in value if v is not None]
        if value is None or value == [] or value == "":
            continue
        out[key] = _iso(value) if hasattr(value, "isoformat") else value
    return out


def drug_recall(row: dict) -> dict:
    firm = _clean(row["recalling_firm"]) or "unknown firm"
    where = ", ".join(x for x in (_clean(row["city"]), _clean(row["state"])) if x)
    return {
        "doc_id": f"drug_recall:{row['recall_number']}",
        "doc_type": "drug_recall",
        "source_id": "openfda_enforcement",
        "title": f"FDA drug recall {row['recall_number']} - {firm}",
        "body": _lines(
            ("Recalling firm", firm),
            ("Firm location", where or row["country"]),
            ("Recall classification", row["classification"]),
            ("Status", row["status"]),
            ("Product type", row["product_type"]),
            ("Product description", row["product_description"]),
            ("Quantity in commerce", row["product_quantity"]),
            ("Reason for recall", row["reason_for_recall"]),
            ("Lot / code information", row["code_info"]),
            ("Distribution pattern", row["distribution_pattern"]),
            ("Voluntary or mandated", row["voluntary_mandated"]),
            ("How the firm notified the public", row["initial_firm_notification"]),
            ("National drug codes involved", row["product_ndcs"]),
            ("Recall initiated", _iso(row["recall_initiation_date"])),
            ("Reported by FDA", _iso(row["report_date"])),
            ("Terminated", _iso(row["termination_date"])),
        ),
        "url": None,
        "metadata": _meta(recall_number=row["recall_number"], event_id=row["event_id"],
                          classification=row["classification"], status=row["status"],
                          state=row["state"], country=row["country"], product_type=row["product_type"],
                          recall_date=row["recall_initiation_date"],
                          year=row["recall_initiation_date"].year if row["recall_initiation_date"] else None),
        "is_active": row["is_active"],
    }


def cpsc_recall(row: dict) -> dict:
    title = _clean(row["title"]) or f"CPSC recall {row['recall_number'] or row['recall_id']}"
    return {
        "doc_id": f"cpsc_recall:{row['recall_id']}",
        "doc_type": "cpsc_recall",
        "source_id": "cpsc_recall",
        "title": f"CPSC consumer product recall: {title}",
        "body": _lines(
            ("Recall number", row["recall_number"]),
            ("Recall date", _iso(row["recall_date"])),
            ("Description", row["description"]),
            ("Products recalled", row["products"]),
            ("Product type", row["product_types"]),
            ("Hazard", row["hazards"]),
            ("Hazard type", row["hazard_types"]),
            ("Reported injuries", row["injuries"]),
            ("Remedy", row["remedies"]),
            ("Remedy options", row["remedy_options"]),
            ("Manufacturer", row["manufacturers"]),
            ("Other companies involved", row["other_companies"]),
            ("Sold at", row["retailers"]),
            ("Manufactured in", row["countries"]),
            ("Units affected (approximate)",
             f"{row['units_total_approx']:,}" if row["units_total_approx"] else None),
        ),
        "url": _clean(row["url"]),
        "metadata": _meta(recall_id=row["recall_id"], recall_number=row["recall_number"],
                          recall_date=row["recall_date"],
                          year=row["recall_date"].year if row["recall_date"] else None,
                          hazard_types=list(row["hazard_types"] or []),
                          units_total_approx=row["units_total_approx"]),
        "is_active": row["is_active"],
    }
