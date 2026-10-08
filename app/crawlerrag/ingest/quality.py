"""Business rules checked on the extracted rows, before anything is written.

The rules live in ``rules/<doc_type>.yaml`` under ``quality:``. They run on the rows of one batch, in
memory, right after the snapshot is read - so a source that went wrong does not replace good documents
and, more importantly, does not move the watermark past the window it went wrong in.

``min_rows`` is the exception that only makes sense on a full load: an incremental batch legitimately
carries three rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from crawlerrag.rules.models import DocTypeRule, QualityRule


@dataclass(frozen=True)
class QualityFinding:
    level: str                      # "error" | "warning"
    doc_type: str
    rule: str
    column: str | None
    failed_rows: int
    checked_rows: int
    message: str

    def render(self) -> str:
        return f"[{self.level}] {self.doc_type}: {self.message}"


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return not [v for v in value if v is not None]
    return isinstance(value, str) and not value.strip()


def _check(rule: QualityRule, doc_type: str, rows: Sequence[Mapping[str, Any]],
           *, mode: str) -> QualityFinding | None:
    total = len(rows)
    if rule.rule == "min_rows":
        if mode != "full" or total >= (rule.min or 0):
            return None
        return QualityFinding(level=rule.level, doc_type=doc_type, rule=rule.rule, column=None,
                              failed_rows=total, checked_rows=total,
                              message=f"the source returned {total} rows, minimum is {rule.min}")

    values = [row[rule.column] for row in rows if rule.column in row]
    if rule.rule == "not_null":
        failed = sum(1 for v in values if _is_empty(v))
        if failed:
            return QualityFinding(level=rule.level, doc_type=doc_type, rule=rule.rule, column=rule.column,
                                  failed_rows=failed, checked_rows=total,
                                  message=f"{rule.column} is empty in {failed} of {total} rows")
        return None

    if rule.rule == "unique":
        seen: dict[Any, int] = {}
        for value in values:
            if _is_empty(value):
                continue
            key = tuple(value) if isinstance(value, list) else value
            seen[key] = seen.get(key, 0) + 1
        repeated = {k: n for k, n in seen.items() if n > 1}
        if repeated:
            examples = ", ".join(str(k) for k in list(repeated)[:3])
            return QualityFinding(level=rule.level, doc_type=doc_type, rule=rule.rule, column=rule.column,
                                  failed_rows=sum(repeated.values()), checked_rows=total,
                                  message=f"{rule.column} repeats {len(repeated)} value(s): {examples}")
        return None

    if rule.rule == "allowed_values":
        allowed = set(rule.values or ())
        offending = sorted({str(v) for v in values if not _is_empty(v) and str(v) not in allowed})
        if offending:
            return QualityFinding(level=rule.level, doc_type=doc_type, rule=rule.rule, column=rule.column,
                                  failed_rows=len([v for v in values
                                                   if not _is_empty(v) and str(v) not in allowed]),
                                  checked_rows=total,
                                  message=f"{rule.column} has {len(offending)} value(s) outside the "
                                          f"allowed set: {', '.join(offending[:5])}")
        return None

    if rule.rule == "max_null_fraction":
        if not total:
            return None
        failed = sum(1 for v in values if _is_empty(v))
        fraction = failed / total
        if fraction > (rule.max or 0):
            return QualityFinding(level=rule.level, doc_type=doc_type, rule=rule.rule, column=rule.column,
                                  failed_rows=failed, checked_rows=total,
                                  message=f"{rule.column} is empty in {fraction:.0%} of rows "
                                          f"({failed}/{total}), limit is {rule.max:.0%}")
        return None
    return None


def check_rows(rule: DocTypeRule, rows: Sequence[Mapping[str, Any]], *,
               mode: str = "incremental") -> list[QualityFinding]:
    found = []
    for check in rule.quality:
        finding = _check(check, rule.doc_type, rows, mode=mode)
        if finding is not None:
            found.append(finding)
    return found
