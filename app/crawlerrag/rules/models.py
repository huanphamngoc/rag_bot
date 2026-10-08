"""The shape of a rule file, and the small value language inside it.

Everything here is pydantic with ``extra="forbid"``: a mistyped key in a YAML file is an error at load
time, never a silently missing line in 28,000 documents. ``ValueSpec`` is the only piece with
behaviour - it renders one value out of a source row, in text mode (for the title and the body) or in
raw mode (for the jsonb metadata, where a year has to stay an integer).

``schema`` is spelled ``db_schema`` in Python because pydantic reserves the name; the YAML key is
``schema``.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crawlerrag.model import TRACKABLE

Strict = ConfigDict(extra="forbid", populate_by_name=True)

SOURCES = ("field", "join", "template", "coalesce", "const")


def fold(text: str) -> str:
    """Lowercase and drop Vietnamese diacritics, for matching only.

    Measured on 2026-10-05: "Tong so vu thu hoi san pham nam 2025 la bao nhieu?" - a perfectly ordinary
    way to type Vietnamese on a keyboard without a Vietnamese layout - matched none of the aggregate
    patterns and was rejected as off topic. Folding both sides makes "tong so" match "tổng số". The
    question the user typed is never changed, only what the rules are compared against.
    """
    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("đ", "d")


_EDGE_PUNCTUATION = " \t\r\n.,!?;:\"'`()[]…-"


def _bare(folded: str) -> str:
    """A folded message with the punctuation around it removed, for whole-message matching.

    ``equals_any`` exists because substring matching cannot carry a short word. Measured: "ok" is inside
    "smoke", "token", "broken" and "brokers", so "ok" as a pattern rejects "Which smoke detectors were
    recalled?". A closing like "thanks" or "ok" is the *entire* message or it is not one at all.
    """
    return " ".join(folded.strip(_EDGE_PUNCTUATION).split())


def _clean(value: Any) -> str | None:
    """Exactly the cleaning the Python builders did: str(), strip, empty becomes nothing."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class ValueSpec(BaseModel):
    """One value of a document, built from one source row."""
    model_config = Strict

    field: str | None = None                      # a parent column or a children alias
    join: list[str] | None = None                 # several columns, non-empty ones joined
    template: str | None = None                   # "{col} - {col}", empty if any column is empty
    coalesce: list["ValueSpec"] | None = None     # the first alternative that renders
    const: str | None = None
    separator: str = ", "
    format: Literal["iso", "year", "thousands"] | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "ValueSpec":
        given = [name for name in SOURCES if getattr(self, name) is not None]
        if len(given) != 1:
            raise ValueError(f"a value needs exactly one of {', '.join(SOURCES)}, got {given or 'nothing'}")
        return self

    # ---- what it reads, for validation and for the generated SELECT
    def columns(self) -> set[str]:
        if self.field:
            return {self.field}
        if self.join:
            return set(self.join)
        if self.template:
            return template_columns(self.template)
        if self.coalesce:
            return {c for spec in self.coalesce for c in spec.columns()}
        return set()

    # ---- rendering
    def value(self, row: Mapping[str, Any]) -> Any:
        """The raw value: an int stays an int, a list stays a list (minus its NULLs), None when empty."""
        if self.const is not None:
            return self.const
        if self.coalesce:
            for spec in self.coalesce:
                if spec.text(row):
                    return spec.value(row)
            return None
        if self.template:
            return self.text(row) or None
        if self.join:
            return self.text(row) or None
        raw = self._formatted(row[self.field])
        if isinstance(raw, (list, tuple)):
            return [v for v in raw if v is not None]
        return raw

    def text(self, row: Mapping[str, Any]) -> str:
        """The rendered text, or "" when there is nothing to show."""
        if self.const is not None:
            return self.const
        if self.coalesce:
            for spec in self.coalesce:
                rendered = spec.text(row)
                if rendered:
                    return rendered
            return ""
        if self.template:
            parts = {}
            for name in template_columns(self.template):
                rendered = _as_text(self._formatted(row[name]), self.separator)
                if not rendered:
                    return ""                       # a template needs every column it names
                parts[name] = rendered
            return self.template.format(**parts)
        if self.join:
            joined = self.separator.join(
                part for part in (_as_text(self._formatted(row[name]), self.separator) for name in self.join)
                if part)
            return joined
        return _as_text(self._formatted(row[self.field]), self.separator)

    def _formatted(self, raw: Any) -> Any:
        if self.format is None:
            return raw
        if isinstance(raw, (list, tuple)):
            return [_apply_format(self.format, v) for v in raw]
        return _apply_format(self.format, raw)


def _apply_format(name: str, raw: Any) -> Any:
    if raw is None:
        return None
    if name == "iso":
        return raw.isoformat() if hasattr(raw, "isoformat") else raw
    if name == "year":
        return raw.year if hasattr(raw, "year") else None
    if name == "thousands":
        try:
            number = int(raw)
        except (TypeError, ValueError):
            return None
        return f"{number:,}" if number else None       # 0 and NULL print no line, as before
    return raw


def _as_text(raw: Any, separator: str) -> str:
    """A scalar or a list as the body wants it: non-empty parts joined, whole thing stripped."""
    if isinstance(raw, (list, tuple)):
        raw = separator.join(str(v) for v in raw if v is not None and str(v).strip())
    return _clean(raw) or ""


def template_columns(template: str) -> set[str]:
    from string import Formatter
    return {name for _, name, _, _ in Formatter().parse(template) if name}


# ---------------------------------------------------------------- where the rows come from
class SourceSpec(BaseModel):
    model_config = Strict

    source_id: str                                # crawl.source.source_id, filters the change log
    db_schema: str = Field(alias="schema")
    table: str
    key: str                                      # must equal crawl.record_change.record_key
    active_column: str = "is_active"
    order_by: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.db_schema}.{self.table}"


class ChildFilter(BaseModel):
    model_config = Strict

    column: str
    eq: str | None = None
    ne: str | None = None

    @model_validator(mode="after")
    def _one_comparison(self) -> "ChildFilter":
        if (self.eq is None) == (self.ne is None):
            raise ValueError("a child filter needs exactly one of eq, ne")
        return self


class ChildSpec(BaseModel):
    """One correlated array_agg: a collection belonging to the parent row."""
    model_config = Strict

    db_schema: str = Field(alias="schema")
    table: str
    join: dict[str, str]                          # child column -> parent column
    select: str
    order_by: str | None = None
    distinct: bool = False
    where: ChildFilter | None = None

    @field_validator("join")
    @classmethod
    def _join_not_empty(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("a child needs at least one join column")
        return value

    @property
    def qualified_name(self) -> str:
        return f"{self.db_schema}.{self.table}"


# ---------------------------------------------------------------- history and quality
class Scd2Spec(BaseModel):
    model_config = Strict

    track: list[str] = ["title", "body"]          # a change opens a new version
    overwrite: list[str] = ["url", "metadata"]    # a change updates the current version in place
    version_on_activation_change: bool = True

    @field_validator("track", "overwrite")
    @classmethod
    def _known_attributes(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - set(TRACKABLE))
        if unknown:
            raise ValueError(f"unknown document attribute(s): {', '.join(unknown)}; "
                             f"known: {', '.join(TRACKABLE)}")
        return value

    @model_validator(mode="after")
    def _no_attribute_in_both(self) -> "Scd2Spec":
        both = sorted(set(self.track) & set(self.overwrite))
        if both:
            raise ValueError(f"attribute(s) both tracked and overwritten: {', '.join(both)}")
        if not self.track:
            raise ValueError("scd2.track cannot be empty: nothing would ever open a version")
        return self


class QualityRule(BaseModel):
    model_config = Strict

    rule: Literal["not_null", "unique", "allowed_values", "max_null_fraction", "min_rows"]
    column: str | None = None
    values: list[str] | None = None
    max: float | None = None
    min: int | None = None
    severity: Literal["error", "warn"] = "error"

    @model_validator(mode="after")
    def _parameters_match_the_rule(self) -> "QualityRule":
        needs_column = self.rule != "min_rows"
        if needs_column and not self.column:
            raise ValueError(f"{self.rule} needs a column")
        if self.rule == "allowed_values" and not self.values:
            raise ValueError("allowed_values needs values")
        if self.rule == "max_null_fraction" and self.max is None:
            raise ValueError("max_null_fraction needs max")
        if self.rule == "min_rows" and self.min is None:
            raise ValueError("min_rows needs min")
        return self

    @property
    def level(self) -> str:
        return "error" if self.severity == "error" else "warning"

    def label(self) -> str:
        return f"{self.rule}({self.column})" if self.column else f"{self.rule}({self.min})"


class BodyLine(BaseModel):
    model_config = Strict

    label: str
    value: ValueSpec


# ---------------------------------------------------------------- a document type
class DocTypeRule(BaseModel):
    model_config = Strict

    doc_type: str
    version: int = 1                              # bump to force a full re-check on purpose
    source: SourceSpec
    children: dict[str, ChildSpec] = {}
    title: ValueSpec
    body: list[BodyLine]
    url: ValueSpec | None = None
    metadata: dict[str, ValueSpec] = {}
    scd2: Scd2Spec = Scd2Spec()
    quality: list[QualityRule] = []

    @field_validator("body")
    @classmethod
    def _body_not_empty(cls, value: list[BodyLine]) -> list[BodyLine]:
        if not value:
            raise ValueError("body needs at least one line")
        return value

    @property
    def text_digest(self) -> str:
        """Fingerprint of everything that decides a document's text.

        It goes into the ingest watermark signature, so editing a rule file switches the next run to a
        full re-check automatically - nobody has to remember to bump ``version``. Quality rules and the
        history settings are left out: tightening a check must not rebuild 28,000 documents.
        """
        shape = {name: self.model_dump(mode="json", by_alias=True)[name]
                 for name in ("version", "source", "children", "title", "body", "url", "metadata")}
        return hashlib.sha256(json.dumps(shape, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    def value_specs(self) -> list[tuple[str, ValueSpec]]:
        """Every spec with a label for error messages."""
        specs = [("title", self.title)]
        specs += [(f"body[{line.label}]", line.value) for line in self.body]
        if self.url is not None:
            specs.append(("url", self.url))
        specs += [(f"metadata[{name}]", spec) for name, spec in self.metadata.items()]
        return specs

    def doc_id(self, record_key: str) -> str:
        return f"{self.doc_type}:{record_key}"


# ---------------------------------------------------------------- the shared files
class QualifyLimits(BaseModel):
    model_config = Strict

    min_chars: int = 3
    min_words: int = 1            # a single word names a subject, not a question
    max_chars: int = 2000
    max_filters: int = 5
    max_doc_types: int = 6
    message_too_short: str = "Could you say a little more about what you are looking for?"
    message_too_vague: str = "What would you like to know about it?"
    message_too_long: str = "That question is longer than this interface accepts."
    message_too_many_filters: str = "Too many filters at once."


class QualifyRule(BaseModel):
    model_config = Strict

    id: str
    kind: Literal["reject", "needs_sql", "clarify"]
    patterns: list[str] = []                      # a substring of the message decides
    equals_any: list[str] = []                    # the WHOLE message decides
    require_any: list[str] = []                   # no match decides
    message: str
    sql_hint: str | None = None
    skip_for_follow_up: bool = False
    enabled: bool = True

    @model_validator(mode="after")
    def _one_way_of_matching(self) -> "QualifyRule":
        given = [name for name in ("patterns", "equals_any", "require_any") if getattr(self, name)]
        if len(given) != 1:
            raise ValueError(f"rule {self.id}: use exactly one of patterns, equals_any, require_any")
        return self

    def matches(self, question: str) -> bool:
        folded = fold(question)
        if self.equals_any:
            return _bare(folded) in {_bare(fold(p)) for p in self.equals_any}
        if self.patterns:
            return any(fold(p) in folded for p in self.patterns)
        return not any(fold(term) in folded for term in self.require_any)


class QualifyRules(BaseModel):
    model_config = Strict

    version: int = 1
    limits: QualifyLimits = QualifyLimits()
    rules: list[QualifyRule] = []

    @field_validator("rules")
    @classmethod
    def _unique_ids(cls, value: list[QualifyRule]) -> list[QualifyRule]:
        seen = [r.id for r in value]
        if len(set(seen)) != len(seen):
            raise ValueError("two qualify rules share an id")
        return value


class CatalogSpec(BaseModel):
    model_config = Strict

    schemas: list[str]
    exclude: list[str] = []

    @field_validator("schemas")
    @classmethod
    def _at_least_one(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("catalog.schemas cannot be empty")
        return value


def tracked_attributes(rule: DocTypeRule) -> Sequence[str]:
    return rule.scd2.track
