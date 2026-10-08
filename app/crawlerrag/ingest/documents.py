"""A document type: a YAML rule plus the few things the pipeline asks of it.

The SQL and the formatting used to live here as Python. They now live in ``rules/*.yaml`` and
``crawlerrag.rules``; what is left is the adapter the pipeline talks to, and the signature that decides
whether a run has to re-check everything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import psycopg

from crawlerrag.model import Document
from crawlerrag.rules import RuleSet, load_rules_cached
from crawlerrag.rules import build as rules_build
from crawlerrag.rules.models import DocTypeRule, Scd2Spec


@dataclass(frozen=True)
class DocType:
    rule: DocTypeRule

    @classmethod
    def from_rule(cls, rule: DocTypeRule) -> "DocType":
        return cls(rule=rule)

    # ---- what the pipeline reads
    @property
    def doc_type(self) -> str:
        return self.rule.doc_type

    @property
    def source_id(self) -> str:
        return self.rule.source.source_id

    @property
    def key_column(self) -> str:
        return self.rule.source.key

    @property
    def version(self) -> int:
        return self.rule.version

    @property
    def scd2(self) -> Scd2Spec:
        return self.rule.scd2

    def doc_id(self, record_key: str) -> str:
        return self.rule.doc_id(record_key)

    def build(self, row: Mapping[str, Any]) -> Document:
        return rules_build.build_document(self.rule, row)

    def signature(self, settings) -> str:
        """Everything that changes a document's chunks: the rule's text shape and the chunk settings.

        ``rule=<digest>`` is what makes editing a YAML file enough: the next run sees a different
        signature, switches to a full re-check and re-chunks. Documents whose text did not actually
        change keep their vectors, because the chunk text is identical.
        """
        return (f"v{self.rule.version}|rule={self.rule.text_digest[:12]}"
                f"|chunk={settings.rag_chunk_chars}/{settings.rag_chunk_overlap}")


def fetch_rows(conn: psycopg.Connection, doc_type: DocType,
               keys: Sequence[str] | None) -> list[dict]:
    """Every row (``keys=None``) or the rows of these record keys, in the caller's snapshot."""
    return rules_build.fetch_rows(conn, doc_type.rule, keys)


def doc_types_from(ruleset: RuleSet, names: Sequence[str] | None = None,
                   default: str = "") -> list[DocType]:
    return [DocType.from_rule(rule) for rule in ruleset.resolve(names, default)]


def doc_types_for(settings, names: Sequence[str] | None = None) -> list[DocType]:
    """The doc types a command was asked for, from the rule folder the settings point at."""
    ruleset = load_rules_cached(settings.rules_dir)
    return doc_types_from(ruleset, names, settings.rag_doc_types)


def available(settings) -> list[str]:
    return list(load_rules_cached(settings.rules_dir).doc_types)
