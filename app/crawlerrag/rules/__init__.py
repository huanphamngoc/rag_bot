"""Business rules as YAML: document types, their SQL, their history settings, their quality checks.

``rules/*.yaml`` is the only place a document type is described. This package reads those files
(:mod:`loader`), turns a source row into a document and a document type into a SELECT
(:mod:`build`), and checks the files against the metadata extracted from Postgres
(:mod:`validate`).
"""
from crawlerrag.rules.loader import RuleError, RuleSet, clear_cache, load_rules, load_rules_cached

__all__ = ["RuleError", "RuleSet", "load_rules", "load_rules_cached", "clear_cache"]
