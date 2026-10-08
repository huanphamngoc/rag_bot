"""Reading the rule folder.

One file per document type under ``doc_types/``, plus ``qualify.yaml`` and ``catalog.yaml``. Every
failure names the file: these files are edited by hand and a message like "1 validation error for
DocTypeRule" with no file name is useless at three in the morning.

``yaml.safe_load`` only - a rule file is configuration, never code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml
from pydantic import ValidationError

from crawlerrag.rules.models import CatalogSpec, DocTypeRule, QualifyRules

log = logging.getLogger(__name__)


class RuleError(RuntimeError):
    """A rule folder that cannot be loaded. The message always names the file."""


@dataclass(frozen=True)
class RuleSet:
    doc_types: dict[str, DocTypeRule]
    qualify: QualifyRules
    catalog: CatalogSpec
    path: Path

    def resolve(self, names: Sequence[str] | None, default: str = "") -> list[DocTypeRule]:
        """Doc types by name; empty means ``default`` (RAG_DOC_TYPES), "all" means every one."""
        wanted = [n.strip() for n in (names or []) if n and n.strip()]
        if not wanted:
            wanted = [n.strip() for n in default.split(",") if n.strip()]
        if wanted == ["all"]:
            wanted = list(self.doc_types)
        unknown = sorted(set(wanted) - set(self.doc_types))
        if unknown:
            raise ValueError(f"unknown doc type(s): {', '.join(unknown)}. "
                             f"available: {', '.join(self.doc_types)}")
        return [self.doc_types[n] for n in dict.fromkeys(wanted)]


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleError(f"{path.name}: cannot be read ({exc})") from exc
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise RuleError(f"{path.name}: not valid YAML - {exc}") from exc


def _parse(path: Path, model, data: Any):
    if not isinstance(data, dict):
        raise RuleError(f"{path.name}: expected a mapping at the top level, got {type(data).__name__}")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
                            for err in exc.errors())
        raise RuleError(f"{path.name}: {problems}") from exc


def load_rules(path: str | Path) -> RuleSet:
    path = Path(path)
    doc_dir = path / "doc_types"
    if not doc_dir.is_dir():
        raise RuleError(f"{path}: no doc_types/ directory - a rule folder needs one file per document type")
    files = sorted(p for p in doc_dir.iterdir() if p.suffix in (".yaml", ".yml"))
    if not files:
        raise RuleError(f"{doc_dir}: no *.yaml files - nothing to ingest")

    doc_types: dict[str, DocTypeRule] = {}
    for file in files:
        rule = _parse(file, DocTypeRule, _read_yaml(file))
        if rule.doc_type != file.stem:
            raise RuleError(f"{file.name}: declares doc_type {rule.doc_type!r} but the file is named "
                            f"{file.stem!r}; one file per document type, named after it")
        if rule.doc_type in doc_types:
            raise RuleError(f"{file.name}: doc_type {rule.doc_type!r} is already declared elsewhere")
        doc_types[rule.doc_type] = rule

    qualify_file, catalog_file = path / "qualify.yaml", path / "catalog.yaml"
    if not qualify_file.is_file():
        raise RuleError(f"{qualify_file}: missing - the chat flow needs its input rules")
    if not catalog_file.is_file():
        raise RuleError(f"{catalog_file}: missing - which schemas to read metadata from")
    qualify = _parse(qualify_file, QualifyRules, _read_yaml(qualify_file))
    catalog = _parse(catalog_file, CatalogSpec, _read_yaml(catalog_file))

    log.debug("rules loaded", extra={"path": str(path), "doc_types": list(doc_types)})
    return RuleSet(doc_types=doc_types, qualify=qualify, catalog=catalog, path=path)


_cache: dict[Path, RuleSet] = {}


def load_rules_cached(path: str | Path) -> RuleSet:
    """Same, but read once per process: the web worker answers every request with these."""
    key = Path(path).resolve()
    if key not in _cache:
        _cache[key] = load_rules(key)
    return _cache[key]


def clear_cache() -> None:
    _cache.clear()
