"""The document, the one data shape that crosses every layer.

It lives here rather than in ``ingest`` or ``rules`` because both build it and both read it: the rules
turn a source row into one, the pipeline hashes it, versions it and splits it into chunks.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

TRACKABLE = ("title", "body", "url", "metadata")


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    source_id: str
    title: str
    body: str
    url: str | None
    metadata: dict[str, Any]
    is_active: bool

    @property
    def content_hash(self) -> str:
        """sha256(title + "\\n" + body) - the formula the whole existing index was hashed with."""
        return hashlib.sha256(f"{self.title}\n{self.body}".encode("utf-8")).hexdigest()

    @property
    def text(self) -> str:
        """What gets chunked and embedded: the title travels with every chunk."""
        return f"{self.title}\n\n{self.body}"

    def tracked_hash(self, attrs: Sequence[str]) -> str:
        """The hash a Type 2 comparison uses: the attributes ``scd2.track`` names.

        ``["title", "body"]`` returns ``content_hash`` unchanged, and that is not a detail: the 27,989
        documents in the live index were hashed with that exact formula, so a different one would make
        every document look changed on the next run and open a version for all of them.
        """
        wanted = sorted(set(attrs))
        if wanted == ["body", "title"]:
            return self.content_hash
        parts = []
        for name in wanted:
            value = getattr(self, name)
            if name == "metadata":
                value = "\x1f".join(f"{k}={value[k]!r}" for k in sorted(value or {}))
            parts.append(f"{name}={value if value is not None else ''}")
        return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()
