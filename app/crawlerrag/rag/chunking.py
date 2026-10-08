"""Splitting a document into embeddable chunks.

These documents are short, line-structured records ("Label: value"), not prose, so the
splitter prefers line boundaries, falls back to sentence boundaries inside a very long
line (an ingredient list or a recall description can be a single 6 kB line), and only
then cuts mid-text. Each chunk after the first repeats a tail of the previous one, so a
fact split across a boundary is still retrievable.
"""
from __future__ import annotations

import hashlib
import re

SENTENCE_END = re.compile(r"(?<=[.;!?])\s+")


def _hard_split(text: str, size: int) -> list[str]:
    """Split one oversized line on sentence boundaries, then on whitespace, then blindly."""
    parts: list[str] = []
    buf = ""
    for piece in SENTENCE_END.split(text):
        if not piece:
            continue
        if len(buf) + len(piece) + 1 <= size:
            buf = f"{buf} {piece}".strip()
            continue
        if buf:
            parts.append(buf)
        while len(piece) > size:
            cut = piece.rfind(" ", 0, size)
            if cut <= size // 2:
                cut = size
            parts.append(piece[:cut].strip())
            piece = piece[cut:].lstrip()
        buf = piece
    if buf:
        parts.append(buf)
    return parts


def split_text(text: str, *, size: int = 1600, overlap: int = 200) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    overlap = max(0, min(overlap, size // 2))
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    units: list[str] = []
    for line in text.split("\n"):
        line = line.rstrip()
        if len(line) <= size:
            units.append(line)
        else:
            units.extend(_hard_split(line, size))

    chunks: list[str] = []
    buf = ""
    for unit in units:
        candidate = f"{buf}\n{unit}" if buf else unit
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            # Start the overlap at a line break so a chunk never opens mid-value.
            if tail and "\n" in tail:
                tail = tail[tail.index("\n") + 1:]
            buf = f"{tail}\n{unit}".strip() if tail else unit
        else:
            buf = unit
    if buf:
        chunks.append(buf)
    return [c for c in (c.strip() for c in chunks) if c]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
