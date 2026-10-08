"""Chunking rules: line boundaries first, overlap, and oversized single lines."""
import pytest

from crawlerrag.rag.chunking import split_text, text_hash


def test_short_text_is_one_chunk():
    assert split_text("Title\n\nBody: short", size=200) == ["Title\n\nBody: short"]


def test_empty_text_yields_nothing():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_splits_on_line_boundaries_and_respects_size():
    text = "\n".join(f"Field {i}: {'x' * 40}" for i in range(20))
    chunks = split_text(text, size=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # No line was cut in half: every line of the original appears whole in some chunk.
    for line in text.split("\n"):
        assert any(line in c for c in chunks)


def test_overlap_repeats_previous_tail():
    text = "\n".join(f"Line {i}: {'y' * 30}" for i in range(12))
    chunks = split_text(text, size=160, overlap=60)
    assert len(chunks) > 2
    # Each chunk after the first starts with content that also appears in its predecessor.
    for previous, current in zip(chunks, chunks[1:]):
        first_line = current.split("\n")[0]
        assert first_line in previous


def test_overlap_never_exceeds_half_the_chunk():
    chunks = split_text("\n".join(f"L{i}" for i in range(200)), size=100, overlap=10_000)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) < 200          # an unbounded overlap would never advance


def test_oversized_single_line_is_split_on_sentences():
    line = "Ingredients: " + " ".join(f"ITEM {i} is present." for i in range(200))
    chunks = split_text(line, size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(c.count("ITEM") for c in chunks) == 200


def test_oversized_line_without_sentence_breaks_still_splits():
    chunks = split_text("Z" * 5000, size=400, overlap=0)
    assert len(chunks) >= 12
    assert all(len(c) <= 400 for c in chunks)
    assert "".join(chunks) == "Z" * 5000


def test_zero_size_is_rejected():
    with pytest.raises(ValueError):
        split_text("anything", size=0)


def test_hash_is_stable_and_content_dependent():
    assert text_hash("a") == text_hash("a")
    assert text_hash("a") != text_hash("b")
    assert len(text_hash("a")) == 64
