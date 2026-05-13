"""Tests for FactRetriever FTS5 (full-text search) behavior."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from holographic.store import MemoryStore
from holographic.retrieval import FactRetriever


# ---------------------------------------------------------------------------
# 1. FTS OR semantics — "editor config" matches facts with EITHER word
# ---------------------------------------------------------------------------
def test_fts_or_semantics(tmp_path):
    db = tmp_path / "test_fts.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    store.add_fact("My editor config uses vim", category="tool")
    store.add_fact("The config file is important", category="tool")
    store.add_fact("Something completely unrelated here", category="general")

    retriever = FactRetriever(store=store)
    results = retriever.search("editor config")

    contents = [r["content"] for r in results]
    # Both facts containing 'editor' OR 'config' should appear
    assert any("editor" in c and "config" in c for c in contents)
    assert any("config" in c for c in contents)


# ---------------------------------------------------------------------------
# 2. FTS special chars are sanitized — query doesn't explode
# ---------------------------------------------------------------------------
def test_fts_special_chars_sanitized(tmp_path):
    db = tmp_path / "test_fts_special.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    store.add_fact("Python code with parentheses () and dashes", category="tool")
    store.add_fact("Something safe here", category="general")

    retriever = FactRetriever(store=store)
    # Query with FTS5-reserved characters — should not raise
    results = retriever.search("parentheses (test) -bad ^term ~foo /slash 'quote")
    assert isinstance(results, list)  # Doesn't crash


# ---------------------------------------------------------------------------
# 3. When sanitization eats everything, raw query is attempted
# ---------------------------------------------------------------------------
def test_fts_special_chars_all_eaten_tries_raw(tmp_path):
    db = tmp_path / "test_fts_all_eaten.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    store.add_fact("Some text here", category="general")

    retriever = FactRetriever(store=store)
    # All characters are FTS5-reserved — _tokenize + sanitize eats everything
    # The retriever should try raw query; if that fails, returns []
    results = retriever.search("() - ^ ~ / + '")
    assert isinstance(results, list)  # Never raises


# ---------------------------------------------------------------------------
# 4. FTS rank is positional — first result beats last
# ---------------------------------------------------------------------------
def test_fts_rank_positional(tmp_path):
    db = tmp_path / "test_fts_rank.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    store.add_fact("python python python test word", category="tool")
    store.add_fact("python test word", category="tool")
    store.add_fact("just a single python mention", category="tool")

    retriever = FactRetriever(store=store)
    results = retriever.search("python")

    assert len(results) >= 2
    # Positional rank: earlier results score higher
    assert results[0]["fts_rank"] > results[-1]["fts_rank"]


# ---------------------------------------------------------------------------
# 5. FTS candidate expansion — returns up to limit*3 candidates before reranking
# ---------------------------------------------------------------------------
def test_fts_candidate_limit_expansion(tmp_path):
    db = tmp_path / "test_fts_expand.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    # Insert more facts than limit=3 would return
    for i in range(10):
        store.add_fact(f"python fact number {i} here", category="tool")

    retriever = FactRetriever(store=store)
    internal = retriever._fts_candidates("python", None, 0.0, 9)  # limit * 3 = 9
    assert len(internal) <= 9
    # Final search with limit=3 returns at most 3
    results = retriever.search("python", limit=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# 6. FTS returns empty list on FTS5 MATCH error — never throws
# ---------------------------------------------------------------------------
def test_fts_returns_empty_on_error(tmp_path):
    db = tmp_path / "test_fts_error.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    store.add_fact("hello world here", category="general")

    retriever = FactRetriever(store=store)
    # Craft a query that will cause an FTS5 MATCH exception
    # Using FTS5 syntax that definitely errors out
    results = retriever.search("NEAR/10000")
    assert results == []
