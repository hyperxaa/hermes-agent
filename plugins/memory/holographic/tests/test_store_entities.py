"""14 entity extraction tests for MemoryStore._extract_entities.

Tests the regex-based entity extractor directly on the store module's
patterns, covering multi-word capitals, acronyms, quoted terms, AKA
patterns, stop-word exclusion, alias normalization, and known skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from holographic.store import MemoryStore


# ---------------------------------------------------------------------------
# Helper: create an in-memory (tmp file) store to call _extract_entities
# ---------------------------------------------------------------------------

def _extract(text: str, tmp_path: Path) -> list[str]:
    """Create a temporary MemoryStore and call _extract_entities."""
    db = tmp_path / "extract.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    try:
        return store._extract_entities(text)
    finally:
        store._conn.close()


# ---------------------------------------------------------------------------
# 1–5: Basic extraction patterns
# ---------------------------------------------------------------------------

def test_extract_multi_capitalized(tmp_path: Path) -> None:
    """test 1: 'Oracle Cloud' detected."""
    entities = _extract("Runs on Oracle Cloud infrastructure", tmp_path)
    assert "Oracle Cloud" in entities


def test_extract_acronym(tmp_path: Path) -> None:
    """test 2: 'OCI' detected."""
    entities = _extract("Works at OCI on gateway", tmp_path)
    assert "OCI" in entities


def test_extract_single_name(tmp_path: Path) -> None:
    """test 3: 'Xavi' detected."""
    entities = _extract("Xavi works here", tmp_path)
    assert "Xavi" in entities


def test_extract_double_quoted(tmp_path: Path) -> None:
    """test 4: 'Python' detected."""
    entities = _extract('The "Python" language is great', tmp_path)
    assert "Python" in entities


def test_extract_single_quoted(tmp_path: Path) -> None:
    """test 5: 'pytest' detected."""
    entities = _extract("Use 'pytest' for testing", tmp_path)
    assert "pytest" in entities


# ---------------------------------------------------------------------------
# 6: AKA pattern
# ---------------------------------------------------------------------------

def test_extract_aka_pattern(tmp_path: Path) -> None:
    """test 6: 'Guido aka BDFL' → two entities."""
    entities = _extract("Guido aka BDFL created Python", tmp_path)
    assert "Guido" in entities
    assert "BDFL" in entities


# ---------------------------------------------------------------------------
# 7–9: Stop words excluded
# ---------------------------------------------------------------------------

def test_stop_words_excluded_single(tmp_path: Path) -> None:
    """test 7: 'The' NOT extracted."""
    entities = _extract("The project uses Python", tmp_path)
    assert "The" not in entities


def test_stop_words_excluded_spanish(tmp_path: Path) -> None:
    """test 8: 'Cuando' NOT extracted."""
    entities = _extract("Cuando voy a casa", tmp_path)
    assert "Cuando" not in entities


def test_acronym_stop_words(tmp_path: Path) -> None:
    """test 9: 'EL', 'SI', 'NO' NOT extracted."""
    entities = _extract("EL SI NO", tmp_path)
    assert "EL" not in entities
    assert "SI" not in entities
    assert "NO" not in entities


# ---------------------------------------------------------------------------
# 10: Alias normalization
# ---------------------------------------------------------------------------

def test_entity_alias_normalization(tmp_path: Path) -> None:
    """test 10: 'Usuario Xavi' → 'Xavi'."""
    entities = _extract("Usuario Xavi is here", tmp_path)
    assert "Xavi" in entities
    assert "Usuario Xavi" not in entities


# ---------------------------------------------------------------------------
# 11–12: Multiple entities and reuse
# ---------------------------------------------------------------------------

def test_multiple_entities_one_fact(tmp_path: Path) -> None:
    """test 11: [Xavi, OCI] extracted from one fact."""
    entities = _extract("Xavi works at OCI", tmp_path)
    assert "Xavi" in entities
    assert "OCI" in entities


def test_reused_entity_shared_id(tmp_path: Path) -> None:
    """test 12: same entity name in 2 facts → same entity_id."""
    db = tmp_path / "reuse.db"
    store = MemoryStore(db_path=str(db), default_trust=0.5)
    try:
        fid1 = store.add_fact("Xavi says hi")
        fid2 = store.add_fact("Xavi says bye")

        # Extract entities for both facts
        entities1 = store._extract_entities("Xavi says hi")
        entities2 = store._extract_entities("Xavi says bye")
        assert "Xavi" in entities1
        assert "Xavi" in entities2

        # Both facts should link to the same entity_id
        rows = store._conn.execute(
            """
            SELECT fe.fact_id, e.name
            FROM fact_entities fe
            JOIN entities e ON e.entity_id = fe.entity_id
            ORDER BY fe.fact_id
            """
        ).fetchall()
        xavi_rows = [r for r in rows if r["name"] == "Xavi"]
        assert len(xavi_rows) == 2
        # Both should map to the same entity
        entity_ids = store._conn.execute(
            "SELECT entity_id FROM entities WHERE name = ?", ("Xavi",)
        ).fetchall()
        assert len(entity_ids) == 1
    finally:
        store._conn.close()


# ---------------------------------------------------------------------------
# 13–14: Edge cases
# ---------------------------------------------------------------------------

def test_extract_lowercase_only_skips(tmp_path: Path) -> None:
    """test 13: no capitalized entities (except quoted) in mixed-case text."""
    # All lowercase - no entities should be extracted
    entities = _extract("hello world lowercase only", tmp_path)
    assert len(entities) == 0


def test_extract_ignores_gateway(tmp_path: Path) -> None:
    """test 14: 'Gateway' NOT extracted."""
    entities = _extract("The Gateway project is active", tmp_path)
    # "Gateway" is in _STOPS, should not be extracted
    assert "Gateway" not in entities
