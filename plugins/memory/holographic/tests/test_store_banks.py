"""7 tests for MemoryStore bank management (memory_banks table).

Covers:
  1. bank_created_on_fact_add
  2. bank_rebuilt_on_fact_update
  3. bank_rebuilt_on_fact_delete
  4. separate_banks_per_category
  5. bank_vector_dimension
  6. bank_fact_count_updated
  7. rebuild_all_vectors
"""

from __future__ import annotations

import struct

from holographic import holographic as hrr
from holographic.store import MemoryStore


def test_bank_created_on_fact_add(store: MemoryStore) -> None:
    """Bank exists after first fact added."""
    bank_name = "cat:general"
    rows = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM memory_banks WHERE bank_name = ?",
        (bank_name,),
    ).fetchone()
    assert rows["cnt"] == 0

    store.add_fact("Xavi works at OCI", category="general")

    row = store._conn.execute(
        "SELECT bank_name, fact_count FROM memory_banks WHERE bank_name = ?",
        (bank_name,),
    ).fetchone()
    assert row is not None, f"Bank {bank_name} should exist after adding a fact"
    assert row["fact_count"] >= 1


def test_bank_rebuilt_on_fact_update(store: MemoryStore) -> None:
    """Bank is updated when fact content changes."""
    if not hrr._HAS_NUMPY:
        from pytest import skip
        skip("numpy required for bank vector operations")

    store.add_fact("Alice works at Google", category="project")
    first_vec = store._conn.execute(
        "SELECT vector FROM memory_banks WHERE bank_name = 'cat:project'"
    ).fetchone()["vector"]
    assert first_vec is not None

    fid = store._conn.execute(
        "SELECT fact_id FROM facts WHERE content = 'Alice works at Google'"
    ).fetchone()["fact_id"]
    store.update_fact(fid, content="Alice works at Microsoft")

    second_vec = store._conn.execute(
        "SELECT vector FROM memory_banks WHERE bank_name = 'cat:project'"
    ).fetchone()["vector"]
    assert first_vec != second_vec, "Bank vector should change when a fact is updated"


def test_bank_rebuilt_on_fact_delete(store: MemoryStore) -> None:
    """Bank is updated after fact removal."""
    fid = store.add_fact("Bob likes Python", category="tool")
    row = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:tool'"
    ).fetchone()
    assert row is not None
    count_before = row["fact_count"]

    store.remove_fact(fid)

    # Bank should either be gone (if no facts remain) or have decremented count
    row_after = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:tool'"
    ).fetchone()
    if row_after is None:
        # Bank was deleted because no facts with vectors remain
        pass
    else:
        assert row_after["fact_count"] == count_before - 1


def test_separate_banks_per_category(store: MemoryStore) -> None:
    """Different categories create separate banks."""
    store.add_fact("User likes cats", category="user_pref")
    store.add_fact("Project uses Rust", category="project")

    cat_pref = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM memory_banks WHERE bank_name = 'cat:user_pref'"
    ).fetchone()["cnt"]
    cat_proj = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM memory_banks WHERE bank_name = 'cat:project'"
    ).fetchone()["cnt"]

    assert cat_pref == 1, "user_pref bank should exist"
    assert cat_proj == 1, "project bank should exist"
    assert cat_pref == cat_proj  # both should have same fact_count (1 each)


def test_bank_vector_dimension(store: MemoryStore) -> None:
    """Vector length in DB == dim * 8 bytes (float64)."""
    if not hrr._HAS_NUMPY:
        from pytest import skip
        skip("numpy required for vector dimension check")

    store.add_fact("Dimension test fact", category="general")
    row = store._conn.execute(
        "SELECT vector, dim FROM memory_banks WHERE bank_name = 'cat:general'"
    ).fetchone()
    assert row is not None

    vector_bytes = row["vector"]
    dim = row["dim"]
    # float64 = 8 bytes per element, so len should be dim * 8
    assert len(vector_bytes) == dim * 8, f"Expected {dim * 8} bytes, got {len(vector_bytes)}"

    # Verify it deserializes correctly
    phases = hrr.bytes_to_phases(vector_bytes)
    assert len(phases) == dim


def test_bank_fact_count_updated(store: MemoryStore) -> None:
    """fact_count reflects total facts in the category."""
    store.add_fact("fact one", category="general")
    store.add_fact("fact two", category="general")
    store.add_fact("fact three", category="general")

    row = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:general'"
    ).fetchone()
    assert row is not None
    assert row["fact_count"] == 3

    # Add a fourth fact
    store.add_fact("fact four", category="general")
    row2 = store._conn.execute(
        "SELECT fact_count FROM memory_banks WHERE bank_name = 'cat:general'"
    ).fetchone()
    assert row2["fact_count"] == 4


def test_rebuild_all_vectors(store: MemoryStore) -> None:
    """All facts with entities get HRR vectors, all banks rebuilt."""
    if not hrr._HAS_NUMPY:
        from pytest import skip
        skip("numpy required for rebuild_all_vectors")

    # Add facts across categories
    store.add_fact("Alice works at OpenAI on GPT models", category="project")
    store.add_fact("Bob prefers Python over Java", category="user_pref")
    store.add_fact("Carol tests with pytest", category="tool")

    # Clear all HRR vectors and delete all banks (simulate migration scenario)
    store._conn.execute("UPDATE facts SET hrr_vector = NULL")
    store._conn.execute("DELETE FROM memory_banks")
    store._conn.commit()

    # Verify everything is cleared
    null_count = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM facts WHERE hrr_vector IS NULL"
    ).fetchone()["cnt"]
    assert null_count == 3
    bank_count = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM memory_banks"
    ).fetchone()["cnt"]
    assert bank_count == 0

    # Rebuild all
    processed = store.rebuild_all_vectors()
    assert processed == 3

    # All facts now have vectors
    non_null = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM facts WHERE hrr_vector IS NOT NULL"
    ).fetchone()["cnt"]
    assert non_null == 3

    # All banks are rebuilt
    banks = store._conn.execute(
        "SELECT bank_name, fact_count FROM memory_banks"
    ).fetchall()
    bank_dict = {r["bank_name"]: r["fact_count"] for r in banks}
    assert "cat:project" in bank_dict
    assert "cat:user_pref" in bank_dict
    assert "cat:tool" in bank_dict
    assert bank_dict["cat:project"] == 1
    assert bank_dict["cat:user_pref"] == 1
    assert bank_dict["cat:tool"] == 1
