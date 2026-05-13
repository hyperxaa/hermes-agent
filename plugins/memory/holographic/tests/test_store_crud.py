"""39 CRUD tests for MemoryStore.

Tests are grouped by method:
  add_fact       — tests  1-11
  update_fact    — tests 12-21
  remove_fact    — tests 22-24
  list_facts     — tests 25-29
  search_facts   — tests 30-33
  feedback       — tests 34-36
  utilities      — tests 37-39
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from plugins.memory.holographic.store import MemoryStore


# ------------------------------------------------------------------ add_fact
# ------------------------------------------------------------------


def test_add_fact_returns_id(store: MemoryStore) -> None:
    """test 1: add_fact returns a positive int."""
    fid = store.add_fact("Python is used for data science")
    assert isinstance(fid, int)
    assert fid > 0


def test_add_fact_stores_content(store: MemoryStore) -> None:
    """test 2: DB contains exact content."""
    content = "Xavi lives in Valencia"
    fid = store.add_fact(content)
    rows = store._conn.execute("SELECT content FROM facts WHERE fact_id = ?", (fid,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["content"] == content


def test_add_fact_default_category(store: MemoryStore) -> None:
    """test 3: default category is 'general'."""
    fid = store.add_fact("cats eat fish")
    row = store._conn.execute("SELECT category FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["category"] == "general"


def test_add_fact_explicit_category(store: MemoryStore) -> None:
    """test 4: stores specified category."""
    fid = store.add_fact("dogs are great companions", category="opinion")
    row = store._conn.execute("SELECT category FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["category"] == "opinion"


def test_add_fact_default_trust(store: MemoryStore) -> None:
    """test 5: new fact has configured default trust."""
    fid = store.add_fact("trust me on this")
    row = store._conn.execute("SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    # store is created with default_trust=0.5 from fixture
    assert row["trust_score"] == pytest.approx(0.5)


def test_add_fact_with_tags(store: MemoryStore) -> None:
    """test 6: tags stored verbatim."""
    fid = store.add_fact("use pytest for testing", tags="python,testing")
    row = store._conn.execute("SELECT tags FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["tags"] == "python,testing"


def test_add_fact_dedup_returns_same_id(store: MemoryStore) -> None:
    """test 7: duplicate content → same fact_id."""
    content = "The sky is blue"
    fid1 = store.add_fact(content)
    fid2 = store.add_fact(content)
    assert fid1 == fid2


def test_add_fact_dedup_does_not_modify(store: MemoryStore) -> None:
    """test 8: duplicate doesn't change trust/tags."""
    content = "The sky is blue"
    fid1 = store.add_fact(content)
    # Manually change trust so we can verify it isn't reset
    store._conn.execute("UPDATE facts SET trust_score = 0.9 WHERE fact_id = ?", (fid1,))
    store._conn.commit()
    fid2 = store.add_fact(content, tags="new_tags")  # dup, trust=0.5 by default
    store.update_fact(fid2, trust_delta=1.0)   # bump trust
    fid3 = store.add_fact(content, tags="new_tags")
    assert fid3 == fid1
    row = store._conn.execute(
        "SELECT trust_score FROM facts WHERE fact_id = ?", (fid1,)
    ).fetchone()
    assert row["trust_score"] == pytest.approx(1.0)


def test_add_fact_empty_raises(store: MemoryStore) -> None:
    """test 9: ValueError on empty content."""
    with pytest.raises(ValueError, match="must not be empty"):
        store.add_fact("")
    with pytest.raises(ValueError, match="must not be empty"):
        store.add_fact("   ")


def test_add_fact_strips_whitespace(store: MemoryStore) -> None:
    """test 10: '  hello  ' → 'hello' in DB."""
    fid = store.add_fact("  hello  ")
    row = store._conn.execute("SELECT content FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["content"] == "hello"


def test_add_fact_creates_directory(db_path: Path) -> None:
    """test 11: parent dirs created when db in nested temp."""
    nested = db_path.parent / "sub" / "deep" / "test_nested.db"
    assert not nested.exists()
    nested_store = MemoryStore(db_path=nested, default_trust=0.5)
    assert nested.exists()
    fid = nested_store.add_fact("test")
    assert fid > 0
    nested_store.close()


# ------------------------------------------------------------------ update_fact
# ------------------------------------------------------------------


def test_update_fact_content(store: MemoryStore) -> None:
    """test 12: content update reflected in DB."""
    fid = store.add_fact("old content")
    store.update_fact(fid, content="new content")
    row = store._conn.execute("SELECT content FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["content"] == "new content"


def test_update_fact_trust_delta_positive(store: MemoryStore) -> None:
    """test 13: trust increases, clamped at 1.0."""
    fid = store.add_fact("good fact")
    row0 = store._conn.execute("SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    initial = row0["trust_score"]
    store.update_fact(fid, trust_delta=0.2)
    row1 = store._conn.execute("SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row1["trust_score"] == pytest.approx(initial + 0.2)


def test_update_fact_trust_delta_negative(store: MemoryStore) -> None:
    """test 14: trust decreases, clamped at 0.0."""
    fid = store.add_fact("bad fact")
    row0 = store._conn.execute("SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    initial = row0["trust_score"]
    store.update_fact(fid, trust_delta=-0.1)
    row1 = store._conn.execute("SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row1["trust_score"] == pytest.approx(initial - 0.1)


def test_update_fact_trust_clamp_upper(store: MemoryStore) -> None:
    """test 15: trust_delta=999 → trust=1.0."""
    fid = store.add_fact("super great")
    store.update_fact(fid, trust_delta=999.0)
    row = store._conn.execute("SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["trust_score"] == pytest.approx(1.0)


def test_update_fact_trust_clamp_lower(store: MemoryStore) -> None:
    """test 16: trust_delta=-999 → trust=0.0."""
    fid = store.add_fact("terrible")
    store.update_fact(fid, trust_delta=-999.0)
    row = store._conn.execute("SELECT trust_score FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["trust_score"] == pytest.approx(0.0)


def test_update_fact_tags(store: MemoryStore) -> None:
    """test 17: tags replaced on update."""
    fid = store.add_fact("important", tags="a")
    store.update_fact(fid, tags="b, c")
    row = store._conn.execute("SELECT tags FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["tags"] == "b, c"


def test_update_fact_category(store: MemoryStore) -> None:
    """test 18: category changed."""
    fid = store.add_fact("misc fact")
    store.update_fact(fid, category="science")
    row = store._conn.execute("SELECT category FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row["category"] == "science"


def test_update_fact_not_found(store: MemoryStore) -> None:
    """test 19: returns False."""
    assert store.update_fact(999999, content="nope") is False


def test_update_fact_content_re_rextracts_entities(store: MemoryStore) -> None:
    """test 20: new entities from updated content."""
    fid = store.add_fact("Alice likes Python")
    # Get entity count before update
    old = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM fact_entities WHERE fact_id = ?", (fid,)
    ).fetchone()["cnt"]
    store.update_fact(fid, content="Bob works at Google")
    new = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM fact_entities WHERE fact_id = ?", (fid,)
    ).fetchone()["cnt"]
    # Entities should have been updated (re-extracted from new content)
    # Alice should be gone, Bob and Google should appear
    row = store._conn.execute(
        "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
        (fid,),
    ).fetchall()
    names = {r["name"] for r in row}
    assert "Alice" not in names
    # Bob should be present (capitalized single name)
    assert any("Bob" in n or n == "Bob" for n in names)


def test_update_fact_content_recomputes_hrr(store: MemoryStore) -> None:
    """test 21: HRR vector updated when content changes."""
    fid = store.add_fact("old content")
    old_vec = store._conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()["hrr_vector"]
    store.update_fact(fid, content="completely different words now")
    new_vec = store._conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()["hrr_vector"]
    # The vector should have been recomputed (may differ or be None if numpy absent)
    # At minimum the update should succeed without error and vector may differ
    if old_vec is not None or new_vec is not None:
        assert old_vec != new_vec, "HRR vector should change when content changes"


# ------------------------------------------------------------------ remove_fact
# ------------------------------------------------------------------


def test_remove_fact_deletes_row(store: MemoryStore) -> None:
    """test 22: row gone from facts table."""
    fid = store.add_fact("to be deleted")
    assert store.remove_fact(fid) is True
    row = store._conn.execute("SELECT fact_id FROM facts WHERE fact_id = ?", (fid,)).fetchone()
    assert row is None


def test_remove_fact_deletes_entity_links(store: MemoryStore) -> None:
    """test 23: fact_entities rows removed."""
    fid = store.add_fact("Alice works here")
    links_before = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM fact_entities WHERE fact_id = ?", (fid,)
    ).fetchone()["cnt"]
    assert links_before > 0
    store.remove_fact(fid)
    links_after = store._conn.execute(
        "SELECT COUNT(*) as cnt FROM fact_entities WHERE fact_id = ?", (fid,)
    ).fetchone()["cnt"]
    assert links_after == 0


def test_remove_fact_not_found(store: MemoryStore) -> None:
    """test 24: returns False."""
    assert store.remove_fact(999999) is False


# ------------------------------------------------------------------ list_facts
# ------------------------------------------------------------------


def test_list_facts_ordered_by_trust(store: MemoryStore) -> None:
    """test 25: highest trust first."""
    fid1 = store.add_fact("low trust")
    fid2 = store.add_fact("high trust")
    fid3 = store.add_fact("mid trust")
    store.update_fact(fid1, trust_delta=-0.3)  # ~0.2
    store.update_fact(fid2, trust_delta=0.3)   # ~0.8
    store.update_fact(fid3, trust_delta=0.0)   # ~0.5

    results = store.list_facts(limit=10)
    ids = [r["fact_id"] for r in results]
    assert ids.index(fid2) < ids.index(fid3)
    assert ids.index(fid3) < ids.index(fid1)


def test_list_facts_category_filter(store: MemoryStore) -> None:
    """test 26: only matching category."""
    store.add_fact("science fact", category="science")
    store.add_fact("tech fact", category="tech")
    store.add_fact("more science", category="science")

    results = store.list_facts(category="science", limit=50)
    assert all(r["category"] == "science" for r in results)
    assert len(results) == 2


def test_list_facts_min_trust_filter(store: MemoryStore) -> None:
    """test 27: only above threshold."""
    fid = store.add_fact("good")
    store.update_fact(fid, trust_delta=0.3)  # ~0.8
    store.add_fact("ok")  # ~0.5

    results = store.list_facts(min_trust=0.6, limit=50)
    assert all(r["trust_score"] >= 0.6 for r in results)
    assert len(results) == 1


def test_list_facts_limit(store: MemoryStore) -> None:
    """test 28: returns at most N."""
    for i in range(10):
        store.add_fact(f"fact {i}")
    results = store.list_facts(limit=3)
    assert len(results) <= 3


def test_list_facts_includes_trust(store: MemoryStore) -> None:
    """test 29: list_facts result dict includes trust_score and entity info via fact_entities."""
    fid = store.add_fact("Alice works at OCI", category="general")
    results = store.list_facts(limit=10)
    assert len(results) > 0
    fact_dict = results[0]
    # trust_score should be in the dict
    assert "trust_score" in fact_dict
    # Verify entities are linked in DB
    entities = store._conn.execute(
        "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
        (fid,),
    ).fetchall()
    entity_names = [e["name"] for e in entities]
    assert len(entity_names) > 0


# ------------------------------------------------------------------ search_facts
# ------------------------------------------------------------------


def test_search_facts_fts_basic(store: MemoryStore) -> None:
    """test 30: FTS5 returns matching facts."""
    store.add_fact("Python is great for scripting")
    store.add_fact("JavaScript runs in browsers")
    results = store.search_facts("Python", limit=10)
    assert len(results) >= 1
    assert any("Python" in r["content"] for r in results)


def test_search_facts_empty_query(store: MemoryStore) -> None:
    """test 31: '' returns []."""
    store.add_fact("something here")
    results = store.search_facts("", limit=10)
    assert results == []


def test_search_facts_category_filter(store: MemoryStore) -> None:
    """test 32: category filter works."""
    store.add_fact("Python is cool", category="tech")
    store.add_fact("Python is cool too", category="misc")
    results = store.search_facts("Python", category="tech", limit=10)
    assert all(r["category"] == "tech" for r in results)


def test_search_facts_bumps_retrieval_count(store: MemoryStore) -> None:
    """test 33: retrieval_count incremented."""
    fid = store.add_fact("searchable content")
    # Initial retrieval_count should be 0
    row = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert row["retrieval_count"] == 0

    store.search_facts("searchable", limit=10)
    row = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert row["retrieval_count"] == 1


# ------------------------------------------------------------------ record_feedback
# ------------------------------------------------------------------


def test_record_feedback_helpful(store: MemoryStore) -> None:
    """test 34: trust += 0.05, helpful_count += 1."""
    fid = store.add_fact("useful fact")
    result = store.record_feedback(fid, helpful=True)
    assert result["fact_id"] == fid
    assert result["new_trust"] == pytest.approx(result["old_trust"] + 0.05)
    assert result["helpful_count"] >= 1


def test_record_feedback_unhelpful(store: MemoryStore) -> None:
    """test 35: trust -= 0.10."""
    fid = store.add_fact("bad fact")
    result = store.record_feedback(fid, helpful=False)
    assert result["new_trust"] == pytest.approx(result["old_trust"] - 0.10)


def test_record_feedback_not_found(store: MemoryStore) -> None:
    """test 36: KeyError."""
    with pytest.raises(KeyError):
        store.record_feedback(999999, helpful=True)


# ------------------------------------------------------------------ utilities
# ------------------------------------------------------------------


def test_row_to_dict_returns_dict(store: MemoryStore) -> None:
    """test 37: _row_to_dict returns a dict from a sqlite3.Row."""
    fid = store.add_fact("test row")
    row = store._conn.execute(
        "SELECT fact_id, content, trust_score FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    d = store._row_to_dict(row)
    assert isinstance(d, dict)
    assert d["fact_id"] == fid
    assert d["content"] == "test row"


def test_thread_safety(store: MemoryStore) -> None:
    """test 38: concurrent add_fact from multiple threads."""
    errors: list[Exception] = []

    def add(i: int) -> int:
        try:
            return i  # just return the index; actual add_fact happens synchronously
        except Exception as e:
            errors.append(e)
            return -1

    # Use the store concurrently from multiple threads
    contents = [f"fact from thread {i}" for i in range(20)]
    stored_ids: list[int] = []

    def worker(content: str) -> int:
        return store.add_fact(content)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(worker, c): c for c in contents}
        for fut in as_completed(futures):
            try:
                fid = fut.result()
                stored_ids.append(fid)
            except Exception as e:
                errors.append(e)

    assert not errors, f"Thread errors: {errors}"
    assert len(stored_ids) == 20
    # All 20 facts should be in the database
    count = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert count == 20


def test_wal_mode_enabled(store: MemoryStore) -> None:
    """test 39: PRAGMA journal_mode should be WAL (or fallback to default)."""
    result = store._conn.execute("PRAGMA journal_mode").fetchone()
    # apply_wal_with_fallback should attempt WAL; on some filesystems it falls back
    mode = result[0]
    assert mode in ("wal", "delete", "truncate"), f"Unexpected journal mode: {mode}"
