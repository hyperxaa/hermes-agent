"""Tests for FTS5 trigger sync between the facts table and facts_fts virtual table.

Each test verifies that SQLite triggers correctly keep the FTS5 index in sync with
INSERT, DELETE, and UPDATE operations on the facts table.
"""

from __future__ import annotations

import pytest

from holographic.store import MemoryStore


@pytest.mark.usefixtures("store")
class TestFtsSync:
    """FTS5 trigger sync tests."""

    def test_fts_sync_on_insert(self, store: MemoryStore) -> None:
        """After add_fact, FTS5 contains the new fact."""
        fid = store.add_fact("The sky is blue", tags="weather")
        assert fid is not None

        results = store.search_facts("sky blue", limit=5)
        assert len(results) >= 1
        fact_contents = [r["content"] for r in results]
        assert "The sky is blue" in fact_contents

    def test_fts_sync_on_delete(self, store: MemoryStore) -> None:
        """After remove_fact, FTS5 no longer contains the deleted fact."""
        fid = store.add_fact("Delete me please", tags="test")
        assert fid is not None

        # Confirm it is there
        before = store.search_facts('"Delete me please"', limit=5)
        assert len(before) >= 1

        store.remove_fact(fid)

        after = store.search_facts('"Delete me please"', limit=5)
        assert len(after) == 0

    def test_fts_sync_on_update(self, store: MemoryStore) -> None:
        """After update_fact(content), FTS5 reflects the new text."""
        fid = store.add_fact("original content", tags="x")
        assert fid is not None

        # Old text searchable
        assert len(store.search_facts('"original content"', limit=5)) >= 1

        store.update_fact(fid, content="updated content here")

        # New text searchable
        new_results = store.search_facts('"updated content"', limit=5)
        assert len(new_results) >= 1
        assert any(r["content"] == "updated content here" for r in new_results)

        # Old text is gone
        old_results = store.search_facts('"original content"', limit=5)
        assert len(old_results) == 0

    def test_fts_matches_tags(self, store: MemoryStore) -> None:
        """Tag tokens are searchable in FTS5."""
        store.add_fact("A fact about tags", tags="important urgent")

        results_important = store.search_facts("important", limit=5)
        assert len(results_important) >= 1
        assert any(r["content"] == "A fact about tags" for r in results_important)

        results_urgent = store.search_facts("urgent", limit=5)
        assert len(results_urgent) >= 1
        assert any(r["content"] == "A fact about tags" for r in results_urgent)

    def test_fts_rebuild_after_bulk(self, store: MemoryStore) -> None:
        """INSERT INTO facts_fts(facts_fts) VALUES('rebuild') repopulates index."""
        facts = [
            ("alpha fact one", "a"),
            ("beta fact two", "b"),
            ("gamma fact three", "c"),
        ]
        for content, tag in facts:
            store.add_fact(content, tags=tag)

        # All three searchable before rebuild
        assert len(store.search_facts("alpha", limit=5)) >= 1
        assert len(store.search_facts("beta", limit=5)) >= 1
        assert len(store.search_facts("gamma", limit=5)) >= 1

        # FTS5 content= external-table rebuild via special command
        store._conn.execute(
            "INSERT INTO facts_fts(facts_fts) VALUES('rebuild')"
        )
        store._conn.commit()

        # After rebuild, all three should still be searchable
        assert len(store.search_facts("alpha", limit=5)) >= 1
        assert len(store.search_facts("beta", limit=5)) >= 1
        assert len(store.search_facts("gamma", limit=5)) >= 1
