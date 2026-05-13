"""Tests that deliberately run without numpy.

These overlap with fallback tests but test the plugin-level no-numpy behavior.
Cover:
  1. no_numpy_store_add_fact
  2. no_numpy_store_list_facts
  3. no_numpy_retriever_search
  4. no_numpy_provider_handles
"""

from __future__ import annotations

from pathlib import Path

from holographic import holographic as hrr
from holographic.store import MemoryStore
from holographic.retrieval import FactRetriever
from holographic import HolographicMemoryProvider


def test_no_numpy_store_add_fact(db_path: Path, patch_no_numpy) -> None:
    """MemoryStore works without numpy, hrr_vector is NULL."""
    patch_no_numpy()

    # Create store — _hrr_available should be False
    store = MemoryStore(db_path=str(db_path), default_trust=0.5)
    assert store._hrr_available is False

    # Adding a fact should not raise
    fid = store.add_fact("Alice works at Google on machine learning", category="project")
    assert fid > 0

    # Verify hrr_vector is NULL (fallback path)
    row = store._conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert row["hrr_vector"] is None, "hrr_vector should be NULL without numpy"

    store.close()


def test_no_numpy_store_list_facts(db_path: Path, patch_no_numpy) -> None:
    """list_facts works without numpy."""
    patch_no_numpy()

    store = MemoryStore(db_path=str(db_path), default_trust=0.5)
    store.add_fact("Python is great for scripting", category="tool")
    store.add_fact("Rust is fast for systems", category="tool")
    store.add_fact("Alice lives in Paris", category="user_pref")

    # List all
    all_facts = store.list_facts(limit=50)
    assert len(all_facts) == 3

    # List by category
    tool_facts = store.list_facts(category="tool", limit=50)
    assert len(tool_facts) == 2
    assert all(f["category"] == "tool" for f in tool_facts)

    store.close()


def test_no_numpy_retriever_search(db_path: Path, patch_no_numpy) -> None:
    """FactRetriever.search works without numpy."""
    patch_no_numpy()

    store = MemoryStore(db_path=str(db_path), default_trust=0.5)
    store.add_fact("Python is great for scripting", category="tool")
    store.add_fact("Rust is fast for systems programming", category="tool")
    store.add_fact("Alice lives in Paris and works on AI", category="user_pref")

    retriever = FactRetriever(store=store)

    # Fallback search (no HRR, relies on FTS + Jaccard)
    results = retriever.search("Python scripting", limit=5)
    assert len(results) >= 1
    assert any("Python" in r["content"] for r in results)

    # Search for another term
    results2 = retriever.search("Rust systems", limit=5)
    assert len(results2) >= 1
    assert any("Rust" in r["content"] for r in results2)

    # Search with no matches
    results3 = retriever.search("quantum computing blockchain", limit=5)
    assert results3 == []

    store.close()


def _make_ephemeral_provider(db_path: Path) -> HolographicMemoryProvider:
    """Helper that returns a provider wired to the given DB path."""
    config = {
        "db_path": str(db_path),
        "auto_extract": False,
        "default_trust": 0.5,
        "min_trust_threshold": 0.3,
        "temporal_decay_half_life": 0,
        "hrr_dim": 1024,
        "hrr_weight": 0.3,
    }
    prov = HolographicMemoryProvider(config=config)
    prov.initialize(session_id="test-session")
    return prov


def test_no_numpy_provider_handles(db_path: Path, patch_no_numpy) -> None:
    """Provider handles operations with null HRR vectors."""
    patch_no_numpy()

    provider = _make_ephemeral_provider(db_path)

    # Add a fact
    import json
    result = provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Alice works at Google on AI",
        "category": "project",
    })
    data = json.loads(result)
    assert data["status"] == "added"
    fid = data["fact_id"]

    # Verify hrr_vector is NULL
    store = provider._store
    row = store._conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert row["hrr_vector"] is None

    # Search should still work
    search_result = provider.handle_tool_call("fact_store", {
        "action": "search",
        "query": "Google AI",
        "limit": 5,
    })
    data = json.loads(search_result)
    assert data["count"] >= 1

    # List should work
    list_result = provider.handle_tool_call("fact_store", {
        "action": "list",
        "limit": 10,
    })
    data = json.loads(list_result)
    assert data["count"] >= 1

    # Update should work
    update_result = provider.handle_tool_call("fact_store", {
        "action": "update",
        "fact_id": fid,
        "content": "Alice now works at Microsoft on AI",
    })
    data = json.loads(update_result)
    assert data["updated"] is True

    # Feedback should work
    fb_result = provider.handle_tool_call("fact_feedback", {
        "action": "helpful",
        "fact_id": fid,
    })
    fb_data = json.loads(fb_result)
    assert "new_trust" in fb_data

    # Probe should work
    probe_result = provider.handle_tool_call("fact_store", {
        "action": "probe",
        "entity": "Alice",
        "limit": 5,
    })
    data = json.loads(probe_result)
    assert data["count"] >= 1

    # on_memory_write should work
    provider.on_memory_write(
        action="add",
        target="user",
        content="User likes dark mode",
    )
    all_facts = json.loads(provider.handle_tool_call("fact_store", {
        "action": "list",
        "limit": 20,
    }))
    assert all_facts["count"] >= 2

    provider.shutdown()
