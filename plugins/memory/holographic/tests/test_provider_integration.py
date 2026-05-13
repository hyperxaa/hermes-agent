"""8 end-to-end integration tests for HolographicMemoryProvider.

Covers:
  1. e2e_add_and_search
  2. e2e_add_and_probe
  3. e2e_add_and_reason
  4. e2e_add_and_feedback
  5. e2e_add_and_list
  6. e2e_full_lifecycle
  7. e2e_provider_tool_dispatch
  8. e2e_memory_write_mirror
"""

from __future__ import annotations

import json
from pathlib import Path

from holographic import HolographicMemoryProvider
from holographic.store import MemoryStore
from holographic.retrieval import FactRetriever


def test_e2e_add_and_search(ephemeral_provider: HolographicMemoryProvider) -> None:
    """Add fact → search finds it."""
    result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Alice works at Oracle on cloud infrastructure",
        "category": "project",
    })
    data = json.loads(result)
    assert data["status"] == "added"
    assert "fact_id" in data

    search_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "search",
        "query": "Oracle cloud",
        "category": "project",
        "limit": 5,
    })
    data = json.loads(search_result)
    assert data["count"] >= 1
    assert any("Alice" in r["content"] for r in data["results"])


def test_e2e_add_and_probe(ephemeral_provider: HolographicMemoryProvider) -> None:
    """Add fact with entity → probe finds it."""
    ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Xavi lives in Valencia",
        "category": "user_pref",
    })

    probe_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "probe",
        "entity": "Xavi",
        "limit": 10,
    })
    data = json.loads(probe_result)
    assert data["count"] >= 1
    assert any("Valencia" in r["content"] for r in data["results"])


def test_e2e_add_and_reason(ephemeral_provider: HolographicMemoryProvider) -> None:
    """Multiple entities → reason finds intersection."""
    ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Xavi and Alice work on the Gateway project",
        "category": "project",
    })
    ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Xavi uses Rust for systems programming",
        "category": "tool",
    })
    ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Alice and Carlos love surfing",
        "category": "user_pref",
    })

    reason_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "reason",
        "entities": ["Xavi", "Alice"],
        "limit": 10,
    })
    data = json.loads(reason_result)
    assert data["count"] >= 1
    assert any("Gateway" in r["content"] for r in data["results"])


def test_e2e_add_and_feedback(ephemeral_provider: HolographicMemoryProvider) -> None:
    """Add → feedback → trust changes."""
    add_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Hermes runs on Oracle Cloud",
        "category": "project",
    })
    data = json.loads(add_result)
    fid = data["fact_id"]

    # Get initial trust
    facts = json.loads(ephemeral_provider.handle_tool_call("fact_store", {
        "action": "list",
        "limit": 10,
    }))
    initial_trust = next(f["trust_score"] for f in facts["facts"] if f["fact_id"] == fid)

    # Mark helpful
    feedback_result = ephemeral_provider.handle_tool_call("fact_feedback", {
        "action": "helpful",
        "fact_id": fid,
    })
    fb = json.loads(feedback_result)
    assert fb["new_trust"] > fb["old_trust"]

    # Mark unhelpful
    feedback_result = ephemeral_provider.handle_tool_call("fact_feedback", {
        "action": "unhelpful",
        "fact_id": fid,
    })
    fb = json.loads(feedback_result)
    assert fb["new_trust"] < fb["old_trust"]


def test_e2e_add_and_list(ephemeral_provider: HolographicMemoryProvider) -> None:
    """Add multiple → list returns all ordered by trust."""
    ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Python is great",
        "category": "tool",
    })
    ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Rust is fast",
        "category": "tool",
    })
    ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Hermes memory is cool",
        "category": "project",
    })

    # Boost one fact
    facts = json.loads(ephemeral_provider.handle_tool_call("fact_store", {
        "action": "list",
        "category": "tool",
        "limit": 10,
    }))
    assert facts["count"] >= 2
    # Both trust scores should be ~0.5 (default)
    assert all(abs(f["trust_score"] - 0.5) < 0.01 for f in facts["facts"][:2])

    # List all
    all_facts = json.loads(ephemeral_provider.handle_tool_call("fact_store", {
        "action": "list",
        "limit": 20,
    }))
    assert all_facts["count"] >= 3


def test_e2e_full_lifecycle(ephemeral_provider: HolographicMemoryProvider) -> None:
    """add → search → probe → update → feedback → search again."""
    # Add
    add_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Carlos loves surfing and uses Linux",
        "category": "user_pref",
    })
    data = json.loads(add_result)
    fid = data["fact_id"]
    assert data["status"] == "added"

    # Search
    search_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "search",
        "query": "surfing Linux",
        "limit": 5,
    })
    data = json.loads(search_result)
    assert data["count"] >= 1

    # Probe
    probe_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "probe",
        "entity": "Carlos",
        "limit": 5,
    })
    data = json.loads(probe_result)
    assert data["count"] >= 1

    # Update
    update_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "update",
        "fact_id": fid,
        "content": "Carlos loves surfing and uses Oracle Linux",
        "category": "user_pref",
    })
    data = json.loads(update_result)
    assert data["updated"] is True

    # Feedback
    feedback_result = ephemeral_provider.handle_tool_call("fact_feedback", {
        "action": "helpful",
        "fact_id": fid,
    })
    fb = json.loads(feedback_result)
    assert fb["new_trust"] > fb["old_trust"]

    # Search again — should still find with updated content
    search_result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "search",
        "query": "Oracle Linux",
        "limit": 5,
    })
    data = json.loads(search_result)
    assert data["count"] >= 1
    assert any("Oracle Linux" in r["content"] for r in data["results"])


def test_e2e_provider_tool_dispatch(ephemeral_provider: HolographicMemoryProvider) -> None:
    """Initialize provider → handle_tool_call → fact stored."""
    # Provider already initialized; dispatch an 'add' tool call
    result = ephemeral_provider.handle_tool_call("fact_store", {
        "action": "add",
        "content": "Dispatch test fact stored via tool call",
        "category": "general",
    })
    data = json.loads(result)
    assert data["status"] == "added"
    fid = data["fact_id"]

    # Verify the fact exists in the underlying store
    store = ephemeral_provider._store
    row = store._conn.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert row is not None
    assert row["content"] == "Dispatch test fact stored via tool call"


def test_e2e_memory_write_mirror(ephemeral_provider: HolographicMemoryProvider) -> None:
    """Provider.on_memory_write → fact exists."""
    # Simulate a memory write
    ephemeral_provider.on_memory_write(
        action="add",
        target="user",
        content="The user prefers dark mode in their IDE",
    )

    # Verify fact was stored
    facts = json.loads(ephemeral_provider.handle_tool_call("fact_store", {
        "action": "list",
        "category": "user_pref",
        "limit": 10,
    }))
    assert facts["count"] >= 1
    assert any(
        "dark mode" in f["content"] for f in facts["facts"]
    )

    # Also test with non-user target (goes to 'general')
    ephemeral_provider.on_memory_write(
        action="add",
        target="system",
        content="System uses SQLite for storage",
    )
    gen_facts = json.loads(ephemeral_provider.handle_tool_call("fact_store", {
        "action": "list",
        "category": "general",
        "limit": 10,
    }))
    assert any(
        "SQLite" in f["content"] for f in gen_facts["facts"]
    )
