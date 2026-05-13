"""32 lifecycle tests for HolographicMemoryProvider.

Tests every public method on the provider: name, is_available, initialize,
system_prompt_block, prefetch, sync_turn, get_tool_schemas, handle_tool_call,
on_memory_write, shutdown, get_config_schema, save_config.

Uses the conftest fixtures: ephemeral_provider, store, provider, db_path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from holographic import HolographicMemoryProvider


# ---------------------------------------------------------------------------
# 1–2: Basic properties
# ---------------------------------------------------------------------------

def test_provider_name(ephemeral_provider: HolographicMemoryProvider) -> None:
    """test 1: name == 'holographic'."""
    assert ephemeral_provider.name == "holographic"


def test_provider_is_available(ephemeral_provider: HolographicMemoryProvider) -> None:
    """test 2: always True."""
    assert ephemeral_provider.is_available() is True


# ---------------------------------------------------------------------------
# 3–4: initialize
# ---------------------------------------------------------------------------

def test_provider_initialize_creates_store(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 3: after init, ._store and ._retriever are set."""
    assert ephemeral_provider._store is not None
    assert ephemeral_provider._retriever is not None


def test_provider_initialize_expands_hermes_home(tmp_path: Path) -> None:
    """test 4: $HERMES_HOME expanded in db_path."""
    fake_home = str(tmp_path / "fake_home")
    os.environ["HERMES_HOME"] = fake_home

    config = {
        "db_path": "$HERMES_HOME/memory_store.db",
        "default_trust": 0.5,
        "min_trust_threshold": 0.3,
        "hrr_dim": 1024,
        "hrr_weight": 0.3,
        "temporal_decay_half_life": 0,
    }
    prov = HolographicMemoryProvider(config=config)
    prov.initialize(session_id="test-session")

    # The store's db_path should have $HERMES_HOME replaced
    assert "$HERMES_HOME" not in str(prov._store.db_path)
    assert fake_home in str(prov._store.db_path)

    prov.shutdown()


# ---------------------------------------------------------------------------
# 5–6: system_prompt_block
# ---------------------------------------------------------------------------

def test_provider_system_prompt_block_empty(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 5: returns 'empty' prompt when 0 facts."""
    block = ephemeral_provider.system_prompt_block()
    assert "empty" in block.lower() or "Empty" in block


def test_provider_system_prompt_block_has_facts(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 6: returns count summary when facts exist."""
    ephemeral_provider._store.add_fact("test fact one")
    ephemeral_provider._store.add_fact("test fact two")
    block = ephemeral_provider.system_prompt_block()
    assert "2 facts" in block


# ---------------------------------------------------------------------------
# 7–10: prefetch
# ---------------------------------------------------------------------------

def test_provider_prefetch_returns_top5(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 7: returns up to 5 results as formatted string."""
    for i in range(8):
        ephemeral_provider._store.add_fact(f"test fact number {i}")
    result = ephemeral_provider.prefetch("test fact")
    assert "Holographic Memory" in result
    lines = [l for l in result.split("\n") if l.startswith("-")]
    assert len(lines) <= 5


def test_provider_prefetch_empty_query(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 8: '' → ''."""
    assert ephemeral_provider.prefetch("") == ""


def test_provider_prefetch_no_results(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 9: no matching facts → ''."""
    ephemeral_provider._store.add_fact("something about apples")
    result = ephemeral_provider.prefetch("quantum physics")
    assert result == ""


def test_provider_prefetch_exception_returns_empty(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 10: any exception → '', never raises."""
    with patch.object(
        ephemeral_provider._retriever, "search", side_effect=RuntimeError("boom")
    ):
        result = ephemeral_provider.prefetch("anything")
    assert result == ""


# ---------------------------------------------------------------------------
# 11: sync_turn
# ---------------------------------------------------------------------------

def test_provider_sync_turn_noop(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 11: does nothing."""
    # Should not raise, should not mutate anything important
    ephemeral_provider.sync_turn("hello user", "hello assistant")


# ---------------------------------------------------------------------------
# 12: get_tool_schemas
# ---------------------------------------------------------------------------

def test_provider_get_tool_schemas(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 12: returns 2 schemas (fact_store + fact_feedback)."""
    schemas = ephemeral_provider.get_tool_schemas()
    assert len(schemas) == 2
    names = [s["name"] for s in schemas]
    assert "fact_store" in names
    assert "fact_feedback" in names


# ---------------------------------------------------------------------------
# 13–21: handle_tool_call → fact_store actions
# ---------------------------------------------------------------------------

def test_provider_handle_fact_store_add(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 13: routes to store.add_fact, returns JSON."""
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "add", "content": "Xavi likes Python"}
    )
    data = json.loads(result)
    assert data["status"] == "added"
    assert "fact_id" in data


def test_provider_handle_fact_store_search(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 14: routes to retriever.search, returns JSON with count."""
    ephemeral_provider._store.add_fact("Xavi likes Python")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "search", "query": "Xavi"}
    )
    data = json.loads(result)
    assert "results" in data
    assert "count" in data
    assert data["count"] >= 1


def test_provider_handle_fact_store_probe(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 15: routes to retriever.probe."""
    ephemeral_provider._store.add_fact("Xavi works at OCI on the gateway project")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "probe", "entity": "Xavi"}
    )
    data = json.loads(result)
    assert "results" in data


def test_provider_handle_fact_store_related(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 16: routes to retriever.related."""
    ephemeral_provider._store.add_fact("Xavi works at OCI on the gateway project")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "related", "entity": "OCI"}
    )
    data = json.loads(result)
    assert "results" in data


def test_provider_handle_fact_store_reason(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 17: routes to retriever.reason."""
    ephemeral_provider._store.add_fact("Xavi works at OCI on the gateway project")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "reason", "entities": ["Xavi", "OCI"]}
    )
    data = json.loads(result)
    assert "results" in data


def test_provider_handle_fact_store_contradict(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 18: routes to retriever.contradict."""
    ephemeral_provider._store.add_fact("Python is great")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "contradict"}
    )
    data = json.loads(result)
    assert "results" in data


def test_provider_handle_fact_store_update(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 19: routes to store.update_fact."""
    fid = ephemeral_provider._store.add_fact("Python is great")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "update", "fact_id": fid, "content": "Python is awesome"}
    )
    data = json.loads(result)
    assert data["updated"] is True


def test_provider_handle_fact_store_remove(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 20: routes to store.remove_fact."""
    fid = ephemeral_provider._store.add_fact("Remove me")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "remove", "fact_id": fid}
    )
    data = json.loads(result)
    assert data["removed"] is True


def test_provider_handle_fact_store_list(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 21: routes to store.list_facts."""
    ephemeral_provider._store.add_fact("List me one")
    result = ephemeral_provider.handle_tool_call(
        "fact_store", {"action": "list"}
    )
    data = json.loads(result)
    assert "facts" in data
    assert "count" in data


# ---------------------------------------------------------------------------
# 22–23: handle_tool_call → fact_feedback actions
# ---------------------------------------------------------------------------

def test_provider_handle_fact_feedback_helpful(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 22: routes to store.record_feedback(True)."""
    fid = ephemeral_provider._store.add_fact("Feedback test")
    result = ephemeral_provider.handle_tool_call(
        "fact_feedback", {"action": "helpful", "fact_id": fid}
    )
    data = json.loads(result)
    assert data["new_trust"] > data["old_trust"]


def test_provider_handle_fact_feedback_unhelpful(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 23: routes to store.record_feedback(False)."""
    fid = ephemeral_provider._store.add_fact("Feedback test two")
    result = ephemeral_provider.handle_tool_call(
        "fact_feedback", {"action": "unhelpful", "fact_id": fid}
    )
    data = json.loads(result)
    assert data["new_trust"] < data["old_trust"]


# ---------------------------------------------------------------------------
# 24: handle_tool_call → unknown tool
# ---------------------------------------------------------------------------

def test_provider_handle_unknown_tool(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 24: returns error via tool_error()."""
    result = ephemeral_provider.handle_tool_call(
        "nonexistent_tool", {}
    )
    data = json.loads(result)
    assert "error" in data


# ---------------------------------------------------------------------------
# 25–29: on_memory_write
# ---------------------------------------------------------------------------

def test_provider_on_memory_write_user_target(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 25: target='user' → category='user_pref'."""
    ephemeral_provider.on_memory_write("add", "user", "likes dark mode")
    facts = ephemeral_provider._store.list_facts()
    assert len(facts) == 1
    assert facts[0]["category"] == "user_pref"


def test_provider_on_memory_write_memory_target(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 26: target='memory' → category='general'."""
    ephemeral_provider.on_memory_write("add", "memory", "some memory note")
    facts = ephemeral_provider._store.list_facts()
    assert len(facts) == 1
    assert facts[0]["category"] == "general"


def test_provider_on_memory_write_only_adds(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 27: action='replace' → no-op."""
    ephemeral_provider.on_memory_write("replace", "user", "replaced content")
    facts = ephemeral_provider._store.list_facts()
    assert len(facts) == 0


def test_provider_on_memory_write_ignores_remove(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 28: action='remove' → no-op."""
    ephemeral_provider.on_memory_write("remove", "user", "should be removed")
    facts = ephemeral_provider._store.list_facts()
    assert len(facts) == 0


def test_provider_on_memory_write_empty_content(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 29: empty content → no fact created."""
    ephemeral_provider.on_memory_write("add", "user", "")
    facts = ephemeral_provider._store.list_facts()
    assert len(facts) == 0


# ---------------------------------------------------------------------------
# 30: shutdown
# ---------------------------------------------------------------------------

def test_provider_shutdown_clears_refs(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 30: ._store and ._retriever set to None."""
    assert ephemeral_provider._store is not None
    ephemeral_provider.shutdown()
    assert ephemeral_provider._store is None
    assert ephemeral_provider._retriever is None


# ---------------------------------------------------------------------------
# 31: get_config_schema
# ---------------------------------------------------------------------------

def test_provider_get_config_schema(
    ephemeral_provider: HolographicMemoryProvider,
) -> None:
    """test 31: returns 4 fields."""
    schema = ephemeral_provider.get_config_schema()
    assert len(schema) == 4
    keys = [s["key"] for s in schema]
    assert "db_path" in keys
    assert "auto_extract" in keys
    assert "default_trust" in keys
    assert "hrr_dim" in keys


# ---------------------------------------------------------------------------
# 32: save_config
# ---------------------------------------------------------------------------

def test_provider_save_config_writes_yaml(
    ephemeral_provider: HolographicMemoryProvider,
    tmp_path: Path,
) -> None:
    """test 32: persists to config.yaml format."""
    import yaml

    config_file = tmp_path / "config.yaml"
    values = {
        "db_path": "/custom/path.db",
        "auto_extract": "true",
    }
    ephemeral_provider.save_config(values, str(tmp_path))

    assert config_file.exists()
    with open(config_file, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert "plugins" in loaded
    assert "hermes-memory-store" in loaded["plugins"]
    assert loaded["plugins"]["hermes-memory-store"]["db_path"] == "/custom/path.db"
