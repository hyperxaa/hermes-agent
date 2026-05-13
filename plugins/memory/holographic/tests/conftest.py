"""Shared pytest fixtures for holographic memory plugin tests.

Each test receives its own isolated SQLite database via tmp_path fixture,
ensuring zero cross-test contamination.

Fixtures:
    db_path          - Temp SQLite DB path (function scope)
    store            - MemoryStore(db_path, default_trust=0.5) (function scope)
    retriever        - FactRetriever(store) (function scope)
    provider         - HolographicMemoryProvider with isolated config (function scope)
    seeded_facts     - 5-10 facts across categories with known entities (function scope)
    patch_no_numpy   - Helper to monkeypatch set hrr._HAS_NUMPY = False (function scope)
    ephemeral_provider - Provider with isolated plugin config dict (function scope)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the holographic plugin is importable.
# The plugin lives as a package at plugins/memory/holographic/.
# We add plugins/memory to sys.path so `from holographic import ...` works.
_PLUGIN_DIR = Path("/home/ubuntu/.hermes/hermes-agent/plugins/memory")
if not _PLUGIN_DIR.exists():
    # Fallback: compute from workspace location
    _PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent / "hermes-agent" / "plugins" / "memory"
_mem_path = str(_PLUGIN_DIR)
if _mem_path not in sys.path:
    sys.path.insert(0, _mem_path)

from holographic import holographic as hrr
from holographic.store import MemoryStore
from holographic.retrieval import FactRetriever
from holographic import HolographicMemoryProvider


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a path to a temporary SQLite database file."""
    return tmp_path / "test_memory.db"


@pytest.fixture
def store(db_path: Path) -> MemoryStore:
    """MemoryStore backed by a temporary database with default_trust=0.5."""
    return MemoryStore(db_path=str(db_path), default_trust=0.5)


@pytest.fixture
def retriever(store: MemoryStore) -> FactRetriever:
    """FactRetriever wired to the temporary store."""
    return FactRetriever(store=store)


@pytest.fixture
def provider() -> HolographicMemoryProvider:
    """HolographicMemoryProvider initialized with a mock config dict.

    Uses an in-memory-like config so no YAML files are read.
    The db_path is left None so the store uses its own default;
    individual tests can re-initialize with a specific db_path if needed.
    """
    config = {
        "db_path": None,
        "auto_extract": False,
        "default_trust": 0.5,
        "min_trust_threshold": 0.3,
        "temporal_decay_half_life": 0,
        "hrr_dim": 1024,
        "hrr_weight": 0.3,
    }
    return HolographicMemoryProvider(config=config)


@pytest.fixture
def seeded_facts(store: MemoryStore) -> list[dict]:
    """Insert 8 facts across categories with known entities.

    Returns the list of inserted fact dicts (as returned by store.list_facts).
    """
    facts = [
        ("Xavi works at OCI on the gateway project", "project"),
        ("Xavi prefers Spanish and Valencian for casual conversation", "user_pref"),
        ("The Hermes agent runs on Oracle Cloud Linux", "project"),
        ("Jack is a debugging engineer persona with dry humor", "user_pref"),
        ("pytest is the testing framework used in this project", "tool"),
        ("OCI stands for Oracle Cloud Infrastructure", "general"),
        ("Xavi uses Telegram and Discord for messaging", "user_pref"),
        ("Gateway hooks are registered per platform adapter", "project"),
    ]
    for content, category in facts:
        store.add_fact(content, category=category)
    return store.list_facts(limit=50)


@pytest.fixture
def patch_no_numpy(monkeypatch):
    """Monkeypatch helper: call it to disable numpy, call again to re-enable.

    Usage:
        def test_something(patch_no_numpy):
            patch_no_numpy()   # numpy disabled
            # ... test fallback behavior ...
    """
    def _disable():
        monkeypatch.setattr(hrr, "_HAS_NUMPY", False)
        monkeypatch.setattr(
            "holographic.holographic._require_numpy",
            lambda: (_ for _ in ()).throw(
                RuntimeError("numpy is required for holographic operations")
            ),
        )

    return _disable


@pytest.fixture
def ephemeral_provider(tmp_path: Path) -> HolographicMemoryProvider:
    """Provider with a completely isolated plugin config dict and temp DB.

    No YAML reads — all config is passed explicitly.
    Automatically calls initialize() with a temp SQLite DB.
    """
    db = tmp_path / "ephemeral.db"
    config = {
        "db_path": str(db),
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
