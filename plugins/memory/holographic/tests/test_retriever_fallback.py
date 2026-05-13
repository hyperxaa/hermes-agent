"""Tests for FactRetriever fallback behavior when numpy is unavailable."""

from __future__ import annotations

import pytest

from holographic import holographic as hrr
from holographic.retrieval import FactRetriever


def _make_no_numpy_retriever(store, patch_no_numpy):
    """Patch numpy off, then construct a fresh FactRetriever.

    A fresh constructor is needed because weight redistribution
    (fts=0.6, jaccard=0.4, hrr=0.0) only runs in __init__.
    """
    patch_no_numpy()
    return FactRetriever(store=store)


# ---------------------------------------------------------------------------
# 1. probe fallback to keyword search
# ---------------------------------------------------------------------------
def test_probe_fallback_to_search(store, seeded_facts, patch_no_numpy):
    retriever = _make_no_numpy_retriever(store, patch_no_numpy)
    result = retriever.probe("Xavi")
    assert isinstance(result, list)
    assert len(result) > 0
    contents = [r["content"] for r in result]
    assert any("Xavi" in c for c in contents)


# ---------------------------------------------------------------------------
# 2. related fallback to keyword search
# ---------------------------------------------------------------------------
def test_related_fallback_to_search(store, seeded_facts, patch_no_numpy):
    retriever = _make_no_numpy_retriever(store, patch_no_numpy)
    result = retriever.related("Xavi")
    assert isinstance(result, list)
    assert len(result) > 0
    contents = [r["content"] for r in result]
    assert any("Xavi" in c for c in contents)


# ---------------------------------------------------------------------------
# 3. reason fallback to FTS search
# ---------------------------------------------------------------------------
def test_reason_fallback_to_search(store, seeded_facts, patch_no_numpy):
    retriever = _make_no_numpy_retriever(store, patch_no_numpy)
    result = retriever.reason(["Xavi", "OCI"])
    assert isinstance(result, list)
    assert len(result) > 0
    contents = [r["content"] for r in result]
    assert any("Xavi" in c or "OCI" in c for c in contents)


# ---------------------------------------------------------------------------
# 4. contradict returns empty list
# ---------------------------------------------------------------------------
def test_contradict_returns_empty(store, seeded_facts, patch_no_numpy):
    retriever = _make_no_numpy_retriever(store, patch_no_numpy)
    result = retriever.contradict()
    assert result == []


# ---------------------------------------------------------------------------
# 5. search weights redistributed — fts=0.6, jaccard=0.4, hrr=0.0
# ---------------------------------------------------------------------------
def test_search_weights_redistributed(store, patch_no_numpy):
    retriever = _make_no_numpy_retriever(store, patch_no_numpy)
    assert retriever.fts_weight == pytest.approx(0.6)
    assert retriever.jaccard_weight == pytest.approx(0.4)
    assert retriever.hrr_weight == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. hrr weight set to 0.0 when no numpy
# ---------------------------------------------------------------------------
def test_hrr_weight_zero_no_numpy(store, patch_no_numpy):
    retriever = _make_no_numpy_retriever(store, patch_no_numpy)
    assert retriever.hrr_weight == 0.0


# ---------------------------------------------------------------------------
# 7. rebuild_all_vectors returns 0 when no numpy
# ---------------------------------------------------------------------------
def test_rebuild_all_vectors_noop(store, seeded_facts, patch_no_numpy):
    patch_no_numpy()
    store._hrr_available = False
    rebuilt = store.rebuild_all_vectors()
    assert rebuilt == 0


# ---------------------------------------------------------------------------
# 8. add_fact still works without numpy — HRR vector is NULL
# ---------------------------------------------------------------------------
def test_add_fact_works_without_numpy(store, patch_no_numpy):
    patch_no_numpy()
    store._hrr_available = False
    fid = store.add_fact("Numpy is not needed for this fact", category="test")
    assert fid is not None
    row = store._conn.execute(
        "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    assert row["hrr_vector"] is None


# ---------------------------------------------------------------------------
# 9. encode_text raises RuntimeError without numpy
# ---------------------------------------------------------------------------
def test_encode_text_raises_without_numpy(patch_no_numpy):
    patch_no_numpy()
    with pytest.raises(RuntimeError, match="numpy is required"):
        hrr.encode_text("hello world")


# ---------------------------------------------------------------------------
# 10. similarity raises RuntimeError without numpy
# ---------------------------------------------------------------------------
def test_similarity_raises_without_numpy(patch_no_numpy):
    patch_no_numpy()
    with pytest.raises(RuntimeError, match="numpy is required"):
        a = [0.0] * 16
        b = [0.0] * 16
        hrr.similarity(a, b)


# ---------------------------------------------------------------------------
# 11. bind raises RuntimeError without numpy
# ---------------------------------------------------------------------------
def test_bind_raises_without_numpy(patch_no_numpy):
    patch_no_numpy()
    with pytest.raises(RuntimeError, match="numpy is required"):
        a = [0.0] * 16
        b = [0.0] * 16
        hrr.bind(a, b)


# ---------------------------------------------------------------------------
# 12. encode_fact raises RuntimeError without numpy
# ---------------------------------------------------------------------------
def test_encode_fact_raises_without_numpy(patch_no_numpy):
    patch_no_numpy()
    with pytest.raises(RuntimeError, match="numpy is required"):
        hrr.encode_fact("some content", ["entity"], dim=16)
