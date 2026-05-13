"""15 tests for text encoding, fact encoding, serialization, and SNR estimation.

Tests the higher-level encoding layer built on top of HRR algebra primitives.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from plugins.memory.holographic.holographic import (
    bind,
    bundle,
    bytes_to_phases,
    encode_atom,
    encode_fact,
    encode_text,
    phases_to_bytes,
    similarity,
    snr_estimate,
    unbind,
)

DIM = 1024
# Role atoms used internally by encode_fact
ROLE_CONTENT = "__hrr_role_content__"
ROLE_ENTITY = "__hrr_role_entity__"


# ---------------------------------------------------------------------------
# encode_text
# ---------------------------------------------------------------------------

def test_encode_text_single_token() -> None:
    """\"hello\" → bundled 1-token vector (same as encode_atom)."""
    v = encode_text("hello", dim=DIM)
    expected = encode_atom("hello", dim=DIM)
    assert np.allclose(v, expected)


def test_encode_text_multi_token() -> None:
    """\"hello world\" → 2 tokens bundled."""
    v = encode_text("hello world", dim=DIM)
    h = encode_atom("hello", dim=DIM)
    w = encode_atom("world", dim=DIM)
    expected = bundle(h, w)
    assert np.allclose(v, expected)


def test_encode_text_punctuation_stripped() -> None:
    """\"Hello, world!\" → same tokens as \"hello world\"."""
    v1 = encode_text("Hello, world!", dim=DIM)
    v2 = encode_text("hello world", dim=DIM)
    assert np.allclose(v1, v2)


def test_encode_text_empty_returns_empty_atom() -> None:
    """\"\" / \"   \" / \"...\" → encode_atom(\"__hrr_empty__\")."""
    expected = encode_atom("__hrr_empty__", dim=DIM)
    for empty in ["", "   ", "..."]:
        v = encode_text(empty, dim=DIM)
        assert np.allclose(v, expected), f"Failed for {empty!r}"


def test_encode_text_case_insensitive() -> None:
    """\"HELLO\" → same as \"hello\" (lowercased)."""
    v1 = encode_text("HELLO", dim=DIM)
    v2 = encode_text("hello", dim=DIM)
    assert np.allclose(v1, v2)


# ---------------------------------------------------------------------------
# encode_fact
# ---------------------------------------------------------------------------

def test_encode_fact_with_entities() -> None:
    """Structured encoding with role binding; fact vector encodes content and entities."""
    fact_vec = encode_fact("Xavi works at OCI", ["Xavi", "OCI"], dim=DIM)
    # The fact vector should be similar to the bundled components
    content_vec = encode_text("xavi works at oci", dim=DIM)
    role_content = encode_atom(ROLE_CONTENT, dim=DIM)
    content_bound = bind(content_vec, role_content)

    sim = similarity(fact_vec, content_bound)
    # Fact should be somewhat similar to its content component
    # Since it also includes entity components, sim won't be 1.0 but should be positive
    assert sim > 0.0, f"Expected positive similarity, got {sim}"


def test_encode_fact_no_entities() -> None:
    """Fact with empty entity list — just content bound to ROLE_CONTENT."""
    fact_vec = encode_fact("Hello world", [], dim=DIM)
    content_vec = encode_text("hello world", dim=DIM)
    role_content = encode_atom(ROLE_CONTENT, dim=DIM)
    extracted = unbind(fact_vec, role_content)
    sim = similarity(extracted, content_vec)
    assert sim > 0.10, f"Expected content similarity > 0.10, got {sim}"


def test_encode_fact_entity_lowercased() -> None:
    """Entity \"Xavi\" and \"xavi\" produce same fact vector."""
    v1 = encode_fact("Test fact", ["Xavi"], dim=DIM)
    v2 = encode_fact("Test fact", ["xavi"], dim=DIM)
    assert np.allclose(v1, v2)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_phases_to_bytes_roundtrip() -> None:
    """bytes_to_phases(phases_to_bytes(v)) ≈ v."""
    v = encode_atom("roundtrip_test", dim=DIM)
    serialized = phases_to_bytes(v)
    restored = bytes_to_phases(serialized)
    assert np.allclose(v, restored)


def test_serialization_size() -> None:
    """len(phases_to_bytes(v)) == dim * 8 bytes (float64)."""
    v = encode_atom("size_test", dim=DIM)
    data = phases_to_bytes(v)
    assert len(data) == DIM * 8


def test_serialization_mutable_copy() -> None:
    """Returned array from bytes_to_phases is writable."""
    v = encode_atom("mutable_test", dim=DIM)
    data = phases_to_bytes(v)
    restored = bytes_to_phases(data)
    # Should not raise — must be writable
    restored[0] = 0.0
    assert restored[0] == 0.0


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------

def test_snr_estimate_zero_items() -> None:
    """snr(dim, 0) == inf."""
    assert snr_estimate(DIM, 0) == float("inf")


def test_snr_estimate_decreases_with_items() -> None:
    """SNR shrinks as n_items grows."""
    snr_1 = snr_estimate(DIM, 1)
    snr_100 = snr_estimate(DIM, 100)
    snr_1000 = snr_estimate(DIM, 1000)
    assert snr_1 > snr_100 > snr_1000


def test_snr_warning_at_capacity(caplog: pytest.LogCaptureFixture) -> None:
    """Log warning when n_items > dim/4 (SNR < 2.0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import logging
        logger = logging.getLogger("holographic.holographic")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            logger.propagate = True
            snr_estimate(DIM, DIM)  # SNR = sqrt(1024/1024) = 1.0 < 2.0
            logger.propagate = False
    assert any("SNR" in msg and "near capacity" in msg for msg in caplog.messages), (
        f"Expected capacity warning, got: {caplog.messages}"
    )


# ---------------------------------------------------------------------------
# Algebraic extraction
# ---------------------------------------------------------------------------

def test_encode_fact_algebraic_extract() -> None:
    """Fact contains more than just content — entity binding affects structure."""
    content = "holographic memory tests"
    fact_with_entities = encode_fact(content, ["Holographic"], dim=DIM)
    fact_no_entities = encode_fact(content, [], dim=DIM)

    # With entities, fact vector should differ from content-only version
    sim = similarity(fact_with_entities, fact_no_entities)
    # They share the same content, so sim > 0 but entities add noise → not 1.0
    assert 0 < sim < 1.0, f"Expected 0 < sim < 1.0, got {sim}"
