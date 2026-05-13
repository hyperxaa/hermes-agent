"""Tests for HRR algebra: encode, bind/unbind, bundle, similarity."""

import numpy as np
import pytest

from plugins.memory.holographic.holographic import (
    bundle,
    encode_atom,
    bind,
    similarity,
    unbind,
)


class TestEncodeAtom:
    """Tests for the encode_atom function."""

    def test_encode_atom_deterministic(self):
        """Given the same input word, encode_atom returns identical vectors."""
        v1 = encode_atom("foo")
        v2 = encode_atom("foo")
        assert np.array_equal(v1, v2)

    def test_encode_atom_different_inputs(self):
        """Given different input words, similarity is below 0.15."""
        v1 = encode_atom("foo")
        v2 = encode_atom("bar")
        assert similarity(v1, v2) < 0.15

    def test_encode_atom_phase_range(self):
        """Given any encoded atom, all phase values are in [0, 2π)."""
        v = encode_atom("hello")
        assert np.all(v >= 0) and np.all(v < 2 * np.pi)

    def test_encode_atom_dim_matches(self):
        """Given a dim parameter, the returned vector has that length."""
        v = encode_atom("test", dim=256)
        assert len(v) == 256


class TestBindUnbind:
    """Tests for bind and unbind operations."""

    def test_bind_unbind_roundtrip(self):
        """Given bound vector, unbinding with key recovers the other operand with sim >= 0.85."""
        a = encode_atom("alice")
        b = encode_atom("bob")
        bound = bind(a, b)
        recovered = unbind(bound, a)
        assert similarity(recovered, b) >= 0.85

    def test_bind_commutes(self):
        """Given two vectors, bind is commutative (element-wise addition mod 2π)."""
        a = encode_atom("foo")
        b = encode_atom("bar")
        assert np.allclose(bind(a, b), bind(b, a))


class TestBundle:
    """Tests for the bundle (superposition) operation."""

    def test_bundle_similarity_to_inputs(self):
        """Given a bundle of two vectors, its similarity to each input exceeds similarity to a random vector."""
        a = encode_atom("foo")
        b = encode_atom("bar")
        bundled = bundle(a, b)
        random_vec = np.random.RandomState(42).uniform(0, 2 * np.pi, size=len(a))
        sim_to_a = similarity(bundled, a)
        sim_to_random = similarity(bundled, random_vec)
        assert sim_to_a > sim_to_random

    def test_bundle_multi_similarity(self):
        """Given a bundle of three vectors, the bundle is similar to each input (>0)."""
        a = encode_atom("alice")
        b = encode_atom("bob")
        c = encode_atom("charlie")
        bundled = bundle(a, b, c)
        assert similarity(bundled, a) > 0
        assert similarity(bundled, b) > 0
        assert similarity(bundled, c) > 0


class TestSimilarity:
    """Tests for the similarity function."""

    def test_similarity_self_perfect(self):
        """Given a vector compared to itself, similarity is 1.0 ±1e-10."""
        v = encode_atom("self_test")
        assert abs(similarity(v, v) - 1.0) < 1e-10

    def test_similarity_orthogonal_near_zero(self):
        """Given two unrelated atoms at dim=1024, absolute similarity is below 0.10."""
        v1 = encode_atom("foo", dim=1024)
        v2 = encode_atom("bar", dim=1024)
        assert abs(similarity(v1, v2)) < 0.10

    def test_similarity_range(self):
        """Given any two vectors, similarity is always in [-1, 1]."""
        v1 = encode_atom("alpha")
        v2 = encode_atom("beta")
        s = similarity(v1, v2)
        assert -1.0 <= s <= 1.0


class TestAlgebraProperties:
    """Tests for emergent HRR algebraic properties."""

    def test_bind_quasi_orthogonal(self):
        """Given a binding result, it is dissimilar to both operands (<0.10)."""
        a = encode_atom("alice")
        b = encode_atom("bob")
        bound = bind(a, b)
        assert similarity(bound, a) < 0.10
        assert similarity(bound, b) < 0.10

    def test_dimension_scaling(self):
        """Given the same inputs, similarity at dim=64 is greater than at dim=1024 (noise falls with √dim — wait, actually orthogonality *improves* with dim, so |sim| shrinks. But the test says sim(dim=64) > sim(dim=1024). Let me verify: for unrelated atoms, mean cos(phase_diff) has variance ~1/dim so the absolute deviation from 0 shrinks. So |sim| at dim=64 > |sim| at dim=1024. We test absolute value to handle sign."""

        # The test plan says: sim(foo,bar) at dim=64 > sim at dim=1024
        # For random phase vectors, |cos diff| has std ~ 1/sqrt(dim)
        # So the magnitude of similarity is larger at lower dimensions.
        s64 = similarity(encode_atom("foo", dim=64), encode_atom("bar", dim=64))
        s1024 = similarity(encode_atom("foo", dim=1024), encode_atom("bar", dim=1024))
        assert abs(s64) > abs(s1024)
