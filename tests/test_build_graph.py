"""
tests/test_build_graph.py
=========================
Unit tests for Task 3 (adjacency graph) artifacts.

Run from the project root::

    .venv/Scripts/python.exe -m pytest tests/test_build_graph.py -v
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT      = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "model"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def district_order():
    path = MODEL_DIR / "district_order.json"
    assert path.exists(), "district_order.json not found."
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["districts"]


@pytest.fixture(scope="session")
def adj_binary():
    path = MODEL_DIR / "adjacency_binary.npy"
    assert path.exists(), "adjacency_binary.npy not found; run build_graph.py first."
    return np.load(path)


@pytest.fixture(scope="session")
def adj_norm():
    path = MODEL_DIR / "adjacency_norm.npy"
    assert path.exists(), "adjacency_norm.npy not found."
    return np.load(path)


# ---------------------------------------------------------------------------
# Tests — binary adjacency
# ---------------------------------------------------------------------------

class TestBinaryAdjacency:
    def test_shape(self, adj_binary, district_order):
        n = len(district_order)
        assert adj_binary.shape == (n, n), (
            f"Binary adjacency shape {adj_binary.shape} ≠ ({n}, {n})"
        )

    def test_values_binary(self, adj_binary):
        unique = set(adj_binary.flatten().tolist())
        assert unique <= {0, 1}, (
            f"Binary adjacency contains non-binary values: {unique}"
        )

    def test_symmetric(self, adj_binary):
        assert np.array_equal(adj_binary, adj_binary.T), (
            "Binary adjacency matrix is not symmetric."
        )

    def test_no_self_loops(self, adj_binary):
        diag_sum = int(np.diag(adj_binary).sum())
        assert diag_sum == 0, (
            f"Binary adjacency diagonal sum = {diag_sum}; expected 0."
        )

    def test_fully_connected(self, adj_binary, district_order):
        """BFS from node 0 should reach all 25 nodes."""
        n = adj_binary.shape[0]
        visited: set[int] = set()
        queue = [0]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            neighbours = np.where(adj_binary[node] == 1)[0].tolist()
            queue.extend(nb for nb in neighbours if nb not in visited)
        isolated = [district_order[i] for i in range(n) if i not in visited]
        assert len(isolated) == 0, (
            f"Isolated districts (not reachable from node 0): {isolated}"
        )

    def test_expected_edge_count(self, adj_binary):
        # 58 undirected edges → 116 directed entries in symmetric matrix
        total_edges = int(adj_binary.sum())
        assert total_edges == 116, (
            f"Expected 116 directed edges (58 undirected), got {total_edges}"
        )

    def test_known_adjacencies(self, adj_binary, district_order):
        """Verify a sample of geographically known adjacencies."""
        idx = {d: i for i, d in enumerate(district_order)}
        expected_adjacent = [
            ("Colombo",       "Gampaha"),
            ("Colombo",       "Kalutara"),
            ("Kandy",         "Matale"),
            ("Jaffna",        "Kilinochchi"),
            ("Galle",         "Matara"),
            ("Anuradhapura",  "Kurunegala"),
        ]
        for a, b in expected_adjacent:
            i, j = idx[a], idx[b]
            assert adj_binary[i, j] == 1 and adj_binary[j, i] == 1, (
                f"Expected {a} ↔ {b} to be adjacent."
            )

    def test_known_non_adjacencies(self, adj_binary, district_order):
        """Verify that geographically distant districts are not adjacent."""
        idx = {d: i for i, d in enumerate(district_order)}
        non_adjacent = [
            ("Colombo", "Jaffna"),
            ("Galle",   "Jaffna"),
            ("Matara",  "Jaffna"),
        ]
        for a, b in non_adjacent:
            i, j = idx[a], idx[b]
            assert adj_binary[i, j] == 0 and adj_binary[j, i] == 0, (
                f"Expected {a} ↔ {b} NOT to be adjacent."
            )

    def test_degree_range(self, adj_binary):
        degrees = adj_binary.sum(axis=1)
        assert degrees.min() >= 1, (
            f"Minimum degree = {degrees.min()}; isolated node found."
        )
        assert degrees.max() <= 24, (
            f"Maximum degree = {degrees.max()}; exceeds possible maximum."
        )

    def test_district_ordering_matches_data(self, adj_binary, district_order):
        """District order used in graph must match district_order.json."""
        # Ratnapura is the highest-degree node (degree 9) according to the report.
        idx = {d: i for i, d in enumerate(district_order)}
        ratnapura_idx = idx["Ratnapura"]
        assert adj_binary[ratnapura_idx].sum() == 9, (
            f"Ratnapura should have degree 9, "
            f"got {adj_binary[ratnapura_idx].sum()}"
        )


# ---------------------------------------------------------------------------
# Tests — normalised adjacency
# ---------------------------------------------------------------------------

class TestNormAdjacency:
    def test_shape(self, adj_norm, district_order):
        n = len(district_order)
        assert adj_norm.shape == (n, n), (
            f"Norm adjacency shape {adj_norm.shape} ≠ ({n}, {n})"
        )

    def test_dtype_float32(self, adj_norm):
        assert adj_norm.dtype == np.float32, (
            f"Expected float32, got {adj_norm.dtype}"
        )

    def test_symmetric(self, adj_norm):
        assert np.allclose(adj_norm, adj_norm.T, atol=1e-6), (
            "Normalised adjacency is not symmetric."
        )

    def test_diagonal_positive(self, adj_norm):
        """Self-loops are added before normalisation, so diagonal should be > 0."""
        assert (np.diag(adj_norm) > 0).all(), (
            "Normalised adjacency diagonal contains zero or negative values; "
            "self-loops missing."
        )

    def test_values_in_01(self, adj_norm):
        assert adj_norm.min() >= 0.0 - 1e-6, "Norm adjacency has negative values."
        assert adj_norm.max() <= 1.0 + 1e-6, "Norm adjacency has values > 1."

    def test_no_nans(self, adj_norm):
        assert not np.isnan(adj_norm).any(), "Norm adjacency contains NaN."
