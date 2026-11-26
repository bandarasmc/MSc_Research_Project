"""
app/modules/spatial_explainability.py
======================================
Spatial Graph & Explainability page.

Shows:
  - District adjacency matrix (heatmap)
  - Per-district degree / connectivity table
  - Spatial graph visualisation (networkx → plotly)
  - Graph validation summary
  - SHAP feature importance (placeholder until model artifacts exist)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "outputs" / "model"

ADJ_BIN_PATH   = MODEL_DIR / "adjacency_binary.npy"
ADJ_NORM_PATH  = MODEL_DIR / "adjacency_norm.npy"
DIST_PATH      = MODEL_DIR / "district_order.json"
FEAT_PATH      = MODEL_DIR / "feature_names.json"
SHAP_PATH      = MODEL_DIR / "shap_values.npy"          # created by run_shap.py
SHAP_FEAT_PATH = MODEL_DIR / "shap_feature_names.json"  # created by run_shap.py


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def _load_graph_artifacts():
    """Return (districts, adj_binary, adj_norm) or raise."""
    if not DIST_PATH.exists():
        raise FileNotFoundError(
            "district_order.json not found. "
            "Run scripts/prepare_data.py first."
        )
    if not ADJ_BIN_PATH.exists():
        raise FileNotFoundError(
            "adjacency_binary.npy not found. "
            "Run scripts/build_graph.py first."
        )
    with open(DIST_PATH, encoding="utf-8") as fh:
        districts: list[str] = json.load(fh)["districts"]
    adj_bin  = np.load(ADJ_BIN_PATH).astype(int)
    adj_norm = np.load(ADJ_NORM_PATH)
    return districts, adj_bin, adj_norm


def _load_shap_artifacts():
    """Return (shap_values, feature_names) if available, else None."""
    if not SHAP_PATH.exists() or not SHAP_FEAT_PATH.exists():
        return None, None
    shap_vals = np.load(SHAP_PATH)
    with open(SHAP_FEAT_PATH, encoding="utf-8") as fh:
        feat_names = json.load(fh)["features"]
    return shap_vals, feat_names


# ---------------------------------------------------------------------------
# Graph layout (fixed geographic positions, lat/lon in EPSG:4326)
# Used to place nodes on a map-like scatter, not a force-directed layout.
# Approximate district centroid coordinates (longitude, latitude).
# ---------------------------------------------------------------------------
DISTRICT_COORDS: dict[str, tuple[float, float]] = {
    "Ampara":        (81.67, 7.30),
    "Anuradhapura":  (80.40, 8.33),
    "Badulla":       (81.05, 6.99),
    "Batticaloa":    (81.70, 7.72),
    "Colombo":       (79.97, 6.92),
    "Galle":         (80.22, 6.03),
    "Gampaha":       (80.02, 7.08),
    "Hambantota":    (81.10, 6.12),
    "Jaffna":        (80.01, 9.66),
    "Kalutara":      (80.14, 6.59),
    "Kandy":         (80.64, 7.29),
    "Kegalle":       (80.35, 7.25),
    "Kilinochchi":   (80.40, 9.38),
    "Kurunegala":    (80.36, 7.49),
    "Mannar":        (79.90, 8.98),
    "Matale":        (80.62, 7.47),
    "Matara":        (80.54, 5.94),
    "Monaragala":    (81.35, 6.87),
    "Mullaitivu":    (80.81, 9.27),
    "Nuwara Eliya":  (80.78, 6.95),
    "Polonnaruwa":   (81.00, 7.94),
    "Puttalam":      (79.84, 7.96),
    "Ratnapura":     (80.40, 6.68),
    "Trincomalee":   (81.23, 8.59),
    "Vavuniya":      (80.50, 8.75),
}


def _build_network_figure(
    districts: list[str],
    adj: np.ndarray,
    highlight_district: str | None = None,
) -> go.Figure:
    """Build a Plotly figure of the district adjacency graph."""
    n = len(districts)
    xs = [DISTRICT_COORDS[d][0] for d in districts]
    ys = [DISTRICT_COORDS[d][1] for d in districts]
    degrees = adj.sum(axis=1).tolist()

    # Edge traces
    edge_x, edge_y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edge_x += [xs[i], xs[j], None]
                edge_y += [ys[i], ys[j], None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="#94a3b8"),
        hoverinfo="none",
        name="Shared boundary",
    )

    # Node colours: highlight selected district
    if highlight_district and highlight_district in districts:
        h_idx = districts.index(highlight_district)
        # neighbours of the selected node
        neighbours = set(np.where(adj[h_idx] == 1)[0].tolist())
        node_colors = []
        for i in range(n):
            if i == h_idx:
                node_colors.append("#ef4444")     # red  — selected
            elif i in neighbours:
                node_colors.append("#f97316")     # orange — neighbours
            else:
                node_colors.append("#3b82f6")     # blue  — other
    else:
        # Colour by degree
        node_colors = degrees

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(
            size=14,
            color=node_colors,
            colorscale="Blues" if highlight_district is None else None,
            showscale=(highlight_district is None),
            colorbar=dict(title="Degree") if highlight_district is None else None,
            line=dict(width=1, color="white"),
        ),
        text=districts,
        textposition="top center",
        textfont=dict(size=8),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Degree: %{customdata}<br>"
            "Lon: %{x:.2f}, Lat: %{y:.2f}<extra></extra>"
        ),
        customdata=degrees,
        name="District",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="District Adjacency Graph (shared land boundaries)",
        xaxis=dict(
            title="Longitude",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            showgrid=False,
            zeroline=False,
        ),
        height=600,
        hovermode="closest",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="#f8fafc",
    )
    return fig


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_spatial_explainability(filtered: pd.DataFrame, **_) -> None:
    """Render the Spatial Graph & Explainability page."""

    st.header("Spatial Graph & Explainability")

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    try:
        districts, adj_bin, adj_norm = _load_graph_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    n = len(districts)
    degrees = adj_bin.sum(axis=1).astype(int)
    n_edges = int(adj_bin.sum()) // 2

    # ------------------------------------------------------------------
    # Section 1 — Graph overview
    # ------------------------------------------------------------------
    st.subheader("1. Graph Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Districts (nodes)", n)
    col2.metric("Adjacency edges", n_edges)
    col3.metric("Mean degree", f"{degrees.mean():.2f}")
    col4.metric("Max degree", f"{int(degrees.max())} ({districts[int(degrees.argmax())]})")

    # BFS connectivity check
    visited: set[int] = set()
    queue = [0]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for nb in np.where(adj_bin[node] == 1)[0]:
            if nb not in visited:
                queue.append(int(nb))
    is_connected = len(visited) == n
    isolated = [districts[i] for i in range(n) if i not in visited]

    if is_connected:
        st.success(
            f"✅ Graph is **fully connected** — all {n} districts are reachable "
            "from each other through shared land boundaries."
        )
    else:
        st.error(
            f"❌ Graph has **{len(isolated)} isolated node(s)**: {isolated}"
        )

    # ------------------------------------------------------------------
    # Section 2 — Network visualisation
    # ------------------------------------------------------------------
    st.subheader("2. District Adjacency Network")

    highlight = st.selectbox(
        "Highlight a district (shows neighbours in orange)",
        options=["— none —"] + districts,
        index=0,
        key="graph_highlight",
    )
    selected = None if highlight == "— none —" else highlight

    fig_network = _build_network_figure(districts, adj_bin, selected)
    st.plotly_chart(fig_network, use_container_width=True)

    if selected:
        nb_names = [
            districts[j]
            for j in np.where(adj_bin[districts.index(selected)] == 1)[0]
        ]
        st.caption(
            f"**{selected}** (degree {degrees[districts.index(selected)]}) "
            f"shares a boundary with: {', '.join(sorted(nb_names))}."
        )

    # ------------------------------------------------------------------
    # Section 3 — Adjacency matrix heatmap
    # ------------------------------------------------------------------
    st.subheader("3. Binary Adjacency Matrix")

    with st.expander("Show adjacency heatmap", expanded=False):
        fig_heatmap = px.imshow(
            adj_bin,
            x=districts,
            y=districts,
            color_continuous_scale=[[0, "#f1f5f9"], [1, "#1d4ed8"]],
            aspect="auto",
            title="A[i, j] = 1 when districts i and j share a land boundary",
            labels=dict(color="Adjacent"),
        )
        fig_heatmap.update_xaxes(tickfont_size=8, tickangle=45)
        fig_heatmap.update_yaxes(tickfont_size=8)
        fig_heatmap.update_layout(height=600, coloraxis_showscale=False)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ------------------------------------------------------------------
    # Section 4 — Per-district connectivity table
    # ------------------------------------------------------------------
    st.subheader("4. District Connectivity")

    degree_df = pd.DataFrame({
        "District":  districts,
        "Degree":    degrees.tolist(),
        "Neighbours": [
            ", ".join(
                sorted(districts[j] for j in np.where(adj_bin[i] == 1)[0])
            )
            for i in range(n)
        ],
    }).sort_values("Degree", ascending=False).reset_index(drop=True)
    degree_df.index += 1   # 1-based display

    st.dataframe(degree_df, use_container_width=True, height=500)

    # ------------------------------------------------------------------
    # Section 5 — Normalised adjacency info
    # ------------------------------------------------------------------
    st.subheader("5. Normalised Adjacency (STGCN Input)")

    st.markdown(
        r"""
        The normalised adjacency matrix fed to the STGCN is:

        $$\hat{A} = D^{-1/2}(A + I)D^{-1/2}$$

        where $A$ is the binary adjacency matrix and $I$ is the identity
        (self-loops). This follows the symmetric normalisation of
        Kipf & Welling (2017).
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Value range min", f"{adj_norm.min():.4f}")
    c2.metric("Value range max", f"{adj_norm.max():.4f}")
    c3.metric("Symmetric", "Yes" if np.allclose(adj_norm, adj_norm.T, atol=1e-5) else "No")

    # ------------------------------------------------------------------
    # Section 6 — SHAP explainability
    # ------------------------------------------------------------------
    st.subheader("6. SHAP Feature Importance")

    shap_vals, shap_feat_names = _load_shap_artifacts()

    if shap_vals is None:
        st.info(
            "🔬 **SHAP explanations will appear here after model training.**\n\n"
            "Once `scripts/train_stgcn.py` and `scripts/run_shap.py` have been "
            "run, this section will show:\n"
            "- Global mean |SHAP| values per input feature\n"
            "- District-specific SHAP bar charts\n"
            "- A beeswarm summary plot\n\n"
            "SHAP explanations are computed using `GradientExplainer`, which "
            "is compatible with the PyTorch STGCN architecture and "
            "produces gradient-based attributions for each input feature."
        )
        st.markdown(
            """
            **Features that will be explained:**
            """
        )
        # Load feature list from artifact if available
        feat_path = MODEL_DIR / "feature_names.json"
        if feat_path.exists():
            with open(feat_path) as fh:
                feat_names = json.load(fh)["features"]
            for i, f in enumerate(feat_names):
                st.markdown(f"- `{f}`")
        return

    # --- SHAP results are available ---
    # shap_vals expected shape: (n_samples, n_districts, n_features, lookback)
    # Collapse to (n_features,) global mean |SHAP|
    mean_abs_shap = np.abs(shap_vals).mean(axis=tuple(range(shap_vals.ndim - 1)))
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=-1)

    shap_df = pd.DataFrame({
        "Feature":          shap_feat_names,
        "Mean |SHAP|":      mean_abs_shap,
    }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

    fig_shap = px.bar(
        shap_df,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title="Global Feature Importance — Mean Absolute SHAP Value",
        color="Mean |SHAP|",
        color_continuous_scale="Blues",
    )
    fig_shap.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig_shap, use_container_width=True)

    st.caption(
        "Mean |SHAP| values reflect the average contribution of each feature "
        "to the model's output across all training/validation samples. "
        "These are model attributions, not causal effects."
    )
