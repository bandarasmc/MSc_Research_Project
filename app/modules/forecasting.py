"""
app/modules/forecasting.py
===========================
Forecasting page — STGCN district-level leptospirosis forecasting.

Before model training this page shows:
  - Experiment configuration
  - Data split timeline with case-volume chart
  - Per-district historical case trends

After model training (once outputs/model/ contains the prediction
and metrics artifacts) it additionally shows:
  - Overall MAE / RMSE on the test set
  - District-level error metrics table
  - Actual vs. predicted plots per district
  - Training history (loss curves)
  - Downloadable predictions
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

DIST_PATH       = MODEL_DIR / "district_order.json"
FEAT_PATH       = MODEL_DIR / "feature_names.json"
SPLIT_PATH      = MODEL_DIR / "split_indices.json"
DATE_PATH       = MODEL_DIR / "date_index.csv"
CASE_PATH       = MODEL_DIR / "case_matrix.npy"

# Artifacts created by train_stgcn.py
METRICS_PATH    = MODEL_DIR / "test_metrics.json"
PRED_PATH       = MODEL_DIR / "test_predictions.csv"
HISTORY_PATH    = MODEL_DIR / "training_history.json"
META_PATH       = MODEL_DIR / "experiment_metadata.json"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data
def _load_base_artifacts():
    """Load artifacts that must exist (produced by prepare_data.py)."""
    missing = [
        p for p in (DIST_PATH, FEAT_PATH, SPLIT_PATH, DATE_PATH, CASE_PATH)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing data artifacts (run scripts/prepare_data.py):\n"
            + "\n".join(str(p) for p in missing)
        )
    with open(DIST_PATH, encoding="utf-8") as fh:
        districts: list[str] = json.load(fh)["districts"]
    with open(FEAT_PATH, encoding="utf-8") as fh:
        features: list[str] = json.load(fh)["features"]
    with open(SPLIT_PATH, encoding="utf-8") as fh:
        splits: dict = json.load(fh)
    dates   = pd.read_csv(DATE_PATH, parse_dates=["date"])
    cases   = np.load(CASE_PATH)                  # (204, 25) float32
    return districts, features, splits, dates, cases


def _load_model_artifacts():
    """Return model artifacts if available, else all None."""
    if not METRICS_PATH.exists():
        return None, None, None, None
    with open(METRICS_PATH, encoding="utf-8") as fh:
        metrics: dict = json.load(fh)
    preds = pd.read_csv(PRED_PATH, parse_dates=["date"]) if PRED_PATH.exists() else None
    history: dict | None = None
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as fh:
            history = json.load(fh)
    meta: dict | None = None
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
    return metrics, preds, history, meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPLIT_COLOURS = {
    "Training":   "#3b82f6",
    "Validation": "#f59e0b",
    "Test":       "#ef4444",
}

_PHASE_LABEL = {
    "train": "Training",
    "val":   "Validation",
    "test":  "Test",
}


def _split_timeline_fig(
    dates: pd.DataFrame,
    cases: np.ndarray,
    splits: dict,
) -> go.Figure:
    """
    Line chart of total monthly cases across all 25 districts,
    with coloured background bands for each split.
    """
    total_cases = cases.sum(axis=1)   # (204,)
    df_plot = pd.DataFrame({
        "date":  dates["date"],
        "cases": total_cases,
    })

    fig = go.Figure()

    # Coloured background bands for each split
    for phase, (s, e) in splits.items():
        label = _PHASE_LABEL[phase]
        x0 = dates.iloc[s]["date"]
        x1 = dates.iloc[e]["date"]
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=_SPLIT_COLOURS[label],
            opacity=0.08,
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color=_SPLIT_COLOURS[label],
        )

    # Total case line
    fig.add_trace(go.Scatter(
        x=df_plot["date"],
        y=df_plot["cases"],
        mode="lines",
        name="Total cases (all districts)",
        line=dict(color="#1e3a5f", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:,} cases<extra></extra>",
    ))

    fig.update_layout(
        title="Total monthly leptospirosis cases — all 25 districts",
        xaxis_title="Month",
        yaxis_title="Total cases",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="#f8fafc",
    )
    return fig


def _per_district_case_fig(
    district: str,
    dates: pd.DataFrame,
    cases: np.ndarray,
    districts: list[str],
    splits: dict,
) -> go.Figure:
    """Line chart of monthly cases for a single district."""
    idx = districts.index(district)
    district_cases = cases[:, idx]

    fig = go.Figure()

    for phase, (s, e) in splits.items():
        label = _PHASE_LABEL[phase]
        fig.add_vrect(
            x0=dates.iloc[s]["date"],
            x1=dates.iloc[e]["date"],
            fillcolor=_SPLIT_COLOURS[label],
            opacity=0.10,
            layer="below",
            line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=dates["date"],
        y=district_cases,
        mode="lines",
        line=dict(color="#1e3a5f", width=1.5),
        hovertemplate="%{x|%b %Y}: %{y:.0f} cases<extra></extra>",
        name=district,
    ))

    fig.update_layout(
        title=f"{district} — monthly leptospirosis cases",
        xaxis_title="Month",
        yaxis_title="Cases",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        plot_bgcolor="#f8fafc",
    )
    return fig


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_forecasting(filtered: pd.DataFrame, **_) -> None:
    """Render the Forecasting page."""

    st.header("Forecasting — STGCN District-Level Model")

    # ------------------------------------------------------------------
    # Load base artifacts
    # ------------------------------------------------------------------
    try:
        districts, features, splits, dates, cases = _load_base_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    n_districts = len(districts)
    n_months    = len(dates)

    # ------------------------------------------------------------------
    # Section 1 — Experiment configuration
    # ------------------------------------------------------------------
    st.subheader("1. Experiment Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model**")
        st.markdown(
            "- Architecture: Spatiotemporal Graph Convolutional Network (STGCN)\n"
            "- Framework: PyTorch (CPU)\n"
            "- Graph: 25-node district adjacency (shared land boundaries)\n"
            "- Forecast horizon: 1 month ahead\n"
            "- Lookback window: 12 months\n"
        )

    with col2:
        st.markdown("**Input features** (per district, per month)")
        for feat in features:
            label = {
                "Cases":               "Cases (lagged)",
                "Precipitation_mm":    "Precipitation (mm)",
                "Temp_Min_C":          "Minimum temperature (°C)",
                "Temp_Max_C":          "Maximum temperature (°C)",
                "Soil_Moisture_0_7cm": "Soil moisture (0–7 cm)",
            }.get(feat, feat)
            st.markdown(f"- `{feat}` — {label}")

    st.markdown("**Training protocol**")
    col3, col4, col5 = st.columns(3)
    train_s, train_e = splits["train"]
    val_s,   val_e   = splits["val"]
    test_s,  test_e  = splits["test"]

    col3.info(
        f"🔵 **Training**\n\n"
        f"{dates.iloc[train_s]['date'].strftime('%b %Y')} – "
        f"{dates.iloc[train_e]['date'].strftime('%b %Y')}\n\n"
        f"{train_e - train_s + 1} months"
    )
    col4.warning(
        f"🟡 **Validation**\n\n"
        f"{dates.iloc[val_s]['date'].strftime('%b %Y')} – "
        f"{dates.iloc[val_e]['date'].strftime('%b %Y')}\n\n"
        f"{val_e - val_s + 1} months"
    )
    col5.error(
        f"🔴 **Test**\n\n"
        f"{dates.iloc[test_s]['date'].strftime('%b %Y')} – "
        f"{dates.iloc[test_e]['date'].strftime('%b %Y')}\n\n"
        f"{test_e - test_s + 1} months"
    )

    st.caption(
        "All feature scalers are fitted on the training period only. "
        "Validation data is used for early stopping and model selection. "
        "Test data is used only for final reporting."
    )

    # ------------------------------------------------------------------
    # Section 2 — Case history with split bands
    # ------------------------------------------------------------------
    st.subheader("2. Historical Case Volume")

    fig_timeline = _split_timeline_fig(dates, cases, splits)
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown(
        f"**Dataset summary:** {n_months} monthly observations "
        f"across {n_districts} districts "
        f"({dates.iloc[0]['date'].strftime('%b %Y')} – "
        f"{dates.iloc[-1]['date'].strftime('%b %Y')}). "
        "Coloured bands show the training (blue), validation (amber), "
        "and test (red) periods."
    )

    # ------------------------------------------------------------------
    # Section 3 — District-level case explorer
    # ------------------------------------------------------------------
    st.subheader("3. District Case History")

    selected_district = st.selectbox(
        "Select district",
        options=districts,
        index=0,
        key="forecast_district_selector",
    )

    fig_district = _per_district_case_fig(
        selected_district, dates, cases, districts, splits
    )
    st.plotly_chart(fig_district, use_container_width=True)

    # Quick stats for selected district
    d_idx    = districts.index(selected_district)
    d_cases  = cases[:, d_idx]
    tr_cases = d_cases[train_s:train_e + 1]
    vl_cases = d_cases[val_s:val_e + 1]
    te_cases = d_cases[test_s:test_e + 1]

    s1, s2, s3 = st.columns(3)
    s1.metric(
        "Training mean cases/month",
        f"{tr_cases.mean():.1f}",
        help="Mean monthly case count over the training period."
    )
    s2.metric(
        "Validation mean cases/month",
        f"{vl_cases.mean():.1f}",
    )
    s3.metric(
        "Test mean cases/month",
        f"{te_cases.mean():.1f}",
    )

    # ------------------------------------------------------------------
    # Section 4 — Model results (post-training)
    # ------------------------------------------------------------------
    st.subheader("4. Model Evaluation Results")

    metrics, preds, history, meta = _load_model_artifacts()

    if metrics is None:
        st.info(
            "📊 **Model results will appear here after training.**\n\n"
            "Run the following command to train the STGCN:\n"
            "```\n"
            ".venv\\Scripts\\python.exe scripts/train_stgcn.py\n"
            "```\n"
            "Once training completes, this section will show:\n"
            "- Overall MAE and RMSE on the 2024 test set\n"
            "- District-level error metrics table\n"
            "- Actual vs. predicted case counts\n"
            "- Training / validation loss curves\n"
            "- Experiment metadata (epochs, hyperparameters)\n"
            "- Downloadable prediction CSV"
        )
        return

    # --- Results available ---
    # Overall metrics
    st.markdown("#### Overall Test-Set Metrics (January–December 2024)")
    m1, m2 = st.columns(2)
    overall_mae  = metrics.get("overall_mae",  metrics.get("mae",  "—"))
    overall_rmse = metrics.get("overall_rmse", metrics.get("rmse", "—"))
    m1.metric("MAE (cases)", f"{overall_mae:.2f}" if isinstance(overall_mae, float) else overall_mae)
    m2.metric("RMSE (cases)", f"{overall_rmse:.2f}" if isinstance(overall_rmse, float) else overall_rmse)

    # District-level metrics table
    if "district_metrics" in metrics:
        st.markdown("#### District-Level Metrics")
        dist_df = pd.DataFrame(metrics["district_metrics"]).T
        dist_df.index.name = "District"
        dist_df = dist_df.reset_index().rename(
            columns={"mae": "MAE", "rmse": "RMSE"}
        )
        dist_df = dist_df.sort_values("MAE", ascending=False)
        st.dataframe(dist_df, use_container_width=True, height=400)

    # Actual vs Predicted
    if preds is not None:
        st.markdown("#### Actual vs. Predicted — Test Period")

        pred_district = st.selectbox(
            "Select district for prediction plot",
            options=districts,
            index=0,
            key="pred_district_selector",
        )

        d_preds = preds[preds["district"] == pred_district].sort_values("date")
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=d_preds["date"], y=d_preds["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1e3a5f"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=d_preds["date"], y=d_preds["predicted"],
            mode="lines+markers",
            name="STGCN predicted",
            line=dict(color="#ef4444", dash="dash"),
        ))
        fig_pred.update_layout(
            title=f"{pred_district} — Actual vs. Predicted (2024)",
            xaxis_title="Month",
            yaxis_title="Cases",
            height=350,
            hovermode="x unified",
            plot_bgcolor="#f8fafc",
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Download
        st.download_button(
            "⬇ Download all test predictions (CSV)",
            data=preds.to_csv(index=False).encode(),
            file_name="stgcn_test_predictions.csv",
            mime="text/csv",
        )

    # Training history
    if history:
        st.markdown("#### Training History")
        hist_df = pd.DataFrame({
            "Epoch":           range(1, len(history["train_loss"]) + 1),
            "Training loss":   history["train_loss"],
            "Validation loss": history.get("val_loss", [None] * len(history["train_loss"])),
        })
        fig_hist = px.line(
            hist_df, x="Epoch",
            y=["Training loss", "Validation loss"],
            title="Loss per epoch",
            color_discrete_map={
                "Training loss":   "#3b82f6",
                "Validation loss": "#f59e0b",
            },
        )
        fig_hist.update_layout(height=300, plot_bgcolor="#f8fafc")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Experiment metadata
    if meta:
        st.markdown("#### Experiment Metadata")
        with st.expander("Show configuration details"):
            st.json(meta)
