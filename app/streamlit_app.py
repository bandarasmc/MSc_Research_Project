
from pathlib import Path

import pandas as pd
import streamlit as st

from modules.overview import render_overview
from modules.district_surveillance import render_district_surveillance
from modules.temporal_analysis import render_temporal_analysis
from modules.climate_analysis import render_climate_analysis
from modules.forecasting import render_forecasting
from modules.spatial_explainability import render_spatial_explainability
# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Leptospirosis Incidences in Sri Lanka - Knowledge Based Approach",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "leptospirosis_2008_2024.csv"
)


# ==================================================
# 3. LOAD AND VALIDATE DATA
# ==================================================

@st.cache_data
def load_data(path):

    df = pd.read_csv(path)

    required_columns = [
        "District", "Year", "Month", "Cases"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["District"] = (
        df["District"].astype("string").str.strip()
    )

    for col in ["Year", "Month", "Cases"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["District", "Year", "Month", "Cases"]
    ).copy()

    valid_calendar = (
        df["Year"].between(1900, 2100)
        & df["Month"].between(1, 12)
    )

    df = df.loc[valid_calendar].copy()

    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    df["Date"] = pd.to_datetime(
        {
            "year": df["Year"],
            "month": df["Month"],
            "day": 1
        },
        errors="coerce"
    )

    df = df.dropna(subset=["Date"]).copy()

    df = df.sort_values(
        ["Date", "District"]
    ).reset_index(drop=True)

    return df


# ==================================================
# 4. APPLICATION HEADER
# ==================================================

st.title("🦠 Sri Lanka Leptospirosis Dashboard")

st.caption(
    "An Interpretable Spatiotemporal Machine Learning "
    "Approach for District-Level Forecasting"
)

st.markdown(
    """
    Explore the temporal, spatial and environmental patterns
    of reported leptospirosis cases in Sri Lanka.
    """
)


# ==================================================
# 5. CHECK DATA AVAILABILITY
# ==================================================

if not DATA_PATH.exists():
    st.error(f"Dataset not found at:\n\n{DATA_PATH}")
    st.stop()

try:
    df = load_data(str(DATA_PATH))

except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if df.empty:
    st.error("No valid observations are available in the dataset.")
    st.stop()


# ==================================================
# 6. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("Research Dashboard")

page = st.sidebar.radio(
    "Navigate to",
    [
        "Overview",
        "District Surveillance",
        "Temporal Analysis",
        "Climate Analysis",
        "Forecasting",
        "Spatial Graph & Explainability"
    ],
    key="dashboard_page"
)


# ==================================================
# 7. GLOBAL FILTERS: YEAR ONLY
# ==================================================

st.sidebar.divider()
st.sidebar.subheader("Data Filters")

years = sorted(df["Year"].unique().tolist())

# From Year
start_year = st.sidebar.selectbox(
    "From Year",
    options=years,
    index=0,
    key="start_year"
)

# To Year must be >= From Year
end_year_options = [
    year for year in years
    if year >= start_year
]

# Use the latest available year by default.
# If From Year changes, reset To Year when necessary.
current_end_year = st.session_state.get("end_year")

if current_end_year not in end_year_options:
    default_end_year = end_year_options[-1]
else:
    default_end_year = current_end_year

end_year = st.sidebar.selectbox(
    "To Year",
    options=end_year_options,
    index=end_year_options.index(default_end_year),
    key="end_year"
)


# ==================================================
# 8. APPLY GLOBAL YEAR FILTER
# ==================================================

filtered = df.loc[
    df["Year"].between(start_year, end_year)
].copy()

if filtered.empty:
    st.warning(
        "No records match the selected year range. "
        "Please change the year filters."
    )
    st.stop()


# ==================================================
# 9. DISPLAY FILTER STATUS
# ==================================================

st.sidebar.divider()

st.sidebar.caption(
    f"Selected period: {start_year}–{end_year}"
)

st.sidebar.caption(
    f"Matching records: {len(filtered):,}"
)

st.sidebar.caption(
    f"Districts represented: {filtered['District'].nunique()}"
)


# ==================================================
# 10. PAGE ROUTING
# ==================================================

if page == "Overview":

    render_overview(
        filtered=filtered,
        start_year=start_year,
        end_year=end_year
    )


elif page == "District Surveillance":

    render_district_surveillance(
        filtered=filtered,
        start_year=start_year,
        end_year=end_year
    )


elif page == "Temporal Analysis":

    st.header("Temporal Analysis")

    render_temporal_analysis(
        filtered=filtered,
        start_year=start_year,
        end_year=end_year
    )

elif page == "Climate Analysis":

    st.header("Climate and Environmental Analysis")

    render_climate_analysis(
        filtered=filtered,
        start_year=start_year,
        end_year=end_year
    )


elif page == "Forecasting":

    render_forecasting(
        filtered=filtered,
        start_year=start_year,
        end_year=end_year,
    )


elif page == "Spatial Graph & Explainability":

    render_spatial_explainability(
        filtered=filtered,
        start_year=start_year,
        end_year=end_year,
    )


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.caption(
    "MSc Research Prototype | Sri Lanka Leptospirosis Analysis"
)


