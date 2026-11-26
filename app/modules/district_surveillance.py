
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_district_surveillance(filtered, start_year, end_year):

    st.title("District Surveillance")

    st.write(
        f"Explore reported leptospirosis cases across Sri Lanka "
        f"from {start_year} to {end_year}."
    )

    if filtered.empty:
        st.warning("No records are available for the selected years.")
        return

    # ==========================================
    # 1. PAGE-LEVEL DISTRICT SELECTION
    # ==========================================

    districts = sorted(filtered["District"].dropna().unique())

    selected_district = st.selectbox(
        "Select district to explore",
        options=districts,
        key="surveillance_district"
    )

    # Data for the selected district
    district_data = filtered.loc[
        filtered["District"] == selected_district
    ].copy()

    # ==========================================
    # 2. SELECTED-DISTRICT SUMMARY STATISTICS
    # ==========================================

    total_cases = district_data["Cases"].sum()
    months_recorded = district_data["Date"].nunique()

    average_monthly_cases = (
        total_cases / months_recorded
        if months_recorded > 0
        else 0
    )

    if not district_data.empty:
        monthly_selected = (
            district_data.groupby("Date", as_index=False)
            .agg(Cases=("Cases", "sum"))
            .sort_values("Date")
        )

        peak_row = monthly_selected.loc[
            monthly_selected["Cases"].idxmax()
        ]

        peak_month_label = peak_row["Date"].strftime("%b %Y")
        peak_month_cases = peak_row["Cases"]
    else:
        monthly_selected = pd.DataFrame(
            columns=["Date", "Cases"]
        )
        peak_month_label = "N/A"
        peak_month_cases = 0

    st.subheader(f"Summary Statistics — {selected_district}")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reported Cases",
        f"{total_cases:,.0f}"
    )

    col2.metric(
        "Average Cases per Recorded Month",
        f"{average_monthly_cases:,.1f}"
    )

    col3.metric(
        "Highest Monthly Total",
        f"{peak_month_cases:,.0f}",
        help=f"Peak month: {peak_month_label}"
    )

    st.caption(
        f"Selected district: {selected_district} | "
        f"Period: {start_year}–{end_year}. "
        "Counts are reported cases, not population-adjusted incidence."
    )

    # ==========================================
    # 3. MONTHLY TREND FOR SELECTED DISTRICT
    # ==========================================

    st.divider()
    st.subheader(
        f"Monthly Reported Cases — {selected_district}"
    )

    if not monthly_selected.empty:
        fig_trend = px.line(
            monthly_selected,
            x="Date",
            y="Cases",
            markers=True,
            labels={
                "Date": "Month",
                "Cases": "Reported cases"
            },
            title=f"Monthly trend: {selected_district}"
        )

        fig_trend.update_layout(
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )

    # ==========================================
    # 4. NATIONAL DISTRICT COMPARISON
    # ==========================================

    st.divider()
    st.subheader("Reported Cases by District")

    # Intentionally use all districts in the selected year range.
    district_summary = (
        filtered.groupby("District", as_index=False)
        .agg(
            Total_Cases=("Cases", "sum"),
            Average_Monthly_Cases=("Cases", "mean"),
            Months_Recorded=("Date", "nunique")
        )
        .sort_values("Total_Cases", ascending=False)
    )

    ranking = district_summary.sort_values(
        "Total_Cases",
        ascending=True
    )

    fig_ranking = px.bar(
        ranking,
        x="Total_Cases",
        y="District",
        orientation="h",
        labels={
            "Total_Cases": "Total reported cases",
            "District": "District"
        },
        title=(
            f"District comparison: {start_year}–{end_year}"
        )
    )

    fig_ranking.update_layout(
        height=650,
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig_ranking,
        use_container_width=True
    )

    st.subheader("District Summary Table")

    st.dataframe(
        district_summary.rename(columns={
            "Total_Cases": "Total reported cases",
            "Average_Monthly_Cases": "Average cases per recorded row",
            "Months_Recorded": "Months represented"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ==========================================
    # 5. FULL SRI LANKA DISTRICT MAP
    # ==========================================

    st.divider()
    st.subheader("Geographic Distribution of Reported Cases")

    project_root = Path(__file__).resolve().parents[2]
    geojson_path = (
        project_root / "data" / "spatial" / "lk.json"
    )

    if not geojson_path.exists():
        st.error(
            f"GeoJSON file not found: {geojson_path}. "
            "Check that lk.json is in data/spatial/."
        )
        return

    try:
        with open(geojson_path, "r", encoding="utf-8") as file:
            geojson = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Unable to read the GeoJSON file: {error}")
        return

    # GeoJSON district names -> CSV district names
    geojson_to_csv = {
        "Ampāra": "Ampara",
        "Anurādhapura": "Anuradhapura",
        "Badulla": "Badulla",
        "Gālla": "Galle",
        "Gampaha": "Gampaha",
        "Hambantŏṭa": "Hambantota",
        "Jaffna": "Jaffna",
        "Kaḷutara": "Kalutara",
        "Kægalla": "Kegalle",
        "Kilinŏchchi": "Kilinochchi",
        "Kŏḷamba": "Colombo",
        "Kuruṇægala": "Kurunegala",
        "Mahanuvara": "Kandy",
        "Mannārama": "Mannar",
        "Maḍakalapuva": "Batticaloa",
        "Mātale": "Matale",
        "Mātara": "Matara",
        "Mŏṇarāgala": "Monaragala",
        "Mulativ": "Mullaitivu",
        "Nuvara Ĕliya": "Nuwara Eliya",
        "Pŏḷŏnnaruva": "Polonnaruwa",
        "Puttalama": "Puttalam",
        "Ratnapura": "Ratnapura",
        "Trikuṇāmalaya": "Trincomalee",
        "Vavuniyāva": "Vavuniya",
        "Yāpanaya": "Jaffna",
    }

    # Standardize the names in the GeoJSON features.
    for feature in geojson.get("features", []):
        properties = feature.setdefault("properties", {})
        original_name = properties.get("name")

        if original_name:
            properties["district_english"] = geojson_to_csv.get(
                original_name,
                original_name
            )

    # Aggregate from the countrywide year-filtered data,
    # NOT district_data.
    map_data = (
        filtered.groupby("District", as_index=False)
        .agg(Total_Cases=("Cases", "sum"))
    )

    geojson_names = {
        feature.get("properties", {}).get("district_english")
        for feature in geojson.get("features", [])
    }

    unmatched = sorted(
        set(map_data["District"]) - geojson_names
    )

    if unmatched:
        st.warning(
            "CSV district names not matched to GeoJSON: "
            + ", ".join(unmatched)
        )

    map_data = map_data[
        map_data["District"].isin(geojson_names)
    ].copy()

    if map_data.empty:
        st.error(
            "No CSV district names matched the GeoJSON district names."
        )
        return

    fig_map = px.choropleth(
        map_data,
        geojson=geojson,
        locations="District",
        featureidkey="properties.district_english",
        color="Total_Cases",
        color_continuous_scale="YlOrRd",
        hover_name="District",
        hover_data={
            "Total_Cases": ":,.0f",
            "District": False
        },
        labels={
            "Total_Cases": "Reported cases"
        },
        title=(
            f"Reported Leptospirosis Cases by District "
            f"({start_year}–{end_year})"
        )
    )

    # Fit to the entire country rather than one selected district.
    fig_map.update_geos(
        fitbounds="locations",
        visible=False
    )

    fig_map.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        coloraxis_colorbar=dict(
            title="Reported cases"
        )
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

    st.caption(
        "The map always shows the countrywide distribution for "
        "the selected year range. Selecting a district above "
        "does not filter the map."
    )
