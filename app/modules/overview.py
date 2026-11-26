

import streamlit as st
import plotly.express as px


def render_overview(filtered, start_year, end_year):


    st.header("Research Overview")

    st.markdown("""
    This research investigates district-level spatiotemporal
    patterns in reported leptospirosis cases in Sri Lanka.

    The study combines disease surveillance data with
    environmental and demographic variables to explore
    patterns that may support district-level forecasting.

    The proposed modelling approach uses a Spatial-Temporal
    Graph Convolutional Network (STGCN), with districts
    represented as nodes and spatial relationships represented
    through a district adjacency graph.
    """)

    st.info(
        "Use the sidebar to explore different study periods "
        "and districts. The statistics and charts below update "
        "according to the selected filters."
    )

    # ---------------------------------------
    # 1. KEY PERFORMANCE INDICATORS
    # ---------------------------------------
    st.subheader("Study at a Glance")

    total_cases = filtered["Cases"].sum()
    number_districts = filtered["District"].nunique()
    number_years = filtered["Year"].nunique()
    number_months = filtered["Date"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Reported Cases",
        f"{total_cases:,.0f}"
    )

    col2.metric(
        "Districts Represented",
        f"{number_districts}"
    )

    col3.metric(
        "Years Represented",
        f"{number_years}"
    )

    col4.metric(
        "District-Month Records",
        f"{len(filtered):,}"
    )

    st.caption(
        "These are reported case counts, not population-adjusted "
        "incidence rates. All metrics reflect the selected filters."
    )

    # ---------------------------------------
    # 2. STUDY COVERAGE
    # ---------------------------------------
    st.divider()
    st.subheader("Study Coverage and Data Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Study coverage**")

        st.write(
            f"Selected period: {start_year}–{end_year}"
        )

        st.write(
            f"First observation: "
            f"{filtered['Date'].min():%B %Y}"
        )

        st.write(
            f"Last observation: "
            f"{filtered['Date'].max():%B %Y}"
        )

        st.write(
            f"Districts represented in selection: "
            f"{number_districts}"
        )

    with col2:
        st.markdown("**Data domains**")

        expected_variables = {
            "Disease surveillance": ["Cases"],
            "Climate": [
                "Precipitation_mm",
                "Temp_Min_C",
                "Temp_Max_C",
                "Soil_Moisture_0_7cm"
            ],
            "Demographic": ["Population"],
            "Agricultural": [
                "MRiceArea",
                "MRiceYield",
                "SRiceArea",
                "SRiceYield"
            ]
        }

        for domain, variables in expected_variables.items():
            available = [
                variable for variable in variables
                if variable in filtered.columns
            ]

            if available:
                st.write(
                    f"**{domain}:** "
                    + ", ".join(available)
                )

    # ---------------------------------------
    # 3. AUTOMATICALLY CALCULATED FINDINGS
    # ---------------------------------------
    st.divider()
    st.subheader("Descriptive Highlights")

    monthly_totals = (
        filtered.groupby("Date", as_index=False)
        .agg(Cases=("Cases", "sum"))
        .sort_values("Date")
    )

    district_totals = (
        filtered.groupby("District", as_index=False)
        .agg(Cases=("Cases", "sum"))
        .sort_values("Cases", ascending=False)
    )

    year_totals = (
        filtered.groupby("Year", as_index=False)
        .agg(Cases=("Cases", "sum"))
        .sort_values("Cases", ascending=False)
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**District with the highest case total**")

        if not district_totals.empty:
            top = district_totals.iloc[0]

            st.metric(
                label=str(top["District"]),
                value=f"{top['Cases']:,.0f} cases"
            )

            st.caption(
                "Highest cumulative reported cases within "
                "the current filter selection."
            )

    with col2:
        st.markdown("**Year with the highest case total**")

        if not year_totals.empty:
            top_year = year_totals.iloc[0]

            st.metric(
                label=str(int(top_year["Year"])),
                value=f"{top_year['Cases']:,.0f} cases"
            )

            st.caption(
                "Highest annual total among the selected years "
                "and other active filters."
            )

    if not monthly_totals.empty:
        peak_month = monthly_totals.loc[
            monthly_totals["Cases"].idxmax()
        ]

        st.write(
            f"**Highest monthly total:** "
            f"{peak_month['Date']:%B %Y} "
            f"({peak_month['Cases']:,.0f} reported cases)."
        )

    # ---------------------------------------
    # 4. OVERVIEW TREND
    # ---------------------------------------
    st.divider()
    st.subheader("Monthly Trend at a Glance")

    if not monthly_totals.empty:
        fig = px.line(
            monthly_totals,
            x="Date",
            y="Cases",
            markers=True,
            title="Monthly reported cases in the selected period"
        )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Reported cases",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------------------------------
    # 5. RESEARCH OBJECTIVES
    # ---------------------------------------
    st.divider()
    st.subheader("Research Objectives")

    st.markdown("""
    1. **Spatiotemporal analysis:** Characterize variation
       in reported leptospirosis cases across districts and time.

    2. **Environmental analysis:** Explore relationships
       between disease patterns and available climate,
       soil-moisture and demographic variables.

    3. **Forecasting:** Develop and evaluate a district-level
       STGCN forecasting model against an appropriate baseline.

    4. **Interpretability:** Examine the contribution of
       relevant inputs and spatial-temporal patterns after
       model validation.
    """)

    # ---------------------------------------
    # 6. RESEARCH PROGRESS
    # ---------------------------------------
    st.divider()
    st.subheader("Research Development Progress")

    progress = [
        ("Data consolidation and preprocessing", "Completed"),
        ("Exploratory data analysis", "Completed"),
        ("Interactive dashboard", "In progress"),
        ("Spatial adjacency graph", "Planned / in progress"),
        ("Baseline forecasting", "Planned / in progress"),
        ("STGCN training and evaluation", "Planned"),
        ("Model interpretation", "Planned")
    ]

    for task, status in progress:
        st.write(f"**{task}:** {status}")

    st.caption(
        "Update the progress labels to reflect the actual "
        "status of your research before the review meeting."
    )
