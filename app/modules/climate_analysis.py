
# app/modules/climate_analysis.py

import pandas as pd
import streamlit as st
import plotly.express as px


CLIMATE_COLUMNS = {
    "Precipitation_mm": "Precipitation (mm)",
    "Temp_Min_C": "Minimum Temperature (°C)",
    "Temp_Max_C": "Maximum Temperature (°C)",
    "Soil_Moisture_0_7cm": "Soil Moisture (0–7 cm)",
}


def prepare_climate_data(filtered, selected_district):
    """Prepare monthly case and climate summaries for the selected area."""

    data = filtered.copy()

    required = ["District", "Year", "Month", "Cases"]
    missing_required = [col for col in required if col not in data.columns]

    if missing_required:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_required)
        )

    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Month"] = pd.to_numeric(data["Month"], errors="coerce")
    data["Cases"] = pd.to_numeric(data["Cases"], errors="coerce")

    data = data.dropna(subset=["Year", "Month", "Cases"]).copy()
    data = data[data["Month"].between(1, 12)]

    for col in CLIMATE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if selected_district != "All districts":
        data = data[data["District"] == selected_district].copy()

    if data.empty:
        return data

    data["Date"] = pd.to_datetime(
        {
            "year": data["Year"].astype(int),
            "month": data["Month"].astype(int),
            "day": 1,
        },
        errors="coerce",
    )

    data = data.dropna(subset=["Date"])

    climate_cols = [c for c in CLIMATE_COLUMNS if c in data.columns]

    # Cases are summed for each month. Climate values are averaged
    # across districts when "All districts" is selected.
    aggregations = {"Cases": "sum"}
    aggregations.update({col: "mean" for col in climate_cols})

    monthly = (
        data.groupby("Date", as_index=False)
        .agg(aggregations)
        .sort_values("Date")
    )

    monthly["Year"] = monthly["Date"].dt.year
    monthly["Month"] = monthly["Date"].dt.month

    return monthly


def render_climate_analysis(filtered, start_year, end_year):
    st.title("Climate Analysis")

    st.write(
        f"Explore monthly climate patterns and their associations with "
        f"leptospirosis cases from {start_year} to {end_year}."
    )

    if filtered.empty:
        st.warning("No records are available for the selected year range.")
        return

    districts = sorted(filtered["District"].dropna().unique().tolist())

    selected_district = st.selectbox(
        "Select district",
        options=["All districts"] + districts,
        index=0,
        key="climate_analysis_district",
    )

    try:
        monthly = prepare_climate_data(filtered, selected_district)
    except ValueError as error:
        st.error(str(error))
        return

    if monthly.empty:
        st.warning("No data are available for this selection.")
        return

    available_climate = [
        col for col in CLIMATE_COLUMNS if col in monthly.columns
    ]

    if not available_climate:
        st.error(
            "No supported climate columns were found. Check the names in "
            "your consolidated CSV file."
        )
        return

    st.subheader("1. Summary statistics")

    total_cases = monthly["Cases"].sum()
    average_monthly_cases = monthly["Cases"].mean()

    metric_cols = st.columns(3)

    metric_cols[0].metric("Total cases", f"{total_cases:,.0f}")
    metric_cols[1].metric(
        "Average monthly cases",
        f"{average_monthly_cases:,.1f}",
    )
    metric_cols[2].metric(
        "Months analysed",
        f"{monthly['Date'].nunique():,}",
    )

    summary_rows = []

    for col in available_climate:
        values = monthly[col].dropna()

        summary_rows.append(
            {
                "Climate variable": CLIMATE_COLUMNS[col],
                "Mean": values.mean() if not values.empty else None,
                "Minimum": values.min() if not values.empty else None,
                "Maximum": values.max() if not values.empty else None,
                "Missing months": int(monthly[col].isna().sum()),
            }
        )

    st.dataframe(
        pd.DataFrame(summary_rows).round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("2. Monthly climate trends")

    selected_variable = st.selectbox(
        "Select climate variable",
        options=available_climate,
        format_func=lambda col: CLIMATE_COLUMNS[col],
        key="climate_trend_variable",
    )

    trend = monthly[["Date", selected_variable]].dropna()

    if not trend.empty:
        fig = px.line(
            trend,
            x="Date",
            y=selected_variable,
            markers=True,
            title=f"Monthly {CLIMATE_COLUMNS[selected_variable]}",
            labels={
                "Date": "Month",
                selected_variable: CLIMATE_COLUMNS[selected_variable],
            },
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid observations are available for this variable.")

    st.subheader("3. Climate and leptospirosis cases")

    scatter_variable = st.selectbox(
        "Choose a climate variable for comparison",
        options=available_climate,
        format_func=lambda col: CLIMATE_COLUMNS[col],
        key="climate_scatter_variable",
    )

    scatter_data = monthly[["Date", "Cases", scatter_variable]].dropna()

    if len(scatter_data) >= 2:
        fig = px.scatter(
            scatter_data,
            x=scatter_variable,
            y="Cases",
            hover_data=["Date"],
            trendline="ols",
            title=(
                f"{CLIMATE_COLUMNS[scatter_variable]} "
                "versus monthly cases"
            ),
            labels={
                scatter_variable: CLIMATE_COLUMNS[scatter_variable],
                "Cases": "Monthly leptospirosis cases",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        correlation = scatter_data["Cases"].corr(
            scatter_data[scatter_variable]
        )

        if pd.notna(correlation):
            st.metric("Pearson correlation", f"{correlation:.3f}")

        st.caption(
            "Correlation describes a linear association, not causation. "
            "Seasonality, reporting patterns, and other factors may affect "
            "this relationship."
        )
    else:
        st.info("At least two valid monthly observations are needed.")

    st.subheader("4. Correlation between cases and climate variables")

    correlation_columns = ["Cases"] + available_climate
    correlation_data = monthly[correlation_columns].corr(
        method="pearson",
        min_periods=3,
    )

    fig = px.imshow(
        correlation_data,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson correlation matrix",
        labels={"color": "Correlation"},
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Correlations are calculated from monthly summaries. For "
        "'All districts', cases are summed across districts while climate "
        "variables are averaged across available district records."
    )

    st.subheader("5. Lag analysis")

    st.write(
        "Explore whether climate values from earlier months are associated "
        "with cases in a later month. A lag of 1 means climate in the "
        "previous month is compared with cases in the current month."
    )

    max_lag = st.slider(
        "Maximum lag in months",
        min_value=0,
        max_value=6,
        value=3,
        key="climate_max_lag",
    )

    lag_variable = st.selectbox(
        "Climate variable for lag analysis",
        options=available_climate,
        format_func=lambda col: CLIMATE_COLUMNS[col],
        key="climate_lag_variable",
    )

    lag_data = monthly[["Date", "Cases", lag_variable]].copy()
    lag_results = []

    # Shift climate values forward so climate at time t-lag is compared
    # with cases at time t. Missing values are excluded, not filled with 0.
    for lag in range(max_lag + 1):
        comparison = pd.DataFrame(
            {
                "Cases": lag_data["Cases"],
                "Climate": lag_data[lag_variable].shift(lag),
            }
        ).dropna()

        corr = (
            comparison["Cases"].corr(comparison["Climate"])
            if len(comparison) >= 3
            else float("nan")
        )

        lag_results.append(
            {
                "Lag (months)": lag,
                "Correlation": corr,
                "Observations": len(comparison),
            }
        )

    lag_results = pd.DataFrame(lag_results)

    fig = px.bar(
        lag_results,
        x="Lag (months)",
        y="Correlation",
        hover_data=["Observations"],
        title=f"Lagged correlation: {CLIMATE_COLUMNS[lag_variable]}",
        range_y=[-1, 1],
    )
    fig.add_hline(y=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        lag_results.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Treat these results as exploratory evidence, not proof that a "
        "climate variable causes future cases. Investigate seasonality, "
        "autocorrelation, reporting delays, and appropriate lag choices "
        "before using climate features in the forecasting model."
    )

    st.subheader("6. Monthly data table")

    display_columns = ["Date", "Cases"] + available_climate
    st.dataframe(
        monthly[display_columns].sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
