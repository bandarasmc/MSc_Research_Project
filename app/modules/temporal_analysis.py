
import pandas as pd
import plotly.express as px
import streamlit as st


def render_temporal_analysis(filtered, start_year, end_year):

    st.title("Temporal Analysis")

    st.write(
        f"Explore monthly, seasonal and annual patterns in "
        f"reported leptospirosis cases from {start_year} to "
        f"{end_year}."
    )

    if filtered.empty:
        st.warning("No records are available for this period.")
        return

    # ==========================================
    # 1. PAGE-LEVEL DISTRICT FILTER
    # ==========================================

    district_options = [
        "All districts",
        *sorted(filtered["District"].dropna().unique().tolist())
    ]

    selected_district = st.selectbox(
        "Select district for temporal analysis",
        options=district_options,
        key="temporal_district"
    )

    if selected_district == "All districts":
        data = filtered.copy()
        scope_label = "All districts"
    else:
        data = filtered.loc[
            filtered["District"] == selected_district
        ].copy()
        scope_label = selected_district

    if data.empty:
        st.warning("No records match this selection.")
        return

    # ==========================================
    # 2. AGGREGATE MONTHLY AND ANNUAL TOTALS
    # ==========================================

    monthly_totals = (
        data.groupby("Date", as_index=False)
        .agg(Cases=("Cases", "sum"))
        .sort_values("Date")
    )

    annual_totals = (
        data.groupby("Year", as_index=False)
        .agg(Cases=("Cases", "sum"))
        .sort_values("Year")
    )

    monthly_totals["Month"] = monthly_totals["Date"].dt.month

    # ==========================================
    # 3. SUMMARY STATISTICS
    # ==========================================

    total_cases = data["Cases"].sum()
    number_of_months = data["Date"].nunique()
    average_monthly = (
        total_cases / number_of_months
        if number_of_months else 0
    )

    peak_month_row = monthly_totals.loc[
        monthly_totals["Cases"].idxmax()
    ]

    peak_month = peak_month_row["Date"].strftime("%b %Y")
    peak_cases = peak_month_row["Cases"]

    peak_year_row = annual_totals.loc[
        annual_totals["Cases"].idxmax()
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Reported Cases",
        f"{total_cases:,.0f}"
    )

    col2.metric(
        "Average Cases per Recorded Month",
        f"{average_monthly:,.1f}"
    )

    col3.metric(
        "Highest Monthly Total",
        f"{peak_cases:,.0f}",
        help=peak_month
    )

    col4.metric(
        "Highest Annual Total",
        f"{peak_year_row['Cases']:,.0f}",
        help=str(int(peak_year_row["Year"]))
    )

    st.caption(
        f"Analysis scope: {scope_label} | "
        f"Selected years: {start_year}–{end_year}"
    )

    # ==========================================
    # 4. MONTHLY TIME SERIES
    # ==========================================

    st.divider()
    st.subheader("Monthly Trend")

    fig_monthly = px.line(
        monthly_totals,
        x="Date",
        y="Cases",
        markers=True,
        labels={
            "Date": "Month",
            "Cases": "Reported cases"
        },
        title=f"Monthly reported cases — {scope_label}"
    )

    fig_monthly.update_layout(
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    # ==========================================
    # 5. MONTH-OF-YEAR SEASONAL PATTERN
    # ==========================================

    st.divider()
    st.subheader("Monthly Seasonality")

    month_names = [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]

    seasonal = (
        data.groupby("Month")["Cases"]
        .mean()
        .reindex(range(1, 13))
        .fillna(0)
        .reset_index(name="Average_Cases")
    )

    seasonal["Month_Name"] = seasonal["Month"].map(
        dict(enumerate(month_names, start=1))
    )

    fig_seasonal = px.bar(
        seasonal,
        x="Month_Name",
        y="Average_Cases",
        category_orders={"Month_Name": month_names},
        labels={
            "Month_Name": "Calendar month",
            "Average_Cases": "Average reported cases per recorded row"
        },
        title=f"Average monthly pattern — {scope_label}"
    )

    st.plotly_chart(
        fig_seasonal,
        use_container_width=True
    )

    st.caption(
        "Seasonal averages are calculated across the selected "
        "years. They describe recurring patterns, not causes."
    )

    # ==========================================
    # 6. YEAR-MONTH HEATMAP
    # ==========================================

    st.divider()
    st.subheader("Year-Month Heatmap")

    heatmap_data = data.pivot_table(
        index="Year",
        columns="Month",
        values="Cases",
        aggfunc="sum"
    )

    heatmap_data = heatmap_data.reindex(
        index=sorted(data["Year"].unique()),
        columns=range(1, 13)
    )

    heatmap_data.columns = month_names

    fig_heatmap = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={
            "x": "Month",
            "y": "Year",
            "color": "Reported cases"
        },
        title=f"Reported cases by year and month — {scope_label}"
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

    # ==========================================
    # 7. ANNUAL COMPARISON
    # ==========================================

    st.divider()
    st.subheader("Annual Comparison")

    fig_annual = px.bar(
        annual_totals,
        x="Year",
        y="Cases",
        labels={
            "Year": "Year",
            "Cases": "Total reported cases"
        },
        title=f"Annual reported cases — {scope_label}"
    )

    fig_annual.update_xaxes(
        tickmode="linear",
        dtick=1
    )

    st.plotly_chart(
        fig_annual,
        use_container_width=True
    )

    # ==========================================
    # 8. ANNUAL SUMMARY TABLE
    # ==========================================

    st.subheader("Annual Summary Table")

    st.dataframe(
        annual_totals.rename(columns={
            "Year": "Year",
            "Cases": "Total reported cases"
        }),
        use_container_width=True,
        hide_index=True
    )
