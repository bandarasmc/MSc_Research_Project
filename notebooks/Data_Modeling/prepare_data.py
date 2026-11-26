"""
notebooks/Data_Modeling/prepare_data.py
========================================
Task 2 — Build and validate the district-by-month dataset.

Outputs (all written to outputs/model/):
  district_order.json        — canonical list of 25 district names (alphabetical)
  feature_matrix.npy         — shape (204, 25, 5)  float32
                               axes: (month, district, feature)
                               features: [Cases, Precipitation_mm, Temp_Min_C,
                                          Temp_Max_C, Soil_Moisture_0_7cm]
  case_matrix.npy            — shape (204, 25) float32  — Cases only (convenience)
  date_index.csv             — 204 rows: month_idx, date string
  split_indices.json         — {train:[0,167], val:[168,191], test:[192,203]}
  data_quality_report.txt    — human-readable audit log

Run from the project root:
    .venv/Scripts/python.exe notebooks/Data_Modeling/prepare_data.py

Design decisions
----------------
* District order: alphabetical sort of the 25 names in the CSV.  This order is
  frozen into district_order.json; every other script loads it from there.
* Zero cases are genuine observations; they are retained as-is.
* Rice-area / rice-yield columns are excluded: they carry 36-348 nulls
  concentrated in 2-5 Northern Province districts, introducing district-
  unequal missingness into the graph node features.
* The 4 climate columns (Precipitation_mm, Temp_Min_C, Temp_Max_C,
  Soil_Moisture_0_7cm) have zero nulls across all 204*25 = 5100 cells.
* No scaling is applied here; scalers are fitted exclusively on training rows
  inside train_stgcn.py.
* Chronological split:
    Train : 2008-01 to 2021-12  -> month indices  0 - 167  (168 months)
    Val   : 2022-01 to 2023-12  -> month indices 168 - 191  ( 24 months)
    Test  : 2024-01 to 2024-12  -> month indices 192 - 203  ( 12 months)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at notebooks/Data_Modeling/, two levels below the root.
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH  = ROOT / "data" / "processed" / "leptospirosis_2008_2024.csv"
OUTPUT_DIR = ROOT / "outputs" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "data_quality_report.txt"

# ---------------------------------------------------------------------------
# Feature columns (order determines feature-axis in the saved tensor)
# ---------------------------------------------------------------------------
FEATURE_COLS: list[str] = [
    "Cases",
    "Precipitation_mm",
    "Temp_Min_C",
    "Temp_Max_C",
    "Soil_Moisture_0_7cm",
]

# ---------------------------------------------------------------------------
# Chronological split boundaries (inclusive, ISO month strings)
# ---------------------------------------------------------------------------
TRAIN_END  = "2021-12"
VAL_START  = "2022-01"
VAL_END    = "2023-12"
TEST_START = "2024-01"
TEST_END   = "2024-12"


# ===========================================================================
# Helpers
# ===========================================================================

def log(msg: str, sink: list[str]) -> None:
    """Print and accumulate a log line."""
    print(msg)
    sink.append(msg)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    report_lines: list[str] = []

    log("=" * 60, report_lines)
    log("DATA PREPARATION — leptospirosis_2008_2024.csv", report_lines)
    log("=" * 60, report_lines)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    log(f"\n[1] Loaded CSV: {len(df)} rows x {len(df.columns)} columns", report_lines)

    # ------------------------------------------------------------------
    # 2. Validate required columns
    # ------------------------------------------------------------------
    required = {"District", "Year", "Month"} | set(FEATURE_COLS)
    missing_cols = required - set(df.columns)
    if missing_cols:
        sys.exit(f"ERROR: missing columns: {sorted(missing_cols)}")
    log(f"[2] Required columns present: {sorted(required)}", report_lines)

    # ------------------------------------------------------------------
    # 3. Type coercion
    # ------------------------------------------------------------------
    df["District"] = df["District"].astype(str).str.strip()
    for col in ["Year", "Month"] + FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 4. Build Date column
    # ------------------------------------------------------------------
    df["Date"] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month"], day=1), errors="coerce"
    )
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates:
        log(f"WARNING: {invalid_dates} rows have invalid dates — dropped", report_lines)
        df = df.dropna(subset=["Date"])

    df = df.sort_values(["Date", "District"]).reset_index(drop=True)
    log(f"[3] Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}", report_lines)

    # ------------------------------------------------------------------
    # 5. District inventory
    # ------------------------------------------------------------------
    districts: list[str] = sorted(df["District"].unique().tolist())
    n_districts = len(districts)
    log(f"\n[4] Districts found: {n_districts}", report_lines)
    for i, d in enumerate(districts):
        log(f"    {i:2d}  {d}", report_lines)

    if n_districts != 25:
        log(f"WARNING: expected 25 districts, found {n_districts}", report_lines)

    # ------------------------------------------------------------------
    # 6. Build continuous monthly date index (2008-01 to 2024-12)
    # ------------------------------------------------------------------
    all_dates = pd.date_range(
        start=df["Date"].min(), end=df["Date"].max(), freq="MS"
    )
    n_months = len(all_dates)
    log(f"\n[5] Monthly index: {n_months} periods "
        f"({all_dates[0].date()} -> {all_dates[-1].date()})", report_lines)

    # ------------------------------------------------------------------
    # 7. Duplicate check
    # ------------------------------------------------------------------
    dups = df.duplicated(subset=["District", "Year", "Month"], keep=False).sum()
    log(f"\n[6] Duplicate (District, Year, Month) rows: {dups}", report_lines)
    if dups > 0:
        log("    ERROR: duplicates must be resolved before continuing.", report_lines)
        sys.exit("Duplicate district-month records found.")

    # ------------------------------------------------------------------
    # 8. Missing-value audit
    # ------------------------------------------------------------------
    log("\n[7] Null counts per feature column:", report_lines)
    for col in FEATURE_COLS:
        n_null = df[col].isna().sum()
        flag = " <- WARNING" if n_null > 0 else ""
        log(f"    {col:30s}: {n_null}{flag}", report_lines)

    # ------------------------------------------------------------------
    # 9. Negative-value check
    # ------------------------------------------------------------------
    neg_cases = (df["Cases"] < 0).sum()
    log(f"\n[8] Negative case counts: {neg_cases}", report_lines)
    if neg_cases > 0:
        log("    ERROR: negative case counts are not expected.", report_lines)
        sys.exit("Negative case counts found.")

    zero_cases = (df["Cases"] == 0).sum()
    log(f"    Zero-case records (genuine observations): {zero_cases}", report_lines)

    # ------------------------------------------------------------------
    # 10. Coverage: every district x every month must exist
    # ------------------------------------------------------------------
    log("\n[9] Checking coverage (every district x every month)...", report_lines)
    full_index = pd.MultiIndex.from_product(
        [districts, all_dates], names=["District", "Date"]
    )
    df_indexed = df.set_index(["District", "Date"])
    missing_cells = full_index.difference(df_indexed.index)

    if len(missing_cells) == 0:
        log("    Coverage: complete — no missing district-month cells.", report_lines)
    else:
        log(f"    WARNING: {len(missing_cells)} missing district-month cells:", report_lines)
        for dist, date in list(missing_cells)[:20]:
            log(f"      {dist}  {date.date()}", report_lines)

    # ------------------------------------------------------------------
    # 11. Pivot each feature into (n_months, n_districts) matrix
    # ------------------------------------------------------------------
    log("\n[10] Building feature matrices...", report_lines)
    feature_arrays: list[np.ndarray] = []

    for col in FEATURE_COLS:
        pivot = (
            df.pivot(index="Date", columns="District", values=col)
            .reindex(index=all_dates, columns=districts)
        )
        n_nan = pivot.isna().sum().sum()
        if n_nan > 0:
            log(f"    WARNING: {col} has {n_nan} NaN cells in pivot", report_lines)
        feature_arrays.append(pivot.values.astype(np.float32))

    # Stack into shape (n_months, n_districts, n_features)
    feature_matrix = np.stack(feature_arrays, axis=-1)
    log(f"    feature_matrix shape: {feature_matrix.shape}  "
        f"(months={n_months}, districts={n_districts}, features={len(FEATURE_COLS)})",
        report_lines)

    # Case matrix convenience slice
    case_matrix = feature_matrix[:, :, 0]  # (n_months, n_districts)
    log(f"    case_matrix shape: {case_matrix.shape}", report_lines)

    # ------------------------------------------------------------------
    # 12. Chronological split indices
    # ------------------------------------------------------------------
    ym_index = all_dates.to_period("M")

    def find_month_idx(ym_str: str, side: str) -> int:
        target = pd.Period(ym_str, freq="M")
        matches = np.where(ym_index == target)[0]
        if len(matches) == 0:
            sys.exit(f"ERROR: period {ym_str} not found in date index")
        return int(matches[0])

    train_start_idx = 0
    train_end_idx   = find_month_idx(TRAIN_END, "end")
    val_start_idx   = find_month_idx(VAL_START, "start")
    val_end_idx     = find_month_idx(VAL_END, "end")
    test_start_idx  = find_month_idx(TEST_START, "start")
    test_end_idx    = find_month_idx(TEST_END, "end")

    split = {
        "train": [train_start_idx, train_end_idx],
        "val":   [val_start_idx,   val_end_idx],
        "test":  [test_start_idx,  test_end_idx],
    }

    log("\n[11] Chronological split (inclusive month indices):", report_lines)
    for phase, (s, e) in split.items():
        n = e - s + 1
        start_str = str(all_dates[s].date())
        end_str   = str(all_dates[e].date())
        log(f"    {phase:5s}: indices {s:3d}-{e:3d}  |  {start_str} -> {end_str}  ({n} months)",
            report_lines)

    assert train_end_idx < val_start_idx, "train/val overlap"
    assert val_end_idx   < test_start_idx, "val/test overlap"
    log("    No temporal overlap between splits: OK", report_lines)

    # ------------------------------------------------------------------
    # 13. Per-split case statistics
    # ------------------------------------------------------------------
    log("\n[12] Case statistics per split (all districts combined):", report_lines)
    for phase, (s, e) in split.items():
        vals = case_matrix[s:e+1].flatten()
        log(f"    {phase:5s}: n={len(vals):6,}  "
            f"mean={vals.mean():.1f}  std={vals.std():.1f}  "
            f"min={vals.min():.0f}  max={vals.max():.0f}  "
            f"zeros={int((vals==0).sum())}", report_lines)

    # ------------------------------------------------------------------
    # 14. Save artifacts
    # ------------------------------------------------------------------
    log("\n[13] Saving artifacts to " + str(OUTPUT_DIR), report_lines)

    dist_order_path = OUTPUT_DIR / "district_order.json"
    with open(dist_order_path, "w", encoding="utf-8") as fh:
        json.dump({"districts": districts, "n_districts": n_districts}, fh, indent=2)
    log(f"    district_order.json  ({n_districts} districts)", report_lines)

    feat_path = OUTPUT_DIR / "feature_matrix.npy"
    np.save(feat_path, feature_matrix)
    log(f"    feature_matrix.npy   shape={feature_matrix.shape}", report_lines)

    case_path = OUTPUT_DIR / "case_matrix.npy"
    np.save(case_path, case_matrix)
    log(f"    case_matrix.npy      shape={case_matrix.shape}", report_lines)

    date_df = pd.DataFrame({
        "month_idx":  range(n_months),
        "date":       all_dates.strftime("%Y-%m-%d"),
        "year_month": all_dates.to_period("M").astype(str),
    })
    date_path = OUTPUT_DIR / "date_index.csv"
    date_df.to_csv(date_path, index=False)
    log(f"    date_index.csv       {n_months} rows", report_lines)

    split_path = OUTPUT_DIR / "split_indices.json"
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)
    log(f"    split_indices.json", report_lines)

    feat_names_path = OUTPUT_DIR / "feature_names.json"
    with open(feat_names_path, "w", encoding="utf-8") as fh:
        json.dump({"features": FEATURE_COLS}, fh, indent=2)
    log(f"    feature_names.json   {FEATURE_COLS}", report_lines)

    # ------------------------------------------------------------------
    # 15. Write quality report
    # ------------------------------------------------------------------
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    print(f"\nReport written to: {REPORT_PATH}")

    log("\n" + "=" * 60, report_lines)
    log("DATA PREPARATION COMPLETE", report_lines)
    log("=" * 60, report_lines)


if __name__ == "__main__":
    main()
