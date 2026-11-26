"""
notebooks/Data_Modeling/build_graph.py
=======================================
Task 3 — Build and validate the district spatial adjacency graph.

Prerequisite: run prepare_data.py first (produces district_order.json).

Outputs (all written to outputs/model/):
  adjacency_binary.npy      — shape (25, 25) int8, binary, no self-loops
  adjacency_norm.npy        — shape (25, 25) float32, normalised with self-loops
  adjacency_report.txt      — human-readable validation log

Run from the project root:
    .venv/Scripts/python.exe notebooks/Data_Modeling/build_graph.py

Adjacency rule
--------------
A[i,j] = 1 when the Shapely intersection of districts i and j has
         positive length, i.e., they share a boundary edge of non-zero
         length (not merely a point vertex contact).

Detection method
----------------
1. Geometries are loaded from the GeoJSON via geopandas (CRS EPSG:4326).
2. The hard-coded LK-id -> English district name mapping resolves the
   Sinhala transliteration mismatch in lk.json.
3. Each pair (i, j) with i < j is tested:
   a. Compute intersection of the two raw (unbuffered) geometries.
   b. If intersection.length > 0  -> shared boundary edge -> A[i,j] = 1.
   c. If intersection.length == 0 -> point contact or no contact -> 0.
   This criterion works correctly for all Shapely geometry types:
   MultiLineString (most common), GeometryCollection containing lines,
   and degenerate cases that produce a zero-area Polygon.
4. No buffer is applied.  Inspection confirmed that all 58 geographically
   correct adjacency pairs already produce positive-length intersections
   in the raw GeoJSON (the polygons share exact boundary vertices thanks
   to the simplemaps simplified topology).

Normalised adjacency (STGCN)
-----------------------------
Following Kipf & Welling (2017), the symmetrically normalised adjacency
with added self-loops is:

    A_hat = D^{-1/2} (A + I) D^{-1/2}

where A is the binary adjacency matrix and I is the identity.
This is the standard form used in graph convolutional networks.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows so Sinhala characters in GeoJSON names
# don't cause UnicodeEncodeError on cp1252 terminals.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import geopandas as gpd
import numpy as np
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at notebooks/Data_Modeling/, two levels below the root.
ROOT       = Path(__file__).resolve().parents[2]
GEOJSON    = ROOT / "data" / "spatial" / "lk.json"
DIST_ORDER = ROOT / "outputs" / "model" / "district_order.json"
OUTPUT_DIR = ROOT / "outputs" / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hard-coded LK administrative-id -> CSV English district name
# Source: inspection of lk.json properties and CSV district names.
# ---------------------------------------------------------------------------
LK_ID_TO_DISTRICT: dict[str, str] = {
    "LK11": "Colombo",
    "LK12": "Gampaha",
    "LK13": "Kalutara",
    "LK21": "Kandy",
    "LK22": "Matale",
    "LK23": "Nuwara Eliya",
    "LK31": "Galle",
    "LK32": "Matara",
    "LK33": "Hambantota",
    "LK41": "Jaffna",
    "LK42": "Kilinochchi",
    "LK43": "Mannar",
    "LK44": "Vavuniya",
    "LK45": "Mullaitivu",
    "LK51": "Batticaloa",
    "LK52": "Ampara",
    "LK53": "Trincomalee",
    "LK61": "Kurunegala",
    "LK62": "Puttalam",
    "LK71": "Anuradhapura",
    "LK72": "Polonnaruwa",
    "LK81": "Badulla",
    "LK82": "Monaragala",
    "LK91": "Ratnapura",
    "LK92": "Kegalle",
}

# Adjacency criterion: intersection.length > LENGTH_THRESHOLD
# Value 0 works because raw GeoJSON already shares exact boundary vertices;
# genuine shared edges produce MultiLineString with length > 0,
# point-only contacts produce a zero-length result.
LENGTH_THRESHOLD = 0.0


# ===========================================================================
# Helpers
# ===========================================================================

def log(msg: str, sink: list[str]) -> None:
    print(msg)
    sink.append(msg)


def has_linear_component(geom) -> bool:
    """Return True if geom has positive 1-D (linear) length."""
    return (not geom.is_empty) and (geom.length > LENGTH_THRESHOLD)


def normalise_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalised adjacency with self-loops:
        A_hat = D^{-1/2} (A + I) D^{-1/2}
    Returns float32 array of same shape.
    """
    n = A.shape[0]
    A_hat = A.astype(np.float64) + np.eye(n)
    degree = A_hat.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm.astype(np.float32)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    report_lines: list[str] = []

    log("=" * 60, report_lines)
    log("SPATIAL ADJACENCY GRAPH CONSTRUCTION", report_lines)
    log("=" * 60, report_lines)

    # ------------------------------------------------------------------
    # 1. Load canonical district ordering
    # ------------------------------------------------------------------
    if not DIST_ORDER.exists():
        sys.exit(
            f"ERROR: district_order.json not found at {DIST_ORDER}.\n"
            "Run prepare_data.py first."
        )
    with open(DIST_ORDER, encoding="utf-8") as fh:
        districts: list[str] = json.load(fh)["districts"]
    n = len(districts)
    log(f"\n[1] Loaded district order: {n} districts", report_lines)

    # ------------------------------------------------------------------
    # 2. Load GeoJSON
    # ------------------------------------------------------------------
    if not GEOJSON.exists():
        sys.exit(f"ERROR: GeoJSON not found at {GEOJSON}")
    gdf = gpd.read_file(GEOJSON)
    log(f"[2] GeoJSON features: {len(gdf)}", report_lines)
    log(f"    CRS: {gdf.crs}", report_lines)

    # ------------------------------------------------------------------
    # 3. Apply LK-id -> district name mapping
    # ------------------------------------------------------------------
    gdf["district_name"] = gdf["id"].map(LK_ID_TO_DISTRICT)

    unmapped = gdf[gdf["district_name"].isna()]["id"].tolist()
    if unmapped:
        log(f"WARNING: unmapped GeoJSON ids: {unmapped}", report_lines)
        sys.exit("ERROR: not all GeoJSON features could be mapped to district names.")

    log("\n[3] GeoJSON id -> district name mapping:", report_lines)
    for _, row in gdf.sort_values("district_name").iterrows():
        log(f"    {row['id']:6s}  {row['name']:30s} -> {row['district_name']}", report_lines)

    # ------------------------------------------------------------------
    # 4. Verify all 25 districts covered and match the canonical order
    # ------------------------------------------------------------------
    geojson_districts    = set(gdf["district_name"].tolist())
    canonical_districts  = set(districts)
    in_geojson_not_data  = geojson_districts - canonical_districts
    in_data_not_geojson  = canonical_districts - geojson_districts

    if in_geojson_not_data:
        log(f"WARNING: in GeoJSON, not in dataset: {sorted(in_geojson_not_data)}", report_lines)
    if in_data_not_geojson:
        log(f"WARNING: in dataset, not in GeoJSON: {sorted(in_data_not_geojson)}", report_lines)
    if in_geojson_not_data or in_data_not_geojson:
        sys.exit("ERROR: district name mismatch between GeoJSON and dataset.")

    log("\n[4] All 25 districts match between GeoJSON and dataset: OK", report_lines)

    # ------------------------------------------------------------------
    # 5. Index geometries by canonical district order
    # ------------------------------------------------------------------
    gdf = gdf.set_index("district_name")
    geom_by_district: dict[str, object] = {}
    for dist in districts:
        geom_by_district[dist] = unary_union(gdf.loc[dist, "geometry"])

    # Validate geometries
    log("\n[5] Geometry validity check:", report_lines)
    invalid = []
    for dist, geom in geom_by_district.items():
        if not geom.is_valid:
            invalid.append(dist)
            log(f"    INVALID geometry: {dist}", report_lines)
    if not invalid:
        log("    All 25 geometries are valid: OK", report_lines)
    else:
        log(f"    {len(invalid)} invalid geometries — applying buffer fix", report_lines)
        for dist in invalid:
            geom_by_district[dist] = geom_by_district[dist].buffer(0)

    # ------------------------------------------------------------------
    # 6. Build binary adjacency matrix
    # ------------------------------------------------------------------
    log(f"\n[6] Building {n}x{n} binary adjacency matrix "
        f"(criterion: intersection.length > {LENGTH_THRESHOLD})...", report_lines)

    A = np.zeros((n, n), dtype=np.int8)
    adjacency_pairs: list[tuple[str, str]] = []
    point_contacts:  list[tuple[str, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            di = districts[i]
            dj = districts[j]
            inter = geom_by_district[di].intersection(geom_by_district[dj])

            if has_linear_component(inter):
                A[i, j] = 1
                A[j, i] = 1
                adjacency_pairs.append((di, dj))
            elif not inter.is_empty:
                point_contacts.append((di, dj))

    log(f"    Adjacent pairs (shared edge, length > 0): {len(adjacency_pairs)}", report_lines)
    if point_contacts:
        log(f"    Point-only contacts (not adjacent): {len(point_contacts)}", report_lines)
        for a, b in point_contacts:
            log(f"      {a} <-> {b}", report_lines)

    # ------------------------------------------------------------------
    # 7. Validation
    # ------------------------------------------------------------------
    log("\n[7] Adjacency matrix validation:", report_lines)

    assert A.shape == (n, n), f"Wrong shape: {A.shape}"
    log(f"    Shape: {A.shape}  (expected ({n}, {n})): OK", report_lines)

    symmetric = np.array_equal(A, A.T)
    log(f"    Symmetric: {symmetric}", report_lines)
    assert symmetric, "Adjacency matrix is not symmetric"

    diag_sum = int(np.diag(A).sum())
    log(f"    Diagonal sum (self-loops): {diag_sum}  (expected 0): "
        + ("OK" if diag_sum == 0 else "FAIL"), report_lines)
    assert diag_sum == 0, "Binary adjacency matrix must have zero diagonal"

    # BFS connectivity
    visited: set[int] = set()
    queue = [0]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        neighbours = np.where(A[node] == 1)[0].tolist()
        queue.extend(nb for nb in neighbours if nb not in visited)
    isolated = [districts[i] for i in range(n) if i not in visited]
    log(f"    Connected nodes: {len(visited)}/{n}", report_lines)
    if isolated:
        log(f"    WARNING: isolated nodes: {isolated}", report_lines)
    else:
        log("    Graph is fully connected: OK", report_lines)

    degrees = A.sum(axis=1).astype(int)
    log(f"    Degree range: {degrees.min()} - {degrees.max()}  "
        f"mean={degrees.mean():.2f}", report_lines)
    log("\n    Per-district degrees:", report_lines)
    for i, dist in enumerate(districts):
        log(f"      {i:2d}  {dist:20s}  degree={degrees[i]}", report_lines)

    log(f"\n    All {len(adjacency_pairs)} adjacent pairs:", report_lines)
    for a, b in sorted(adjacency_pairs):
        log(f"      {a} <-> {b}", report_lines)

    # ------------------------------------------------------------------
    # 8. Normalised adjacency
    # ------------------------------------------------------------------
    A_norm = normalise_adjacency(A)
    log(f"\n[8] Normalised adjacency (Kipf & Welling D^-0.5 (A+I) D^-0.5):", report_lines)
    log(f"    Shape: {A_norm.shape}", report_lines)
    log(f"    Value range: [{A_norm.min():.4f}, {A_norm.max():.4f}]", report_lines)
    log(f"    Symmetric: {np.allclose(A_norm, A_norm.T)}", report_lines)

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    bin_path  = OUTPUT_DIR / "adjacency_binary.npy"
    norm_path = OUTPUT_DIR / "adjacency_norm.npy"
    np.save(bin_path,  A)
    np.save(norm_path, A_norm)

    log(f"\n[9] Saved artifacts:", report_lines)
    log(f"    adjacency_binary.npy   shape={A.shape}  dtype={A.dtype}", report_lines)
    log(f"    adjacency_norm.npy     shape={A_norm.shape}  dtype={A_norm.dtype}", report_lines)

    report_path = OUTPUT_DIR / "adjacency_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines))
    print(f"\nReport written to: {report_path}")

    log("\n" + "=" * 60, report_lines)
    log("GRAPH CONSTRUCTION COMPLETE", report_lines)
    log("=" * 60, report_lines)


if __name__ == "__main__":
    main()
