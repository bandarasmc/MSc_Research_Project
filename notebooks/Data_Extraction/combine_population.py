import sys
import pandas as pd

POP_CSV = sys.argv[1] if len(sys.argv) > 1 else "data\\Extracted_Data\\Population\\district_population_2011.csv"
A1_CSV = sys.argv[2] if len(sys.argv) > 2 else "data\\Extracted_Data\\Population\\table_A1_population_by_district_sex_sector.csv"
OUT_CSV = sys.argv[3] if len(sys.argv) > 3 else "data\\Extracted_Data\\Population\\district_population_merged.csv"

SI_TO_EN = {
    "ෙකොළඹ": "Colombo",
    "ගම්පහ": "Gampaha",
    "කළුතර": "Kalutara",
    "මහනුවර": "Kandy",
    "මාතෙල්": "Matale",
    "නුවරඑළිය": "Nuwara Eliya",
    "ගාල්ල": "Galle",
    "මාතර": "Matara",
    "හම්බන්ෙතොට": "Hambanthota",
    "යාපනය": "Jaffna",
    "මන්නාරම": "Mannar",
    "වවුනියාව": "Vavuniya",
    "මුලතිව්": "Mullaitivu",
    "කිළිෙනොච්චි": "Kilinochchi",
    "මඩකලපුව": "Batticaloa",
    "අම්පාර": "Ampara",
    "තිකුණාමලය": "Trincomalee",
    "කුරුණෑගල": "Kurunegala",
    "පුත්තලම": "Puttalam",
    "අනුරාධපුර": "Anuradhapura",
    "ෙපොෙලොන්නරුව": "Polonnaruwa",
    "බදුල්ල": "Badulla",
    "ෙමොනරාගල": "Moneragala",
    "රත්නපුර": "Ratnapura",
    "කෑගල්ල": "Kegalle",
}


pop = pd.read_csv(POP_CSV, encoding="utf-8-sig")
pop.columns = ["District_Sinhala", "Population_1981", "Population_2001", "Population_2012"]
pop["District_Sinhala"] = pop["District_Sinhala"].str.strip()
pop["District"] = pop["District_Sinhala"].map(SI_TO_EN)

unmapped = pop.loc[pop["District"].isna(), "District_Sinhala"].tolist()
if unmapped:
    sys.exit(f"No English mapping for: {unmapped}")

pop = pop[["District", "District_Sinhala", "Population_2001", "Population_2012"]]

# --- File 2: Table A1 --------------------------------------------------------
a1 = pd.read_csv(A1_CSV, encoding="utf-8-sig", usecols=["District", "All_Sectors_Both_Sexes"])
a1 = a1.rename(columns={"All_Sectors_Both_Sexes": "Population_2024"})
a1["District"] = a1["District"].str.strip()
a1 = a1[a1["District"] != "Sri Lanka"]  # national total, not a district

# --- Merge (validate that every district matches exactly once) ---------------
merged = pop.merge(a1, on="District", how="inner", validate="one_to_one")
if len(merged) != len(pop) or len(merged) != len(a1):
    sys.exit(f"Row mismatch: pop={len(pop)}, a1={len(a1)}, merged={len(merged)}")

# Nullable ints so blanks stay empty instead of turning the columns into floats.
for col in ["Population_2001", "Population_2012", "Population_2024"]:
    merged[col] = merged[col].astype("Int64")

merged = merged.sort_values("District").reset_index(drop=True)
merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"Wrote {len(merged)} rows to {OUT_CSV}")
print(merged.isna().sum().to_string())
