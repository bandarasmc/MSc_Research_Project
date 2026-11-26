
import pdfplumber
import pandas as pd
import re

# Input and output files
pdf_path = "data\\Extracted_Data\\Population\\Raw\\CPH_2024_Population_Tables.pdf"
output_csv = "data\\Extracted_Data\\Population\\table_A1_population_by_district_sex_sector.csv"

# PDF page 10 (Python uses zero-based indexing)
page_number = 9

# District names as shown in the PDF
districts = [
    "Sri Lanka",
    "Colombo", "Gampaha", "Kalutara", "Kandy", "Matale",
    "Nuwara Eliya", "Galle", "Matara", "Hambanthota",
    "Jaffna", "Mannar", "Vavuniya", "Mullaitivu",
    "Kilinochchi", "Batticaloa", "Ampara", "Trincomalee",
    "Kurunegala", "Puttalam", "Anuradhapura",
    "Polonnaruwa", "Badulla", "Moneragala",
    "Ratnapura", "Kegalle"
]

# Column names in the table
columns = [
    "All_Sectors_Both_Sexes",
    "All_Sectors_Male",
    "All_Sectors_Female",
    "Urban_Both_Sexes",
    "Urban_Male",
    "Urban_Female",
    "Rural_Both_Sexes",
    "Rural_Male",
    "Rural_Female",
    "Estate_Rural_Both_Sexes",
    "Estate_Rural_Male",
    "Estate_Rural_Female"
]

# Match numeric values (including commas) and dashes
value_pattern = re.compile(r"(?<!\S)(?:[\d,]+|-)(?!\S)")

# Extract text from PDF page 10
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[page_number]
    text = page.extract_text()

if not text:
    raise ValueError("No text found on PDF page 10.")

records = []
errors = []

# Match district names at the beginning of each row
district_pattern = re.compile(
    r"^(" + "|".join(
        re.escape(d) for d in sorted(districts, key=len, reverse=True)
    ) + r")\s+(.*)$"
)

for line in text.splitlines():
    line = line.strip()
    match = district_pattern.match(line)

    if not match:
        continue

    district = match.group(1)
    row_text = match.group(2)

    values = value_pattern.findall(row_text)

    if len(values) != 12:
        errors.append({
            "district": district,
            "values_found": len(values),
            "line": line
        })
        continue

    record = {"District": district}

    for column, value in zip(columns, values):
        record[column] = (
            int(value.replace(",", ""))
            if value != "-"
            else pd.NA
        )

    records.append(record)

# Validate extraction
df = pd.DataFrame(records, columns=["District"] + columns)

if errors:
    print("Some rows need inspection:")
    for error in errors:
        print(
            f"{error['district']}: "
            f"{error['values_found']} values found"
        )

if len(df) != 26:
    print(f"Warning: Expected 26 rows, but extracted {len(df)}.")

if df.empty:
    print("\nExtracted page text:")
    print(text)
    raise ValueError("No complete table rows were extracted.")

# Save the CSV with Excel-compatible UTF-8 encoding
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"\nSuccessfully extracted {len(df)} rows.")
print(f"Saved CSV to: {output_csv}")
print(df.to_string(index=False))
