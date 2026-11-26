
import pdfplumber
import pandas as pd
import re


pdf_path = "data\\Extracted_Data\\Population\\Raw\\CensusPopulationHousing-PreliminaryReport_2011.pdf"
output_path = "data\\Extracted_Data\\Population\\district_population_2011.csv"

# PDF page 21 = Python index 20
with pdfplumber.open(pdf_path) as pdf:
    text = pdf.pages[20].extract_text()

if not text:
    raise ValueError("No text extracted from PDF page 21.")

records = []

# Match each row's district name and first three population values.
row_pattern = re.compile(
    r"^(.+?)\s+"
    r"([\d,]+|-)\s+"
    r"([\d,]+|-)\s+"
    r"([\d,]+|-)\s+"
)

for line in text.splitlines():
    line = line.strip()
    match = row_pattern.match(line)

    if not match:
        continue

    district = match.group(1).strip()
    pop_1981, pop_2001, pop_2012 = match.groups()[1:]

    # Keep district rows only; exclude the national total.
    if district == "ශී ලංකාව":
        continue

    # Keep the original Sinhala district name and population figures.
    records.append({
        "දිස්ත්‍රික්කය": district,
        "ජනගහනය_1981": pop_1981,
        "ජනගහනය_2001": pop_2001,
        "ජනගහනය_2012": pop_2012
    })

df = pd.DataFrame(records)

if df.empty:
    print("No rows were matched. Extracted page text:")
    print(text)
else:
    # Remove commas and convert valid population values to integers.
    for column in [
        "ජනගහනය_1981",
        "ජනගහනය_2001",
        "ජනගහනය_2012"
    ]:
        df[column] = df[column].apply(
            lambda x: int(x.replace(",", "")) if x != "-" else pd.NA
        ).astype("Int64")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Extracted {len(df)} rows.")
    print(df.to_string(index=False))
    print(f"\nSaved to: {output_path}")
