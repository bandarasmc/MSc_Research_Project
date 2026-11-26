import pdfplumber
import pandas as pd
import os
import re

## CODE FOR 4-4-5 MAPPING -To convert weeks to Months
def week_to_month_445(week):
    month_ranges = {
        1: range(1, 6),    2: range(6, 10),    3: range(10, 15),
        4: range(15, 19),  5: range(19, 23),  6: range(23, 28),
        7: range(28, 32),  8: range(32, 36),  9: range(36, 41),
        10: range(41, 45), 11: range(45, 49), 12: range(49, 54)
    }

    for month, weeks in month_ranges.items():
        if week in weeks:
            return f"Month_{month:02d}"
        


PDF_FOLDER = "data\\Extracted_Data\\Leptospirosis\\WER_2010"
OUTPUT_FILE = "data\\Extracted_Data\\Leptospirosis\\leptospirosis_monthly_2010.csv"

data = {}

for file in sorted(os.listdir(PDF_FOLDER)):
    if not file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, file)

    match = re.search(r"No_(\d+)", file)
    if not match:
        continue

    week_num = int(match.group(1))

    # Convert week to month-445 method
    month = week_to_month_445(week_num)


    print(f"Processing {file} -> {month}")

    with pdfplumber.open(pdf_path) as pdf:
        last_page = pdf.pages[-1]
        text = last_page.extract_text()

    lines = text.split("\n")

    for line in lines:
        parts = line.split()

        if len(parts) < 13:
            continue

        district = parts[0]

        if district.upper() in ["SRI", "SRI LANKA"]:
            continue

        try:
            if district.replace(".", "").upper() == "NUWARA" and parts[1].replace(".", "").upper() == "ELIYA":
                lepto_A = int(parts[12])   # Special case
            elif district.upper() == "NUWARAELIYA":
                lepto_A = int(parts[12])   # Merged word
            else:
                lepto_A = int(parts[11])   # Normal case
        except:
            continue

        if district not in data:
            data[district] = {}

        # Monthly SUMMATION happens here
        if month not in data[district]:
            data[district][month] = 0

        data[district][month] += lepto_A


# Final Monthly Output
final_df = pd.DataFrame.from_dict(data, orient="index").sort_index()
final_df.index.name = "District"
final_df = final_df.sort_index(axis=1)

final_df.to_csv(OUTPUT_FILE)

print("\n SUCCESS:")
print(" - leptospirosis_monthly.csv")

