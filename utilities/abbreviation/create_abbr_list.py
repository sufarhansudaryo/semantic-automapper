import pandas as pd

"""
Expand Abbreviation Lists from Excel into a Flat Table.

This script reads an Excel file where:
- Column A (index 0) contains one cell per row with multiple abbreviation mappings,
  stored as a comma-separated list in the form:
      "ABBR1=Description 1, ABBR2=Description 2, ..."

- Column B (index 1) contains the corresponding sheet/category name for that row.

The script expands each abbreviation mapping into its own row and outputs a clean table
with the columns:
    Code | Description | Sheet

Malformed entries (without "=") are still included, but with an empty Description.

Input:
    /Users/sufarhansudaryo/Documents/Work/semantic-automapper/data/excel_files/Abbreviation.xlsx

Output:
    /Users/sufarhansudaryo/Documents/Work/semantic-automapper/data/excel_files/Abbreviation_expanded.xlsx
"""


# Input & Output paths
input_file = "path/to/your/input/Abbreviation.xlsx"
output_file = "path/to/your/output/Abbreviation_expanded.xlsx"

# Read the Excel file
df = pd.read_excel(input_file, header=None)

# Prepare a list to store expanded rows
rows = []

for _, row in df.iterrows():
    abbr_list = row[0]        # first column = abbreviation list
    sheet_name = row[1]       # second column = sheet/category name

    if pd.isna(abbr_list):
        continue

    # Split by comma to get each pair
    entries = str(abbr_list).split(",")

    for entry in entries:
        entry = entry.strip()
        if "=" in entry:
            code, desc = entry.split("=", 1)
            rows.append({
                "Code": code.strip(),
                "Description": desc.strip(),
                "Sheet": sheet_name
            })
        else:
            # Handle malformed entries gracefully
            rows.append({
                "Code": entry.strip(),
                "Description": "",
                "Sheet": sheet_name
            })

# Create final DataFrame
expanded_df = pd.DataFrame(rows)

# Save to new Excel file
expanded_df.to_excel(output_file, index=False)

print("Done! Output saved to:", output_file)
