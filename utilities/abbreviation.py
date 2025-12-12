import pandas as pd

# Input & Output paths
input_file = "/Users/sufarhansudaryo/Documents/Work/semantic-automapper/data/excel_files/Abbreviation.xlsx"
output_file = "/Users/sufarhansudaryo/Documents/Work/semantic-automapper/data/excel_files/Abbreviation_expanded.xlsx"

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
