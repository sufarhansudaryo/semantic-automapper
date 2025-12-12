import pandas as pd

# --- Load both Excel files ---
au01_df = pd.read_excel("data/excel_files/AU01_Items_Extracted.xlsx")
map_df = pd.read_excel("/Users/sufarhansudaryo/Documents/Work/semantic-automapper/data/excel_files/Australian items to be mapped.xlsx")

# --- Ensure the ID column names are correct ---
# Replace the column names below with the actual column names in your sheets
au01_id_col = "No."               # Column in AU01 items
map_id_col = "no."                # Column in mapping file

# --- Filter AU01 items where id is in the mapping list ---
filtered_df = au01_df[au01_df[au01_id_col].isin(map_df[map_id_col])]

# --- Save the result ---
filtered_df.to_excel("AU01_Australian.xlsx", index=False)

print("Filtering complete! Saved as AU01_filtered_output.xlsx")
