import pandas as pd

# ---------- CONFIG ----------
SOURCE_FILE = "data/excel_files/item_class_assignment_validated vs Australian mapping.xlsx"
TARGET_FILE = "automapper_llm.xlsx"
OUTPUT_FILE = "merged.xlsx"

ID_COL = "ID"                 # must exist in BOTH files
SOURCE_COL = "Equivalent to Resourcly conclusion"    # column to copy from source
NEW_COL = "Ground Truth"       # name in target (can be different)
# ----------------------------

df_source = pd.read_excel(SOURCE_FILE)
df_target = pd.read_excel(TARGET_FILE)

# create mapping: ID -> value
mapping = dict(zip(df_source[ID_COL], df_source[SOURCE_COL]))

# add new column to target
df_target[NEW_COL] = df_target[ID_COL].map(mapping)

# save
df_target.to_excel(OUTPUT_FILE, index=False)

print("Done! Saved to", OUTPUT_FILE)
