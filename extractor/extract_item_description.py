from __future__ import annotations

import os
import time
import json
import re
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from google import genai
from dotenv import load_dotenv


# Config
DESC_COL_PRIMARY = "Description_Expanded"
DESC_COL_FALLBACK = "Description"
OUT_COL = "Item_Name"

MAX_WORKERS = 10
RETRIES = 3
RETRY_DELAY = 2

# If your ID is always the first column, we’ll detect it per sheet by column order.
# If you have a fixed column name, set it here:
ID_COL_NAME: Optional[str] = None  # e.g. "Item No." or "Part Number"


# Few-shot examples (EDIT / EXTEND THESE)
# Output MUST be only the main item name (single line), short, no numbers.
FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "input": {
            "description": "HOSE; PVC; SUCTION; 50MM;20MTR/ROLL; 100PSI; GREY",
        },
        "output": "HOSE",
    },
    {
        "input": {
            "description": "VALVE; BALL; 20MM; 1 PIECE;SG IRON BODY; LEVER",
        },
        "output": "VALVE",
    },
    {
        "input": {
            "description": "HOLLOW BAR R32S/L=3000 mm",
        },
        "output": "Hollow Bar",
    },
    {
        "input": {
            "description": "BULLFLEX STRUCTURAL SEALING",
        },
        "output": "BULLFLEX STRUCTURAL SEALING",
    },
    {
        "input": {
            "description": "CT BOLT STEEL 1800 150/65 L/H",
        },
        "output": "CT BOLT STEEL",
    },
    {
        "input": {
            "description": "40MM-950/1050MPa THREADBAR ANCHOR",
        },
        "output": "THREADBAR ANCHOR",
    },
    {
        "input": {
            "description": "57MM HOLE DCP CABLE BOLT",
        },
        "output": "CABLE BOLT",
    },
    {
        "input": {
            "description": "CONNECTION PLATE 260x145x12",
        },
        "output": "CONNECTION PLATE",
    },
    {
        "input": {
            "description": "SPANNER T38 HEX41 FEM 720LG",
        },
        "output": "SPANNER",
    },
     {
        "input": {
            "description": "Cable Bolt Twin Strand Bulbed 6.5M",
        },
        "output": "Cable Bolt Twin Strand Bulbed",
    },
     {
        "input": {
            "description": "Dragonfly Plate 300X280 Complete With D15151501HL Galvanised",
        },
        "output": "Dragonfly Plate",
    },
     {
        "input": {
            "description": "Mesh Module Strap Galvanised 3.0X0.4M 8MM WIRE",
        },
        "output": "Mesh Module Strap Galvanised",
    },
    {
        "input": {
            "description": "Spanner Square Drive 22 300LG 24MM 36AF Drive",
        },
        "output": "Spanner Square Drive",
    },
]


def _few_shot_block() -> str:
    blocks: List[str] = []
    for ex in FEW_SHOT_EXAMPLES:
        blocks.append(
            "EXAMPLE INPUT:\n"
            f"{json.dumps(ex['input'], ensure_ascii=False, indent=2)}\n"
            "EXAMPLE OUTPUT:\n"
            f"{ex['output']}\n"
        )
    return "\n".join(blocks)


# Env + client (Gemini)
def init_gemini_client() -> genai.Client:
    # Path to this script: <project_root>/utilities/xyz.py
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    dotenv_file_path = project_root / ".env.local"
    if not dotenv_file_path.exists():
        raise FileNotFoundError(f".env.local not found at: {dotenv_file_path}")

    load_dotenv(dotenv_path=dotenv_file_path)
    return genai.Client()


client = init_gemini_client()


# Prompt
def build_main_name_prompt(description: str) -> str:
    few_shots = _few_shot_block()

    return f"""
You extract the MAIN ITEM NAME (product type) from an industrial item description.

Hard rules:
- Return ONLY the main item name/description as a SINGLE LINE of text.
- Do NOT include any numbers, dimensions, units, quantities, standards, part numbers, locations, freight, packaging, or "C/W..." details.
- Use Title Case where reasonable (e.g., "Cable Bolt", "Butterfly Plate").
- Do not invent information. If uncertain, output the best short product type implied by the description.
- No JSON, no quotes, no markdown, no explanations.

{few_shots}

NOW PROCESS THIS ITEM:

INPUT:
{json.dumps({
    "description": description
}, ensure_ascii=False, indent=2)}

OUTPUT (single line only):
""".strip()


# Model call with retries
def call_llm_with_retries(prompt: str, retries: int = RETRIES, delay: int = RETRY_DELAY) -> str:
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return f"ERROR: {str(e)}"


# Output cleaning / validation
def clean_single_line(text: str) -> str:
    t = (text or "").strip()
    t = t.strip("`").strip()
    t = t.strip('"').strip("'").strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def is_valid_main_name(name: str) -> bool:
    if not name:
        return False
    # main name should not contain digits
    if re.search(r"\d", name):
        return False
    # avoid super long model rambles
    if len(name) > 60:
        return False
    # keep it reasonably short
    if len(name.split()) > 8:
        return False
    return True


def fallback_main_name(description: str) -> str:
    """
    Safe fallback: return the first 2-3 alphabetic-ish tokens
    (only used if model fails). You can replace this later.
    """
    tokens = re.split(r"[\s,;/]+", (description or "").strip())
    keep: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if re.search(r"\d", tok):
            continue
        if len(tok) <= 1:
            continue
        keep.append(tok)
        if len(keep) >= 3:
            break
    return " ".join(keep).strip() or (description or "").strip()


# Row extraction
def get_id_col(df: pd.DataFrame) -> str:
    if ID_COL_NAME and ID_COL_NAME in df.columns:
        return ID_COL_NAME
    return df.columns[0]


def pick_description(row: pd.Series, primary: str, fallback: str) -> str:
    """
    Prefer primary column (Description_Expanded). If empty, use fallback (Description).
    Returns "" if both are missing/empty.
    """
    def _val(col: str) -> str:
        if col not in row or pd.isna(row[col]):
            return ""
        s = str(row[col])
        return s.strip()

    p = _val(primary)
    if p:
        return p
    return _val(fallback)


# Main processing (sheet by sheet)
def extract_main_names_workbook(
    input_xlsx: str,
    output_xlsx: str,
    primary_desc_col: str = DESC_COL_PRIMARY,
    fallback_desc_col: str = DESC_COL_FALLBACK,
    out_col: str = OUT_COL,
    max_workers: int = MAX_WORKERS,
):

    xls = pd.ExcelFile(input_xlsx)
    out_sheets: Dict[str, pd.DataFrame] = {}

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(input_xlsx, sheet_name=sheet_name)

        if df.empty:
            out_sheets[sheet_name] = df
            continue

       # If neither description column exists, skip this sheet
        if (primary_desc_col not in df.columns) and (fallback_desc_col not in df.columns):
            out_sheets[sheet_name] = df
            continue

        id_col = get_id_col(df)

        # Create output column (do not disturb other data)
        df[out_col] = ""

        # Work on rows where either primary or fallback description is non-empty
        def _has_text(series: pd.Series) -> pd.Series:
            return series.notna() & (series.astype(str).str.strip() != "")

        mask_primary = _has_text(df[primary_desc_col]) if primary_desc_col in df.columns else pd.Series([False] * len(df))
        mask_fallback = _has_text(df[fallback_desc_col]) if fallback_desc_col in df.columns else pd.Series([False] * len(df))

        mask = mask_primary | mask_fallback
        work_df = df[mask].copy()


        if work_df.empty:
            out_sheets[sheet_name] = df
            continue

        futures = {}
        results_by_row_idx: Dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, row in work_df.iterrows():
                item_id = str(row[id_col]).strip()
                description = pick_description(row, primary_desc_col, fallback_desc_col)
                prompt = build_main_name_prompt(description)

                fut = executor.submit(call_llm_with_retries, prompt)
                futures[fut] = (idx, item_id, description)

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting names: {sheet_name}"):
                idx, item_id, original_desc = futures[fut]
                raw = fut.result()

                name = clean_single_line(raw)

                if (not name) or name.startswith("ERROR:") or (not is_valid_main_name(name)):
                    name = fallback_main_name(original_desc)

                results_by_row_idx[idx] = name

        # Write results back by row index (no dependence on model output IDs)
        for idx, name in results_by_row_idx.items():
            df.at[idx, out_col] = name

        out_sheets[sheet_name] = df

    # Save to a new workbook
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name, df in out_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Saved: {output_xlsx}")


if __name__ == "__main__":
    extract_main_names_workbook(
        input_xlsx="/Users/sufarhansudaryo/Documents/Work/semantic-automapper/data/excel_files/AU01_Australian_merged.xlsx",
        output_xlsx="Australian_Items_Naming.xlsx",
        max_workers=MAX_WORKERS,
    )
