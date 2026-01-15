from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from google import genai
from dotenv import load_dotenv

"""
LLM-based Abbreviation Expansion for Multi-Sheet Excel Workbooks (Gemini).

Purpose
-------
This script expands abbreviations in industrial item descriptions stored in an Excel workbook.
Instead of applying a fixed rule-based replacement table, it uses a Gemini LLM to expand only
those abbreviations that are strongly supported by the row context (other columns in the same row)
and the worksheet name. This reduces incorrect expansions when abbreviations are ambiguous.

What this script needs
----------------------
1) Input Excel workbook (multi-sheet) containing at least:
   - A description column (default: "Description") to be expanded.
   - An item identifier column (default: first column in each sheet, or ID_COL_NAME if set).
   - Optional context columns (any additional columns) that help disambiguate abbreviations.

2) Gemini API access via google-genai:
   - A `.env.local` file must exist in the project root (one directory above this script).
   - The `.env.local` file must contain the environment variables required by `google.genai`
     (e.g., API key configuration depending on your setup).

How it works
------------
- Reads all worksheets in the input workbook.
- Only processes worksheets that contain the configured description column (DESC_COL).
- For each row with a non-empty description:
  - Builds a `context` dict from all other non-empty columns (excluding ID and description).
  - Builds a prompt containing:
      * strict expansion rules (no invention, preserve numbers/units/IDs, single-line output)
      * few-shot examples (FEW_SHOT_EXAMPLES) to enforce style and known expansions
      * the current sheet name, item ID, description, and context
  - Sends the prompt to Gemini.
- Writes the model result to a new output column (OUT_COL). If the model output fails validation,
  it falls back to the original description.

Inputs / Outputs
----------------
Input:
  - `input_xlsx`: path to the source workbook (e.g., items_logic_complete.xlsx)

Output:
  - `output_xlsx`: path to the new workbook containing all original sheets and a new expanded
    description column (OUT_COL), written using the openpyxl engine.

Configuration
-------------
- DESC_COL: source description column name (default: "Description")
- OUT_COL: output column name for expanded text (default: "Description_Expanded")
- ID_COL_NAME: optional fixed ID column name; if None, the first column is treated as ID
- MAX_WORKERS, RETRIES, RETRY_DELAY: runtime and robustness parameters
- FEW_SHOT_EXAMPLES: editable examples controlling style and common expansions
"""


# ----------------------------
# Config
# ----------------------------
DESC_COL = "Description"
OUT_COL = "Description_Expanded"
MAX_WORKERS = 10
RETRIES = 3
RETRY_DELAY = 2

# If your ID is always the first column, we’ll detect it per sheet by column order.
# If you have a fixed column name, set it here:
ID_COL_NAME: Optional[str] = None  # e.g. "Item No." or "Part Number"


# ----------------------------
# Few-shot examples (EDIT / EXTEND THESE)
# ----------------------------
# IMPORTANT:
# - Output MUST be one single line (expanded description only).
# - Keep numbers/units unchanged.
# - These examples are meant to teach style + abbreviation patterns.
FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    # -------------------------
    # Bolt Matrix
    # -------------------------
    {
        "input": {
            "sheet_name": "Bolt Matrix",
            "item_id": "AS4150DFP2M050",
            "description": "BOLT Standard Strength Steel 24X1500 C/W AF,BALL",
            "context": {
                "Bolt Type Code": "AS",
                "Bolt Type Description": "Standard Strength Steel",
            },
        },
        "output": "Bolt Standard Strength Steel 24X1500 Complete With AF, Ball",
    },
    {
        "input": {
            "sheet_name": "Bolt Matrix",
            "item_id": "AH4060DFP2MG050",
            "description": "BOLT GRD HIGH 24mmX0.6M GALV",
            "context": {
                "Bolt Type Code": "AH",
                "Bolt Type Description": "High Strength Steel",
                "Coating": "Galvanised",
            },
        },
        "output": "Bolt Grade High 24mmX0.6M Galvanised",
    },

    # -------------------------
    # Ultra Strand
    # -------------------------
    {
        "input": {
            "sheet_name": "Strand Matrix",
            "item_id": "ULN410W50C",
            "description": "ULTRA STRAND IND 21.8mmX4.1M",
            "context": {
                "Bolt Type Code": "ULN",
                "Bolt Type Description": "21.8mm Indented Ultra Strand Cable",
            },
        },
        "output": "Ultra Strand Indented 21.8mmX4.1M",
    },

    # -------------------------
    # Cable Matrix
    # -------------------------
    {
        "input": {
            "sheet_name": "Cable Matrix",
            "item_id": "CBD1B040DT05WA",
            "description": "C/BLT TWIN STRAND BULBED 4.0M",
            "context": {
                "Designator Code": "CB",
                "Designator Description": "Cable Bolt",
                "Freight element": "Freight Cost",
            },
        },
        "output": "Cable Bolt Twin Strand Bulbed 4.0M",
    },

    # -------------------------
    # Butterfly Plates
    # -------------------------
    {
        "input": {
            "sheet_name": "Butterfly",
            "item_id": "BUTTD30025SCM",
            "description": "B/FLY 300X280 STD SLOT C/W35MM",
            "context": {
                "Base Plate Type Code": "BUTTD",
                "Plate Type Description": "BUTTERFLY PLATES",
            },
        },
        "output": "Butterfly Plate 300X280 Standard Slot Complete With 35MM",
    },

    # -------------------------
    # Spanners
    # -------------------------
    {
        "input": {
            "sheet_name": "Dollies & Spanners Coal",
            "item_id": "SPD22EH40400Q",
            "description": "SPAN PD HEX22X400 DRV/EXT 36AF",
            "context": {
                "Type Code": "S",
                "Type Description": "Spanner",
                "Drive End": "Pixi Drive 22mm Hex",
            },
        },
        "output": "Spanner Pixi Drive Hex 22X400 Drive/Extension 36AF",
    },

    # -------------------------
    # Mine Hangers
    # -------------------------
    {
        "input": {
            "sheet_name": "Mine Hangers",
            "item_id": "MH7948",
            "description": "M/HANG NUT 20MM T/BAR NUT C/W",
            "context": {
                "Mine Hanger": "Mine Hanger",
                "Style Code": "7948",
                "Style Description": "Closed loop in 12mm round bar to suit 20mm Thread Bar (36AF Nut)",
                "1. Type Code": "79",
                "1. Type Description": "Nuts",
                "2. Type Code": "48",
            },
        },
        "output": "Mine Hanger Nut 20MM Thread Bar Nut Complete With",
    },
    {
        "input": {
            "sheet_name": "Mine Hangers",
            "item_id": "MH7948EZWA",
            "description": "M/HANG NUT 20MM T/BAR NUT WA",
            "context": {
                "Mine Hanger": "Mine Hanger",
                "Style Code": "7948",
                "Style Description": "Closed loop in 12mm round bar to suit 20mm Thread Bar (36AF Nut)",
                "1. Type Code": "79",
                "1. Type Description": "Nuts",
                "2. Type Code": "48",
                "Coating": "Electroplated Zinc",
                "Freight": "West Australia",
            },
        },
        "output": "Mine Hanger Nut 20MM Thread Bar Nut West Australia",
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


# ----------------------------
# Env + client (Gemini)
# ----------------------------
def init_gemini_client() -> genai.Client:
    # Path to this script: <project_root>/utilities/expand_abbreviation.py
    script_dir = Path(__file__).resolve().parent

    # Project root = one level up
    project_root = script_dir.parent

    dotenv_file_path = project_root / ".env.local"

    if not dotenv_file_path.exists():
        raise FileNotFoundError(f".env.local not found at: {dotenv_file_path}")

    load_dotenv(dotenv_path=dotenv_file_path)
    return genai.Client()


client = init_gemini_client()


# ----------------------------
# Prompt
# ----------------------------
def build_expand_prompt(
    sheet_name: str,
    item_id: str,
    description: str,
    context: Dict[str, str],
) -> str:
    """
    Output is ONLY the expanded description as a single line.
    No JSON, no id echoing.
    Includes few-shot examples.
    """
    # Put context in a stable readable format
    context_lines = "\n".join([f"- {k}: {v}" for k, v in context.items()]) if context else "- (none)"

    few_shots = _few_shot_block()

    return f"""
You expand abbreviations in industrial item descriptions using ONLY the provided context.

Hard rules:
- Do not invent information.
- Do not change any numbers, dimensions, units, standards, part numbers, or IDs.
- Expand abbreviations only if strongly supported by context; otherwise keep them unchanged.
- Keep the meaning and order of the description; do not add extra clauses.
- Return ONLY the expanded description as a SINGLE LINE of text.
- No JSON, no quotes, no markdown, no explanations.

{few_shots}

NOW EXPAND THIS ITEM:

INPUT:
{json.dumps({
    "sheet_name": sheet_name,
    "item_id": item_id,
    "description": description,
    "context": context
}, ensure_ascii=False, indent=2)}

OUTPUT (single line only):
""".strip()


# ----------------------------
# Model call with retries
# ----------------------------
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


# ----------------------------
# Output cleaning / validation
# ----------------------------
def clean_single_line(text: str) -> str:
    t = (text or "").strip()

    # remove common wrappers
    t = t.strip("`").strip()
    t = t.strip('"').strip("'").strip()

    # keep first non-empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    return lines[0] if lines else ""


def numbers_preserved(original: str, expanded: str) -> bool:
    import re
    orig_nums = re.findall(r"\d+(?:\.\d+)?", original)
    return all(n in expanded for n in orig_nums)


# ----------------------------
# Row extraction
# ----------------------------
def build_context(row: pd.Series, desc_col: str, id_col: str) -> Dict[str, str]:
    ctx = {}
    for k, v in row.items():
        if k in (desc_col, id_col):
            continue
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            ctx[str(k)] = s
    return ctx


def get_id_col(df: pd.DataFrame) -> str:
    if ID_COL_NAME and ID_COL_NAME in df.columns:
        return ID_COL_NAME
    # fallback: first column
    return df.columns[0]


# ----------------------------
# Main processing (sheet by sheet)
# ----------------------------
def expand_workbook(
    input_xlsx: str,
    output_xlsx: str,
    desc_col: str = DESC_COL,
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

        if desc_col not in df.columns:
            # keep sheet unchanged
            out_sheets[sheet_name] = df
            continue

        id_col = get_id_col(df)

        # Only process rows with non-empty description
        mask = df[desc_col].notna() & (df[desc_col].astype(str).str.strip() != "")
        work_df = df[mask].copy()

        if work_df.empty:
            df[out_col] = ""
            out_sheets[sheet_name] = df
            continue

        # Prepare tasks
        futures = {}
        results_by_item_id: Dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, row in work_df.iterrows():
                item_id = str(row[id_col]).strip()
                description = str(row[desc_col]).strip()
                context = build_context(row, desc_col=desc_col, id_col=id_col)
                prompt = build_expand_prompt(sheet_name, item_id, description, context)

                fut = executor.submit(call_llm_with_retries, prompt)
                futures[fut] = (idx, item_id, description)

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Expanding {sheet_name}"):
                idx, item_id, original_desc = futures[fut]
                raw = fut.result()

                expanded = clean_single_line(raw)

                # fallback behavior
                if (not expanded) or expanded.startswith("ERROR:"):
                    expanded = original_desc
                elif not numbers_preserved(original_desc, expanded):
                    expanded = original_desc

                # map back by id (NOT model output)
                results_by_item_id[item_id] = expanded

        # Write result column back by row (safer than only id, handles duplicate IDs too)
        df[out_col] = df[desc_col]  # default = original
        for idx, row in work_df.iterrows():
            item_id = str(row[id_col]).strip()
            df.at[idx, out_col] = results_by_item_id.get(item_id, str(row[desc_col]).strip())

        out_sheets[sheet_name] = df

    # Save to a new workbook
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet_name, df in out_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Saved: {output_xlsx}")


if __name__ == "__main__":
    expand_workbook(
        input_xlsx="/Users/sufarhansudaryo/Documents/Work/semantic-automapper/data/excel_files/items_logic_complete.xlsx",
        output_xlsx="items_expanded.xlsx",
        max_workers=10,
    )
