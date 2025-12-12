#!/usr/bin/env python3
"""
Extractor: Value + Unit Extraction using Qwen2.5-3B (MLX, Apple Silicon)
------------------------------------------------------------------------

Features:
- Loads Qwen2.5-3B normally (FP16) using MLX-LM (no quantization needed)
- Batch inference for speed (configurable batch size)
- Few-shot prompting for accuracy (Siemens-style extraction)
- JSON output for reliable parsing
- SI-unit normalization (mm, cm, m, km) only
- Preserves uppercase acronyms (GSM, PSI, kN, Nm, Pa, °C)
- Outputs new columns: extracted_values, extracted_units
"""

import argparse
import json
import pandas as pd
from tqdm import tqdm
from mlx_lm import load, batch_generate


# ---------------- GLOBAL MODEL ----------------
_model = None
_tokenizer = None

def get_model(model_name: str):
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    print(f"[INFO] Loading model: {model_name}")
    _model, _tokenizer = load(path_or_hf_repo=model_name)
    print("[INFO] Model loaded.\n")

    return _model, _tokenizer


# ---------------- FEW-SHOT PROMPT ----------------

FEW_SHOT = """
You are a precise extractor for engineering product attributes.
Your job: extract ONLY numeric values that have valid SI length units.

IMPORTANT RULES:
- For each input, output exactly ONE JSON object: {"values": [...], "units": [...]}
- Never output a JSON list.
- Never explain.
- Values and units must align by index.
- All values MUST be strings in JSON.
- SI length units MUST be normalized to lowercase:
      MM → mm
      CM → cm
      M  → m
      KM → km
- Keep uppercase technical units unchanged (GSM, PSI, kN, Nm, Pa, °C).

UNIT NORMALIZATION RULE:
When the entire input description is written in uppercase, SI units tend
to appear uppercase accidentally (e.g., "800MM", "24CM", "2.1M").
Normalize SI length units to lowercase even if the text is all uppercase.
But do NOT lowercase technical acronyms (GSM, PSI, etc.).

Few-shot examples:

Input: "GEWI PLUS CHUCK EXTENSION - 800mm"
Output: {"values": ["800"], "units": ["mm"]}

Input: "47.0MM L = 2.1m"
Output: {"values": ["47.0", "2.1"], "units": ["mm", "m"]}

Input: "WYE; VENT; 1220MM; EVERTUFF;515 GSM"
Output: {"values": ["1220"], "units": ["mm"]}

Input: "CHUCK EXTENSION - 800mm"
Output: {"values": ["800"], "units": ["mm"]}

Input: "EXPANSION SHELL 20mmR/HAND THD (PRE GROUTED BOLT)"
Output: {"values": ["20"], "units": ["mm"]}

Input: "BOLT AX 2100X24MM C/W AF/BALL/MOLYNUT"
Output: {"values": ["2100x24"], "units": ["mm"]}

Input: "FLAT PLATE 180 X 180 X 45MM"
Output: {"values": ["180x180x45"], "units": ["mm"]}

Input: "KINLOC 47MMX2400MM INDIE GAL"
Output: {"values": ["47x2400"], "units": ["mm"]}

Input: "NUT/PLATE, PULL TEST EQUIPMENT"
Output: {"values": [], "units": []}
"""


# ---------------- BUILD PROMPT ----------------

def build_prompt(text: str):
    return FEW_SHOT + f'\n\nInput: "{text}"\nOutput:'



def extract_batch(texts, model_name):
    """Run inference on a batch of texts and return list[dict]."""

    model, tokenizer = get_model(model_name)

    encoded_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(t)}],
            add_generation_prompt=True,
        )
        for t in texts
    ]

    result = batch_generate(
        model,
        tokenizer,
        encoded_prompts,
        max_tokens=128,
        verbose=False
    )

    outputs = []
    for raw in result.texts:
        cleaned = raw.strip()

        # ---- Robust JSON parsing ----
        try:
            parsed = json.loads(cleaned)

            # MODEL MAY RETURN LISTS → FIX IT
            if isinstance(parsed, list):
                if len(parsed) > 0 and isinstance(parsed[0], dict):
                    parsed = parsed[0]
                else:
                    parsed = {"values": [], "units": []}

            # If model returns anything weird (string, number, bool)
            if not isinstance(parsed, dict):
                parsed = {"values": [], "units": []}

        except Exception:
            parsed = {"values": [], "units": []}

        # Normalize values to strings
        parsed["values"] = [str(v) for v in parsed.get("values", [])]
        parsed["units"] = [str(u) for u in parsed.get("units", [])]

        outputs.append(parsed)

    return outputs



# ---------------- PROCESS EXCEL ----------------

def process_excel(input_path, output_path, column_name, model_name, batch_size=8, limit=None):

    print(f"[INFO] Reading Excel: {input_path}")
    df = pd.read_excel(input_path)

    # NEW: apply row limit
    if limit is not None:
        print(f"[INFO] Limiting to first {limit} rows.")
        df = df.head(limit)

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in Excel.")

    texts = df[column_name].fillna("").astype(str).tolist()

    extracted_vals = []
    extracted_units = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch = texts[i:i+batch_size]
        results = extract_batch(batch, model_name)

        for r in results:
            extracted_vals.append("; ".join(r.get("values", [])))
            extracted_units.append("; ".join(r.get("units", [])))

    df["extracted_values"] = extracted_vals
    df["extracted_units"] = extracted_units

    df.to_excel(output_path, index=False)
    print(f"[INFO] Saved results to: {output_path}")


# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch", type=int, default=8)

    # NEW: limit number of rows processed
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of rows to process (optional)")

    args = parser.parse_args()

    process_excel(args.input, args.output, args.column, args.model, args.batch, args.limit)



if __name__ == "__main__":
    main()
