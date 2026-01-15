from __future__ import annotations

import argparse
import pickle
import time
import json
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from dotenv import load_dotenv


"""
Automatic Item-to-Class Assignment (Fuzzy + Cosine + Optional Gemini Rerank)
---------------------------------------------------------------------------

Purpose
-------
Assign the best matching taxonomy/class to each item by combining:
- Fuzzy string matching (RapidFuzz) to narrow the class search space,
- Embedding cosine similarity to rank candidate classes, and
- OPTIONAL Gemini reranking to choose the single best class from Top-K candidates.

Workflow (Two-phase)
--------------------
Phase A (always):
1) Load items + classes from Excel and embeddings from pickle.
2) If fuzzy hits exist for an item, expand matched classes into their lowest-level descendants
   using the class hierarchy (parent_id).
3) Compute cosine similarity against lowest-level class embeddings and store:
   - Top-N (default 3) cosine predictions for output
   - Top-K (default 5) candidates for the LLM stage

Phase B (optional, --use_llm):
4) After cosine is finished, send each item’s Top-K candidates + optional context columns to Gemini.
5) Gemini must pick exactly ONE candidate class_id and return JSON:
   {"choice_id": "...", "reasoning": "..."}
6) Store Gemini results in llm_class_id / llm_class_name / llm_reasoning.

Reliability Features
--------------------
- Autosave partial LLM results every N items: --autosave_every N (writes <output_base>.partial.xlsx)
- Resume from partial autosave: --resume_partial (skips items with llm_class_id already filled)
- Per-call Gemini timeout: --llm_timeout_s <seconds>
- Optional LLM gating: --llm_only_if_margin_below <m> (call LLM only if top1-top2 cosine margin < m)

Inputs Needed
-------------
- Items Excel: must include item_id_col + item_name_col (plus optional context columns)
- Classes Excel: must include class_id_col + class_name_col + parent_id
  (optional: lowest_level_col, level, and extra class context columns)
- Pickles: item_embeddings (item_id -> vector), class_embeddings (class_id -> vector)
- Gemini credentials (only if --use_llm), loaded from --dotenv_path or <project_root>/.env.local

Output
------
An Excel file with:
- Top-1/2/3 cosine predictions (+ similarity)
- Fuzzy debug columns
- Optional Gemini rerank fields (llm_*),
merged back with the original items and Top-1 class metadata.

Example
-------
python assign.py --items_embeddings items.pkl --classes_embeddings classes.pkl \
  --items_excel items.xlsx --classes_excel classes.xlsx \
  --item_id_col item_id --item_name_col item_name \
  --class_id_col class_id --class_name_col class_name \
  --use_llm --llm_top_k 5 --autosave_every 100 --resume_partial \
  --llm_timeout_s 60 --output assignments.xlsx
"""



# Utilities
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def fuzzy_match_score(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    return max(
        fuzz.partial_ratio(a, b),
        fuzz.token_set_ratio(a, b),
    ) / 100.0


def parse_cols_arg(cols: Optional[str]) -> Optional[List[str]]:
    if not cols:
        return None
    cols = cols.strip()
    if not cols:
        return None
    return [c.strip().lower() for c in cols.split(",") if c.strip()]


def build_context_from_row(
    row: pd.Series,
    *,
    exclude_cols: set,
    include_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    ctx: Dict[str, str] = {}
    for k, v in row.items():
        key = str(k).strip().lower()
        if key in exclude_cols:
            continue
        if include_cols is not None and key not in include_cols:
            continue
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            ctx[str(k)] = s  # keep original header name
    return ctx


# Few-shot examples for reranking (EDIT THESE)
RERANK_FEW_SHOTS: List[Dict[str, Any]] = [
    {
        "input": {
            "item": {
                "item_name": "FRICTION BOLT 47 DBW X 900MM GAL WA",
                "item_context": {
                    "Type Code": "FB",
                    "Type Description": "Friction Bolt",
                    "Coating": "Galvanised",
                },
            },
            "candidates": [
                {"class_id": "197", "class_name": "39 mm FB", "cosine_similarity": 0.7713},
                {"class_id": "199", "class_name": "33 mm FB", "cosine_similarity": 0.7433},
                {"class_id": "200", "class_name": "47 mm FB", "cosine_similarity": 0.7147},
            ],
        },
        "output": {
            "choice_id": "200",
            "reasoning": "The item name explicitly specifies a 47 mm friction bolt; among the candidates, '47 mm FB' matches this diameter best. Cosine similarity is lower, but the diameter match is the most decisive signal here.",
        },
    },
    {
        "input": {
            "item": {
                "item_name": "PLATE DOME HANDLE BAR 150X150X5MM",
                "item_context": {
                    "Type Code": "D",
                    "Type Description": "Dome Plate",
                    "Coating": "Stainless Steel",
                },
            },
            "candidates": [
                {"class_id": "155", "class_name": "Domed Omega Bolt Plates", "cosine_similarity": 0.8425},
                {"class_id": "156", "class_name": "Combi Plates / X-plates for MD-MDX", "cosine_similarity": 0.8324},
                {"class_id": "157", "class_name": "Flat Plates", "cosine_similarity": 0.8120},
            ],
        },
        "output": {
            "choice_id": "155",
            "reasoning": "The item is explicitly a dome plate (confirmed by both the item name and context). Among the candidates, 'Domed Omega Bolt Plates' is the only domed plate option and matches best.",
        },
    },
]


def _few_shot_block() -> str:
    blocks: List[str] = []
    for ex in RERANK_FEW_SHOTS:
        blocks.append(
            "EXAMPLE INPUT:\n"
            f"{json.dumps(ex['input'], ensure_ascii=False, indent=2)}\n"
            "EXAMPLE OUTPUT:\n"
            f"{json.dumps(ex['output'], ensure_ascii=False)}\n"
        )
    return "\n".join(blocks)


# Gemini client + call
def init_gemini_client(dotenv_path: Optional[str] = None) -> genai.Client:
    if dotenv_path:
        dp = Path(dotenv_path).expanduser().resolve()
        if not dp.exists():
            raise FileNotFoundError(f"dotenv file not found: {dp}")
        load_dotenv(dotenv_path=dp)
        return genai.Client()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    default_dp = project_root / ".env.local"
    if default_dp.exists():
        load_dotenv(dotenv_path=default_dp)
    return genai.Client()


def call_gemini_with_retries(
    client: genai.Client,
    prompt: str,
    *,
    model: str,
    retries: int,
    delay: float,
    timeout_s: float = 0.0,  # NEW
) -> str:
    def _one_call() -> str:
        resp = client.models.generate_content(model=model, contents=prompt)
        return (resp.text or "").strip()

    for attempt in range(retries):
        try:
            if timeout_s and timeout_s > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(_one_call)
                    return fut.result(timeout=timeout_s)
            else:
                return _one_call()
        except concurrent.futures.TimeoutError:
            err = f"ERROR: Gemini call timed out after {timeout_s}s"
        except Exception as e:
            err = f"ERROR: {str(e)}"

        if attempt < retries - 1:
            time.sleep(delay)
        else:
            return err

    return "ERROR: Unknown error"


def clean_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    t = (text or "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except Exception:
            return None
    return None


def build_rerank_prompt(
    *,
    item_name: str,
    item_context: Dict[str, str],
    candidates: List[Dict[str, Any]],
) -> str:
    few_shots = _few_shot_block()

    payload = {
        "item": {
            "item_name": item_name,
            "item_context": item_context,
        },
        "candidates": candidates,  # can be < K, totally fine
    }

    return f"""
You are an expert industrial taxonomist. Your job is to select the SINGLE best matching class for an item.

You are given:
1) The item name (often short and noisy)
2) Optional structured item context (columns from Excel)
3) A short list of candidate classes (Top-K from embedding cosine similarity)

Your output MUST select ONE class_id from the provided candidates.

----------------------------
DECISION GOAL
----------------------------
Choose the candidate class that best matches the REAL meaning of the item.
Prefer the class that a human domain expert would choose in a catalog / ERP system.

----------------------------
HOW TO USE INFORMATION
----------------------------
Use ALL of these signals:
- Item name tokens (keywords, abbreviations, model terms)
- Structured context fields (material, standard, discipline, dimensions, type, etc.)
- Candidate class names
- Candidate class_context (if provided)
- Cosine similarity as a helpful hint (NOT the final truth)

----------------------------
IMPORTANT RULES
----------------------------
1) You MUST choose exactly ONE of the candidate class_ids.
2) NEVER invent a new class_id.
3) If the item context directly indicates one candidate (e.g. standard DIN/ISO, material, discipline), prioritize that.
4) If two candidates are similar:
   - prefer the more specific / precise class (e.g. "Hex bolt DIN 933" over "Bolts")
5) If a candidate class is clearly wrong category (e.g. washer vs bolt), reject it even if similarity is high.
6) Numbers (dimensions) may help but do not overfit to them.
7) If the item is ambiguous, select the most general candidate that is still correct.

----------------------------
OUTPUT FORMAT (STRICT)
----------------------------
Return ONLY valid JSON with exactly these keys:
- "choice_id": string (must match one candidate class_id)
- "reasoning": short explanation (1–3 sentences)

No markdown, no extra text, no bullet points.

{few_shots}

----------------------------
NOW CLASSIFY THIS ITEM
----------------------------

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}

OUTPUT (JSON only):
""".strip()


def llm_choose_best_class(
    *,
    client: genai.Client,
    model: str,
    item_name: str,
    item_context: Dict[str, str],
    candidates: List[Dict[str, Any]],
    retries: int,
    retry_delay: float,
    llm_timeout_s: float = 0.0,  # NEW
) -> Tuple[Optional[str], str]:
    prompt = build_rerank_prompt(item_name=item_name, item_context=item_context, candidates=candidates)
    raw = call_gemini_with_retries(
        client,
        prompt,
        model=model,
        retries=retries,
        delay=retry_delay,
        timeout_s=llm_timeout_s,
    )

    if raw.startswith("ERROR:"):
        return None, raw

    obj = clean_json_from_text(raw)
    if not obj:
        return None, f"Unparseable LLM output: {raw[:400]}"

    choice_id = obj.get("choice_id")
    reasoning = str(obj.get("reasoning", "")).strip()

    valid_ids = {str(c["class_id"]) for c in candidates}
    if str(choice_id) not in valid_ids:
        return None, reasoning or f"Invalid choice_id: {choice_id}"

    return str(choice_id), reasoning


# Core logic (2-phase)
def assign_classes(
    items_df: pd.DataFrame,
    classes_df: pd.DataFrame,
    item_embeddings: Dict[Any, np.ndarray],
    class_embeddings: Dict[Any, np.ndarray],
    item_id_col: str,
    item_name_col: str,  # used for fuzzy + cosine pipeline
    class_id_col: str,
    class_name_col: str,
    lowest_level_col: str = "lowest_level_class",
    fuzzy_threshold: float = 0.6,
    batch_size: int = 256,
    # Output:
    final_top_n: int = 3,  # output cosine top-3
    llm_top_k: int = 5,  # send top-5 cosine candidates to LLM
    # LLM:
    use_llm: bool = False,
    llm_model: str = "gemini-2.5-flash",
    dotenv_path: Optional[str] = None,
    item_context_cols: Optional[List[str]] = None,
    class_context_cols: Optional[List[str]] = None,
    llm_retries: int = 3,
    llm_retry_delay: float = 2.0,
    llm_only_if_margin_below: Optional[float] = None,
    llm_item_name_col: Optional[str] = None,  # LLM uses this column for item name (e.g. "description")
    # NEW:
    autosave_every: int = 0,
    autosave_path: Optional[str] = None,
    resume_partial: bool = False,
    llm_timeout_s: float = 0.0,
) -> pd.DataFrame:
    """
    Two-phase processing:
    1) Compute cosine similarities for all items (and store top-K candidates for LLM).
    2) Run LLM rerank after cosine is finished (with its own progress bar).

    NEW:
    - Autosave partial results every N LLM items
    - Resume from partial autosave (skip items already done + carry their llm_* fields forward)
    - Per-call timeout for Gemini
    """

    if llm_item_name_col is None:
        llm_item_name_col = item_name_col

    if autosave_path is None:
        autosave_path = "autosave_partial.xlsx"

    results: List[Dict[str, Any]] = []
    classes_df_all = classes_df.copy()

    # lowest-level for cosine
    if lowest_level_col in classes_df.columns:
        classes_df_lowest = classes_df[classes_df[lowest_level_col] == True]
        print(f"Using only lowest-level classes for cosine similarity: {len(classes_df_lowest)} entries")
    else:
        classes_df_lowest = classes_df
        print("No lowest-level column found. Using all classes for similarity.")

    # ALL classes list for fuzzy
    all_class_ids = classes_df_all[class_id_col].tolist()
    all_class_names = classes_df_all[class_name_col].astype(str).tolist()
    all_class_levels = (
        classes_df_all["level"].tolist()
        if "level" in classes_df_all.columns
        else [None] * len(all_class_ids)
    )

    # lowest-level vectors for cosine
    class_ids: List[Any] = []
    class_names: List[str] = []
    class_vectors: List[np.ndarray] = []
    for _, row in classes_df_lowest.iterrows():
        cid = row[class_id_col]
        if cid in class_embeddings:
            class_ids.append(cid)
            class_names.append(str(row[class_name_col]))
            class_vectors.append(class_embeddings[cid])
    class_vectors_np = np.array(class_vectors)
    print(f"Loaded {len(class_vectors_np)} lowest-level class embeddings.\n")

    class_id_to_idx: Dict[Any, int] = {cid: i for i, cid in enumerate(class_ids)}

    # class row lookup for class_context
    classes_by_id: Dict[Any, pd.Series] = {row[class_id_col]: row for _, row in classes_df_all.iterrows()}

    # hierarchy children map
    if "parent_id" not in classes_df_all.columns:
        raise ValueError("classes_df must contain a 'parent_id' column for hierarchy expansion.")

    children_map: Dict[Any, List[Any]] = {}
    for _, row in classes_df_all.iterrows():
        pid = row["parent_id"]
        cid = row[class_id_col]
        children_map.setdefault(pid, []).append(cid)

    # descendants cache
    descendants_cache: Dict[Any, List[Any]] = {}

    def get_lowest_descendants(class_id: Any) -> List[Any]:
        if class_id in descendants_cache:
            return descendants_cache[class_id]

        row = classes_df_all.loc[classes_df_all[class_id_col] == class_id]
        if not row.empty and row[lowest_level_col].iloc[0] == True:
            descendants_cache[class_id] = [class_id]
            return [class_id]

        if class_id not in children_map:
            descendants_cache[class_id] = []
            return []

        result: List[Any] = []
        for child in children_map[class_id]:
            result.extend(get_lowest_descendants(child))

        descendants_cache[class_id] = result
        return result

    # Phase 0: group items (fuzzy vs non-fuzzy)
    fuzzy_groups = []
    nonfuzzy_items = []

    print("Grouping items based on fuzzy match results...")
    for item_id, item_vec in tqdm(item_embeddings.items(), desc="Fuzzy grouping", unit="item"):
        row = items_df.loc[items_df[item_id_col] == item_id]
        if row.empty:
            continue
        item_name = str(row[item_name_col].iloc[0])

        fuzzy_hits = [
            i
            for i, cname in enumerate(all_class_names)
            if fuzzy_match_score(item_name, cname) >= fuzzy_threshold
        ]

        if fuzzy_hits:
            candidate_indices: List[int] = []
            for idx in fuzzy_hits:
                matched_class_id = all_class_ids[idx]
                lowest_descendants = get_lowest_descendants(matched_class_id)
                for cid in lowest_descendants:
                    ci = class_id_to_idx.get(cid)
                    if ci is not None:
                        candidate_indices.append(ci)

            candidate_indices = sorted(set(candidate_indices))

            candidate_names_disp: List[str] = []
            for i in fuzzy_hits[:20]:
                level = all_class_levels[i]
                if level is not None and str(level) != "":
                    candidate_names_disp.append(f"{all_class_names[i]} (L{level})")
                else:
                    candidate_names_disp.append(all_class_names[i])
            if len(fuzzy_hits) > 20:
                candidate_names_disp.append(f"...(+{len(fuzzy_hits) - 20} more)")

            fuzzy_groups.append((item_id, item_name, item_vec, candidate_indices, candidate_names_disp))
        else:
            nonfuzzy_items.append((item_id, item_name, item_vec))

    print("\nGrouping complete!")
    print(f"  - Items with fuzzy matches: {len(fuzzy_groups)}")
    print(f"  - Items with no fuzzy matches: {len(nonfuzzy_items)}\n")

    # Phase A: cosine for ALL items (store top3 + topK candidates info for LLM)
    cosine_payload_by_item: Dict[Any, Dict[str, Any]] = {}

    pbar_cos = tqdm(total=len(item_embeddings), desc="Cosine similarity (all items)", unit="item")

    def compute_sims_for_item(item_vec: np.ndarray, eval_indices: List[int]) -> List[Tuple[int, float]]:
        item_vec_2d = item_vec.reshape(1, -1)
        sims: List[Tuple[int, float]] = []
        for i in range(0, len(eval_indices), batch_size):
            batch_idx = eval_indices[i : i + batch_size]
            batch = class_vectors_np[batch_idx]
            sim = cosine_similarity(batch, item_vec_2d).flatten()
            sims.extend(zip(batch_idx, sim))
        sims_sorted_full = sorted(sims, key=lambda x: x[1], reverse=True)
        return sims_sorted_full

    # fuzzy group
    for item_id, item_name, item_vec, candidate_indices, candidate_names_disp in fuzzy_groups:
        eval_indices = candidate_indices if candidate_indices else list(range(len(class_vectors_np)))
        sims_sorted_full = compute_sims_for_item(item_vec, eval_indices)

        cosine_payload_by_item[item_id] = {
            "item_name_pipeline": item_name,
            "candidate_indices": candidate_indices,
            "candidate_names_disp": candidate_names_disp,
            "sims_sorted_full": sims_sorted_full,
            "sims_sorted_topk": sims_sorted_full[:llm_top_k],
        }
        pbar_cos.update(1)

    # non-fuzzy group
    all_indices = list(range(len(class_vectors_np)))
    for item_id, item_name, item_vec in nonfuzzy_items:
        sims_sorted_full = compute_sims_for_item(item_vec, all_indices)
        cosine_payload_by_item[item_id] = {
            "item_name_pipeline": item_name,
            "candidate_indices": [],
            "candidate_names_disp": ["No fuzzy match"],
            "sims_sorted_full": sims_sorted_full,
            "sims_sorted_topk": sims_sorted_full[:llm_top_k],
        }
        pbar_cos.update(1)

    pbar_cos.close()

    # Prepare base results rows (cosine-only output always filled)
    def append_base_row(item_id: Any, item_name_pipeline: str, payload: Dict[str, Any]):
        sims_sorted_full: List[Tuple[int, float]] = payload["sims_sorted_full"]
        candidate_indices: List[int] = payload["candidate_indices"]
        candidate_names_disp: List[str] = payload["candidate_names_disp"]

        row_out: Dict[str, Any] = {
            item_id_col: item_id,
            item_name_col: item_name_pipeline,
            "fuzzy_candidates_count": len(candidate_indices),
            "fuzzy_candidate_names": "; ".join(candidate_names_disp),
            "llm_class_id": None,
            "llm_class_name": None,
            "llm_reasoning": "",
        }

        top3 = sims_sorted_full[:final_top_n]
        for rank in range(final_top_n):
            if rank < len(top3):
                idx, sim = top3[rank]
                row_out[f"top_{rank+1}_class_id"] = class_ids[idx]
                row_out[f"top_{rank+1}_class_name"] = class_names[idx]
                row_out[f"top_{rank+1}_similarity"] = round(float(sim), 4)
            else:
                row_out[f"top_{rank+1}_class_id"] = None
                row_out[f"top_{rank+1}_class_name"] = None
                row_out[f"top_{rank+1}_similarity"] = None

        results.append(row_out)

    # create rows in stable order (items_df order that exist in embeddings)
    emb_id_set = set(item_embeddings.keys())
    for item_id in items_df[item_id_col].tolist():
        if item_id in emb_id_set and item_id in cosine_payload_by_item:
            append_base_row(
                item_id=item_id,
                item_name_pipeline=cosine_payload_by_item[item_id]["item_name_pipeline"],
                payload=cosine_payload_by_item[item_id],
            )

    # Phase B: LLM rerank AFTER cosine is done (optional)
    if not use_llm:
        print("\nLLM disabled. Returning cosine-only results.\n")
        return pd.DataFrame(results)

    # init LLM client
    gemini_client = init_gemini_client(dotenv_path=dotenv_path)

    # cache identical prompts
    llm_cache: Dict[str, Tuple[Optional[str], str]] = {}

    def should_call_llm(sims_sorted_topk: List[Tuple[int, float]]) -> bool:
        if llm_only_if_margin_below is None:
            return True
        if len(sims_sorted_topk) < 2:
            return True
        margin = float(sims_sorted_topk[0][1]) - float(sims_sorted_topk[1][1])
        return margin < float(llm_only_if_margin_below)

    # index results by item_id for fast fill
    result_idx_by_item: Dict[Any, int] = {row[item_id_col]: i for i, row in enumerate(results)}

    # ----------------------------
    # RESUME: load partial autosave (if requested) and carry over llm_* fields
    # ----------------------------
    done_ids: set = set()
    carried_llm: Dict[Any, Dict[str, Any]] = {}

    if resume_partial:
        ap = Path(autosave_path)
        if ap.exists():
            try:
                prev = pd.read_excel(ap)
                prev.columns = prev.columns.str.strip().str.lower()
                if item_id_col in prev.columns and "llm_class_id" in prev.columns:
                    # items that are "done"
                    done = prev[
                        prev["llm_class_id"].notna()
                        & (prev["llm_class_id"].astype(str).str.strip().str.len() > 0)
                    ].copy()
                    done_ids = set(done[item_id_col].tolist())

                    # carry llm_* results into current results
                    for _, r in done.iterrows():
                        carried_llm[r[item_id_col]] = {
                            "llm_class_id": r.get("llm_class_id", None),
                            "llm_class_name": r.get("llm_class_name", None),
                            "llm_reasoning": r.get("llm_reasoning", ""),
                        }

                    # apply carried fields
                    for iid, fields in carried_llm.items():
                        ridx = result_idx_by_item.get(iid)
                        if ridx is not None:
                            results[ridx]["llm_class_id"] = fields["llm_class_id"]
                            results[ridx]["llm_class_name"] = fields["llm_class_name"]
                            results[ridx]["llm_reasoning"] = fields["llm_reasoning"] or ""

                    print(f"Resume enabled: loaded {len(done_ids)} completed LLM items from '{ap.name}'.")
            except Exception as e:
                print(f"Resume warning: failed to read '{autosave_path}': {e}")

    def _autosave_now():
        try:
            pd.DataFrame(results).to_excel(autosave_path, index=False)
        except Exception as e:
            print(f"Autosave failed ({autosave_path}): {e}")

    # build LLM worklist (skip already done)
    llm_items: List[Any] = []
    for item_id, payload in cosine_payload_by_item.items():
        if item_id in done_ids:
            continue
        if should_call_llm(payload["sims_sorted_topk"]):
            llm_items.append(item_id)

    print(f"\nLLM phase: {len(llm_items)} / {len(cosine_payload_by_item)} items will be sent to Gemini.\n")
    pbar_llm = tqdm(total=len(llm_items), desc="Gemini rerank", unit="item")

    for item_id in llm_items:
        payload = cosine_payload_by_item[item_id]
        sims_sorted_topk: List[Tuple[int, float]] = payload["sims_sorted_topk"]

        item_row = items_df.loc[items_df[item_id_col] == item_id]
        if item_row.empty:
            pbar_llm.update(1)
            continue

        llm_item_name = (
            str(item_row[llm_item_name_col].iloc[0])
            if (llm_item_name_col in item_row.columns)
            else payload["item_name_pipeline"]
        )

        item_ctx: Dict[str, str] = build_context_from_row(
            item_row.iloc[0],
            exclude_cols={item_id_col, item_name_col, llm_item_name_col},
            include_cols=item_context_cols,
        )

        candidates: List[Dict[str, Any]] = []
        for idx, sim in sims_sorted_topk:
            cid = class_ids[idx]
            cdict: Dict[str, Any] = {
                "class_id": str(cid),
                "class_name": class_names[idx],
                "cosine_similarity": round(float(sim), 4),
            }
            if class_context_cols is not None:
                crow = classes_by_id.get(cid)
                if crow is not None:
                    cdict["class_context"] = build_context_from_row(
                        crow,
                        exclude_cols={class_id_col, class_name_col},
                        include_cols=class_context_cols,
                    )
            candidates.append(cdict)

        cache_key = json.dumps(
            {"item_name": llm_item_name, "item_ctx": item_ctx, "candidates": candidates},
            ensure_ascii=False,
        )

        if cache_key in llm_cache:
            choice_id, reasoning = llm_cache[cache_key]
        else:
            choice_id, reasoning = llm_choose_best_class(
                client=gemini_client,
                model=llm_model,
                item_name=llm_item_name,
                item_context=item_ctx,
                candidates=candidates,
                retries=llm_retries,
                retry_delay=llm_retry_delay,
                llm_timeout_s=llm_timeout_s,
            )
            llm_cache[cache_key] = (choice_id, reasoning)

        choice_name = None
        if choice_id is not None:
            for c in candidates:
                if str(c["class_id"]) == str(choice_id):
                    choice_name = str(c["class_name"])
                    break

        ridx = result_idx_by_item.get(item_id)
        if ridx is not None:
            results[ridx]["llm_class_id"] = choice_id
            results[ridx]["llm_class_name"] = choice_name
            results[ridx]["llm_reasoning"] = reasoning or ""

        # autosave every N completed LLM items
        if autosave_every and autosave_every > 0:
            completed_after_this = pbar_llm.n + 1
            if completed_after_this % autosave_every == 0:
                _autosave_now()

        pbar_llm.update(1)

    pbar_llm.close()

    # final autosave (if enabled)
    if autosave_every and autosave_every > 0:
        _autosave_now()

    print("\nMatching completed!\n")
    return pd.DataFrame(results)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Item-to-class matching using fuzzy + cosine + optional Gemini rerank (two-phase, with Gemini progress bar)."
    )

    parser.add_argument("--items_embeddings", required=True)
    parser.add_argument("--classes_embeddings", required=True)
    parser.add_argument("--items_excel", required=True)
    parser.add_argument("--classes_excel", required=True)

    parser.add_argument("--item_id_col", required=True)
    parser.add_argument("--item_name_col", required=True)  # used for fuzzy/cosine
    parser.add_argument(
        "--llm_item_name_col",
        default=None,
        help='Column name to use ONLY for LLM item_name (e.g. "description"). Defaults to item_name_col.',
    )
    parser.add_argument("--class_id_col", required=True)
    parser.add_argument("--class_name_col", required=True)
    parser.add_argument("--lowest_level_col", default="lowest_level_class")

    parser.add_argument("--fuzzy_threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=256)

    # Sampling for quick tests
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Randomly sample N items from items_excel for quick testing (only items that have embeddings).",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=42,
        help="Random seed for --sample_n (default 42).",
    )

    # LLM candidates provided to LLM
    parser.add_argument("--llm_top_k", type=int, default=5, help="How many cosine candidates to provide to the LLM.")

    # LLM args (Gemini)
    parser.add_argument("--use_llm", action="store_true", help="Enable Gemini reranking using Top-K candidates.")
    parser.add_argument("--llm_model", default="gemini-2.5-flash")
    parser.add_argument("--dotenv_path", default=None, help="Path to your .env.local (optional).")
    parser.add_argument("--item_context_cols", default=None, help='Comma list, e.g. "colA,colB,colC" (optional).')
    parser.add_argument("--class_context_cols", default=None, help='Comma list, e.g. "level,parent_id" (optional).')
    parser.add_argument("--llm_retries", type=int, default=3)
    parser.add_argument("--llm_retry_delay", type=float, default=2.0)
    parser.add_argument(
        "--llm_only_if_margin_below",
        type=float,
        default=None,
        help="Only call LLM if (top1 - top2) cosine margin is below this value, e.g. 0.03",
    )

    # NEW: autosave/resume/timeout
    parser.add_argument(
        "--autosave_every",
        type=int,
        default=0,
        help="Autosave partial results every N LLM items (0 disables).",
    )
    parser.add_argument(
        "--resume_partial",
        action="store_true",
        help="If partial autosave exists, resume and skip items that already have llm_class_id filled.",
    )
    parser.add_argument(
        "--llm_timeout_s",
        type=int,
        default=0,
        help="Timeout (seconds) for each Gemini call. 0 disables.",
    )

    parser.add_argument("--output", "-o", default="item_class_assignments_merged.xlsx")

    args = parser.parse_args()

    print("\nLoading data ...")
    items_df = pd.read_excel(args.items_excel)
    classes_df = pd.read_excel(args.classes_excel)
    item_embeddings = load_pickle(args.items_embeddings)
    class_embeddings = load_pickle(args.classes_embeddings)

    # normalize columns
    items_df.columns = items_df.columns.str.strip().str.lower()
    classes_df.columns = classes_df.columns.str.strip().str.lower()

    args.item_id_col = args.item_id_col.strip().lower()
    args.item_name_col = args.item_name_col.strip().lower()
    args.class_id_col = args.class_id_col.strip().lower()
    args.class_name_col = args.class_name_col.strip().lower()
    args.lowest_level_col = args.lowest_level_col.strip().lower()

    llm_item_name_col = args.llm_item_name_col.strip().lower() if args.llm_item_name_col else None

    item_context_cols = parse_cols_arg(args.item_context_cols)
    class_context_cols = parse_cols_arg(args.class_context_cols)

    print("Normalized column names for robust matching.")

    # Optional random sampling for quick tests
    if args.sample_n is not None:
        n = int(args.sample_n)
        if n <= 0:
            raise ValueError("--sample_n must be a positive integer")

        emb_ids = set(item_embeddings.keys())
        items_with_emb = items_df[items_df[args.item_id_col].isin(emb_ids)].copy()

        if items_with_emb.empty:
            raise ValueError("No overlap between items_df IDs and item_embeddings keys.")

        if n >= len(items_with_emb):
            print(f"--sample_n={n} >= available rows with embeddings ({len(items_with_emb)}). Using all.")
            sampled_items_df = items_with_emb
        else:
            sampled_items_df = items_with_emb.sample(n=n, random_state=args.sample_seed)

        sampled_ids = set(sampled_items_df[args.item_id_col].tolist())

        items_df = sampled_items_df.reset_index(drop=True)
        item_embeddings = {k: v for k, v in item_embeddings.items() if k in sampled_ids}

        print(f"Sampling enabled: using {len(items_df)} items (seed={args.sample_seed}).")

    # autosave file sits next to output (same base name)
    out_path = Path(args.output)
    autosave_path = out_path.with_suffix("").as_posix() + ".partial.xlsx"

    results_df = assign_classes(
        items_df=items_df,
        classes_df=classes_df,
        item_embeddings=item_embeddings,
        class_embeddings=class_embeddings,
        item_id_col=args.item_id_col,
        item_name_col=args.item_name_col,
        llm_item_name_col=llm_item_name_col,
        class_id_col=args.class_id_col,
        class_name_col=args.class_name_col,
        lowest_level_col=args.lowest_level_col,
        fuzzy_threshold=args.fuzzy_threshold,
        batch_size=args.batch_size,
        final_top_n=3,
        llm_top_k=args.llm_top_k,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        dotenv_path=args.dotenv_path,
        item_context_cols=item_context_cols,
        class_context_cols=class_context_cols,
        llm_retries=args.llm_retries,
        llm_retry_delay=args.llm_retry_delay,
        llm_only_if_margin_below=args.llm_only_if_margin_below,
        autosave_every=int(args.autosave_every or 0),
        autosave_path=autosave_path,
        resume_partial=bool(args.resume_partial),
        llm_timeout_s=float(args.llm_timeout_s or 0),
    )

    # Merge results with items
    print("\nMerging results...")
    merged = results_df.merge(items_df, on=args.item_id_col, how="left")

    # Merge class information of cosine top-1 (top_1_class_id)
    class_merge = classes_df.add_prefix("class_")
    merged = merged.merge(
        class_merge,
        left_on="top_1_class_id",
        right_on=f"class_{args.class_id_col}",
        how="left",
    )

    merged.to_excel(args.output, index=False)
    print(f"\nDone! Saved merged file with {len(merged)} rows to '{args.output}'")

    if args.autosave_every and int(args.autosave_every) > 0:
        print(f"Partial autosave file: '{autosave_path}'")


if __name__ == "__main__":
    main()
