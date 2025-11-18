import argparse
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

"""
Automatic Item-to-Class Assignment using Fuzzy + Embedding Similarity (Top-3 Version)
-------------------------------------------------------------------------------------

Process:
1. Fuzzy string matching (RapidFuzz)
   - Finds classes whose names partially match the item name.
   - Groups items with fuzzy matches for selective cosine comparison.
2. Cosine similarity (scikit-learn)
   - For fuzzy-matched items → compare only with their fuzzy candidates.
   - For non-matched items → compare with all classes.

Output:
A merged Excel file containing:
 - All original item columns
 - Top 3 matched class columns (IDs, names, similarities)
 - Fuzzy match info (candidate count, candidate names)

Example usage:
--------------
python semantic_automapper.py \
  --items_embeddings item_embeddings.pkl \
  --classes_embeddings class_embeddings.pkl \
  --items_excel items.xlsx \
  --classes_excel classes.xlsx \
  --item_id_col ID \
  --item_name_col Description \
  --class_id_col ID \
  --class_name_col name \
  --lowest_level_col lowest_level_class \
  --fuzzy_threshold 0.65 \
  --batch_size 256 \
  --output item_class_assignments_merged.xlsx
"""


def load_pickle(path):
    """Load a pickle (.pkl) file safely."""
    with open(path, "rb") as f:
        return pickle.load(f)


def fuzzy_match_score(a: str, b: str) -> float:
    """Return fuzzy string similarity score (0–1)."""
    a, b = a.lower(), b.lower()
    return max(
        fuzz.partial_ratio(a, b),
        fuzz.token_set_ratio(a, b)
    ) / 100.0


def assign_classes(
    items_df,
    classes_df,
    item_embeddings,
    class_embeddings,
    item_id_col,
    item_name_col,
    class_id_col,
    class_name_col,
    lowest_level_col=None,
    fuzzy_threshold=0.6,
    batch_size=256
):
    """Assign each item to the top-3 most similar classes using fuzzy + cosine similarity."""
    results = []

    # --- Filter for lowest-level classes if applicable ---
    if lowest_level_col and lowest_level_col in classes_df.columns:
        classes_df = classes_df[classes_df[lowest_level_col] == True]
        print(f"Using only lowest-level classes: {len(classes_df)} entries")
    else:
        print(f"No lowest-level column specified. Using all {len(classes_df)} classes.")

    # --- Prepare class vectors ---
    class_ids, class_names, class_vectors = [], [], []
    for _, row in classes_df.iterrows():
        cid = row[class_id_col]
        if cid in class_embeddings:
            class_ids.append(cid)
            class_names.append(str(row[class_name_col]))
            class_vectors.append(class_embeddings[cid])
    class_vectors = np.array(class_vectors)
    print(f"Loaded {len(class_vectors)} class embeddings for comparison.\n")

    # --- Group items by fuzzy match results ---
    fuzzy_groups, nonfuzzy_items = [], []
    print("🔍 Grouping items based on fuzzy match results...")
    for item_id, item_vec in tqdm(item_embeddings.items(), desc="Fuzzy grouping", unit="item"):
        item_row = items_df.loc[items_df[item_id_col] == item_id]
        if item_row.empty:
            continue
        item_name = str(item_row[item_name_col].iloc[0])
        candidate_indices = [
            i for i, cname in enumerate(class_names)
            if fuzzy_match_score(item_name, cname) >= fuzzy_threshold
        ]
        if candidate_indices:
            fuzzy_groups.append((item_id, item_name, item_vec, candidate_indices))
        else:
            nonfuzzy_items.append((item_id, item_name, item_vec))

    print(f"\nGrouping complete!")
    print(f"  - Items with fuzzy matches: {len(fuzzy_groups)}")
    print(f"  - Items with no fuzzy matches: {len(nonfuzzy_items)}\n")

    # --- Helper: append top-3 matches to result list ---
    def append_result(item_id, item_name, sims_sorted, candidate_indices, candidate_names):
        """Store top-3 results for an item."""
        top_ids, top_names, top_sims = [], [], []
        for rank in range(3):
            if rank < len(sims_sorted):
                idx, sim = sims_sorted[rank]
                top_ids.append(class_ids[idx])
                top_names.append(class_names[idx])
                top_sims.append(round(float(sim), 4))
            else:
                top_ids.append(None)
                top_names.append(None)
                top_sims.append(None)

        results.append({
            item_id_col: item_id,
            item_name_col: item_name,
            "top_1_class_id": top_ids[0],
            "top_1_class_name": top_names[0],
            "top_1_similarity": top_sims[0],
            "top_2_class_id": top_ids[1],
            "top_2_class_name": top_names[1],
            "top_2_similarity": top_sims[1],
            "top_3_class_id": top_ids[2],
            "top_3_class_name": top_names[2],
            "top_3_similarity": top_sims[2],
            "fuzzy_candidates_count": len(candidate_indices),
            "fuzzy_candidate_names": "; ".join(candidate_names)
        })

    # --- Compute similarities ---
    pbar = tqdm(total=len(item_embeddings), desc="Computing similarities", unit="item")

    # --- Fuzzy groups ---
    for (item_id, item_name, item_vec, candidate_indices) in fuzzy_groups:
        item_vec = item_vec.reshape(1, -1)
        sims = []
        for i in range(0, len(candidate_indices), batch_size):
            batch_indices = candidate_indices[i:i + batch_size]
            batch = class_vectors[batch_indices]
            batch_sims = cosine_similarity(batch, item_vec).flatten()
            sims.extend(zip(batch_indices, batch_sims))
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        candidate_names = [class_names[i] for i in candidate_indices[:20]]
        if len(candidate_indices) > 20:
            candidate_names.append(f"...(+{len(candidate_indices)-20} more)")
        append_result(item_id, item_name, sims_sorted, candidate_indices, candidate_names)
        pbar.update(1)

    # --- Non-fuzzy items (compare with all classes) ---
    for (item_id, item_name, item_vec) in nonfuzzy_items:
        item_vec = item_vec.reshape(1, -1)
        sims = []
        for i in range(0, len(class_vectors), batch_size):
            batch = class_vectors[i:i + batch_size]
            batch_sims = cosine_similarity(batch, item_vec).flatten()
            sims.extend(zip(range(i, i + len(batch_sims)), batch_sims))
        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        append_result(item_id, item_name, sims_sorted, [], ["No fuzzy match"])
        pbar.update(1)

    pbar.close()
    print("\nMatching completed!\n")
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Generic item-to-class matching using fuzzy + embedding similarity (Top-3)."
    )

    # Input paths
    parser.add_argument("--items_embeddings", required=True)
    parser.add_argument("--classes_embeddings", required=True)
    parser.add_argument("--items_excel", required=True)
    parser.add_argument("--classes_excel", required=True)

    # Column names
    parser.add_argument("--item_id_col", required=True)
    parser.add_argument("--item_name_col", required=True)
    parser.add_argument("--class_id_col", required=True)
    parser.add_argument("--class_name_col", required=True)
    parser.add_argument("--lowest_level_col", default=None)

    # Parameters
    parser.add_argument("--fuzzy_threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", "-o", default="item_class_assignments_merged.xlsx")

    args = parser.parse_args()

    print("\nLoading data ...")
    items_df = pd.read_excel(args.items_excel)
    classes_df = pd.read_excel(args.classes_excel)
    item_embeddings = load_pickle(args.items_embeddings)
    class_embeddings = load_pickle(args.classes_embeddings)

    # Normalize column names
    items_df.columns = items_df.columns.str.strip().str.lower()
    classes_df.columns = classes_df.columns.str.strip().str.lower()

    args.item_id_col = args.item_id_col.strip().lower()
    args.item_name_col = args.item_name_col.strip().lower()
    args.class_id_col = args.class_id_col.strip().lower()
    args.class_name_col = args.class_name_col.strip().lower()
    if args.lowest_level_col:
        args.lowest_level_col = args.lowest_level_col.strip().lower()

    print("Normalized column names for robust matching.")

    # Run assignment
    results_df = assign_classes(
        items_df,
        classes_df,
        item_embeddings,
        class_embeddings,
        item_id_col=args.item_id_col,
        item_name_col=args.item_name_col,
        class_id_col=args.class_id_col,
        class_name_col=args.class_name_col,
        lowest_level_col=args.lowest_level_col,
        fuzzy_threshold=args.fuzzy_threshold,
        batch_size=args.batch_size
    )

    # Merge results back dynamically
    print("\nMerging results with original Excel data ...")
    merged = results_df.merge(items_df, on=args.item_id_col, how="left", suffixes=("", "_item"))

    class_merge = classes_df.add_prefix("class_")
    merged = merged.merge(
        class_merge,
        left_on="top_1_class_id",
        right_on=f"class_{args.class_id_col}",
        how="left"
    )

    merged.to_excel(args.output, index=False)
    print(f"\nDone! Saved merged file with {len(merged)} rows to '{args.output}'")


if __name__ == "__main__":
    main()

