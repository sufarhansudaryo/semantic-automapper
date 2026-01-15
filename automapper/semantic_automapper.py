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

- Fuzzy search is performed across ALL classes (all hierarchy levels).
- For each fuzzy match, we recursively expand down the hierarchy to collect
  ALL lowest-level descendants (lowest_level_class == True).
- Cosine similarity is computed only against the collected lowest-level children.
- Items with no fuzzy match still compare against ALL lowest-level classes.
- fuzzy_candidate_names now include the hierarchy level, e.g. "name (L3)".
"""


# =============================================================================
# Utilities
# =============================================================================
def load_pickle(path: str):
    """Load a pickle (.pkl) file safely."""
    with open(path, "rb") as f:
        return pickle.load(f)


def fuzzy_match_score(a: str, b: str) -> float:
    """Return fuzzy string similarity score (0–1)."""
    a, b = a.lower(), b.lower()
    return max(
        fuzz.partial_ratio(a, b),
        fuzz.token_set_ratio(a, b),
    ) / 100.0


# =============================================================================
# Core logic
# =============================================================================
def assign_classes(
    items_df,
    classes_df,
    item_embeddings,
    class_embeddings,
    item_id_col,
    item_name_col,
    class_id_col,
    class_name_col,
    lowest_level_col="lowest_level_class",
    fuzzy_threshold=0.6,
    batch_size=256,
):
    results = []

    # Keep a full copy for fuzzy search + tree traversal
    classes_df_all = classes_df.copy()

    # Prepare only lowest-level classes for cosine similarity
    if lowest_level_col in classes_df.columns:
        classes_df_lowest = classes_df[classes_df[lowest_level_col] == True]
        print(
            "Using only lowest-level classes for cosine similarity: "
            f"{len(classes_df_lowest)} entries"
        )
    else:
        classes_df_lowest = classes_df
        print("No lowest-level column found. Using all classes for similarity.")

    # Prepare full class lists (ALL classes → fuzzy)
    all_class_ids = classes_df_all[class_id_col].tolist()
    all_class_names = classes_df_all[class_name_col].astype(str).tolist()

    # Levels for display in fuzzy_candidate_names
    if "level" in classes_df_all.columns:
        all_class_levels = classes_df_all["level"].tolist()
    else:
        all_class_levels = [None] * len(all_class_ids)

    # Prepare cosine list (lowest-level only)
    class_ids, class_names, class_vectors = [], [], []
    for _, row in classes_df_lowest.iterrows():
        cid = row[class_id_col]
        if cid in class_embeddings:
            class_ids.append(cid)
            class_names.append(str(row[class_name_col]))
            class_vectors.append(class_embeddings[cid])
    class_vectors = np.array(class_vectors)

    print(f"Loaded {len(class_vectors)} lowest-level class embeddings.\n")

    # ==============================
    # Precompute children map
    # ==============================
    children_map = {}
    for _, row in classes_df_all.iterrows():
        pid = row["parent_id"]
        cid = row[class_id_col]
        children_map.setdefault(pid, []).append(cid)

    # ==============================
    # Recursive collection of lowest-level descendants
    # ==============================
    def get_lowest_descendants(class_id):
        row = classes_df_all.loc[classes_df_all[class_id_col] == class_id]

        # If this is already lowest level → return it
        if not row.empty and row[lowest_level_col].iloc[0] == True:
            return [class_id]

        # If this class has no children → nothing to expand
        if class_id not in children_map:
            return []

        # Recurse into children
        result = []
        for child in children_map[class_id]:
            result.extend(get_lowest_descendants(child))

        return result

    # ==============================
    # Group items into fuzzy-hit group or non-fuzzy group
    # ==============================
    fuzzy_groups = []
    nonfuzzy_items = []

    print("🔍 Grouping items based on fuzzy match results...")
    for item_id, item_vec in tqdm(
        item_embeddings.items(), desc="Fuzzy grouping", unit="item"
    ):
        row = items_df.loc[items_df[item_id_col] == item_id]
        if row.empty:
            continue

        item_name = str(row[item_name_col].iloc[0])

        # Fuzzy search across ALL classes
        fuzzy_hits = [
            i
            for i, cname in enumerate(all_class_names)
            if fuzzy_match_score(item_name, cname) >= fuzzy_threshold
        ]

        if fuzzy_hits:
            # Expand matched class → lowest-level children recursively
            candidate_indices = []

            for idx in fuzzy_hits:
                matched_class_id = all_class_ids[idx]
                lowest_descendants = get_lowest_descendants(matched_class_id)

                # Map descendants to indices for cosine similarity
                for cid in lowest_descendants:
                    if cid in class_ids:
                        candidate_indices.append(class_ids.index(cid))

            candidate_indices = list(set(candidate_indices))

            # Build fuzzy candidate display names with levels: "name (L3)"
            candidate_names = []
            for i in fuzzy_hits[:20]:
                level = all_class_levels[i]
                if level is not None and level != "":
                    candidate_names.append(f"{all_class_names[i]} (L{level})")
                else:
                    candidate_names.append(all_class_names[i])

            if len(fuzzy_hits) > 20:
                candidate_names.append(f"...(+{len(fuzzy_hits) - 20} more)")

            fuzzy_groups.append(
                (item_id, item_name, item_vec, candidate_indices, candidate_names)
            )

        else:
            nonfuzzy_items.append((item_id, item_name, item_vec))

    print("\nGrouping complete!")
    print(f"  - Items with fuzzy matches: {len(fuzzy_groups)}")
    print(f"  - Items with no fuzzy matches: {len(nonfuzzy_items)}\n")

    # ==============================
    # Helper to store results
    # ==============================
    def append_result(item_id, item_name, sims_sorted, candidate_indices, candidate_names):
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

        results.append(
            {
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
                "fuzzy_candidate_names": "; ".join(candidate_names),
            }
        )

    # ==============================
    # Cosine similarity evaluation
    # ==============================
    pbar = tqdm(
        total=len(item_embeddings),
        desc="Computing similarities",
        unit="item",
    )

    # Fuzzy group → compare against expanded candidates
    for item_id, item_name, item_vec, candidate_indices, candidate_names in fuzzy_groups:
        item_vec = item_vec.reshape(1, -1)
        sims = []

        for i in range(0, len(candidate_indices), batch_size):
            batch_idx = candidate_indices[i : i + batch_size]
            batch = class_vectors[batch_idx]
            sim = cosine_similarity(batch, item_vec).flatten()
            sims.extend(zip(batch_idx, sim))

        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        append_result(
            item_id,
            item_name,
            sims_sorted,
            candidate_indices,
            candidate_names,
        )
        pbar.update(1)

    # Non-fuzzy → compare against ALL lowest-level classes
    for item_id, item_name, item_vec in nonfuzzy_items:
        item_vec = item_vec.reshape(1, -1)
        sims = []

        for i in range(0, len(class_vectors), batch_size):
            batch = class_vectors[i : i + len(class_vectors[i:i + batch_size])]
            sim = cosine_similarity(batch, item_vec).flatten()
            sims.extend(zip(range(i, i + len(sim)), sim))

        sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[:3]
        append_result(
            item_id,
            item_name,
            sims_sorted,
            [],
            ["No fuzzy match"],
        )
        pbar.update(1)

    pbar.close()
    print("\nMatching completed!\n")

    return pd.DataFrame(results)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generic item-to-class matching using fuzzy + "
            "embedding similarity (Top-3)."
        )
    )

    parser.add_argument("--items_embeddings", required=True)
    parser.add_argument("--classes_embeddings", required=True)
    parser.add_argument("--items_excel", required=True)
    parser.add_argument("--classes_excel", required=True)

    parser.add_argument("--item_id_col", required=True)
    parser.add_argument("--item_name_col", required=True)
    parser.add_argument("--class_id_col", required=True)
    parser.add_argument("--class_name_col", required=True)
    parser.add_argument("--lowest_level_col", default="lowest_level_class")

    parser.add_argument("--fuzzy_threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--output",
        "-o",
        default="item_class_assignments_merged.xlsx",
    )

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
        batch_size=args.batch_size,
    )

    # Merge results with items
    print("\nMerging results...")
    merged = results_df.merge(items_df, on=args.item_id_col, how="left")

    # Merge class information of top-1 match
    class_merge = classes_df.add_prefix("class_")
    merged = merged.merge(
        class_merge,
        left_on="top_1_class_id",
        right_on=f"class_{args.class_id_col}",
        how="left",
    )

    merged.to_excel(args.output, index=False)
    print(
        f"\nDone! Saved merged file with {len(merged)} rows to '{args.output}'"
    )


if __name__ == "__main__":
    main()
