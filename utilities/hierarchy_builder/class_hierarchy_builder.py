import pandas as pd
from typing import Optional, List


def mark_leaf_nodes(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'lowest_level_class' column indicating whether each node is a leaf.
    """
    all_ids = set(nodes_df["id"])
    parent_ids = set(nodes_df["parent_id"].dropna())
    leaf_ids = all_ids - parent_ids


    nodes_df["lowest_level_class"] = nodes_df["id"].apply(
    lambda x: True if x in leaf_ids else False
    )
    return nodes_df


def extract_node_table(
    input_path: str,
    sheet_name: str,
    header_row_index: int,
    level_column_keywords: Optional[List[str]] = None,
    level_column_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extracts a product class hierarchy from an Excel file and returns a node table.

    Returns:
        pd.DataFrame with columns:
        id, name, level, level_label, parent_id, lowest_level_class
    """
    if level_column_keywords is None:
        level_column_keywords = ["Level"]

    df_raw = pd.read_excel(input_path, sheet_name=sheet_name, header=None)

    # Set headers from the specified row
    df = df_raw.iloc[header_row_index + 1 :].copy()
    df.columns = df_raw.iloc[header_row_index]

    # Pick the columns that represent levels
    if level_column_names:
        level_cols = level_column_names
    else:
        level_cols = [
            col
            for col in df.columns
            if isinstance(col, str)
            and any(kw in col for kw in level_column_keywords)
        ]

    # Forward fill to handle empty parent cells
    df[level_cols] = df[level_cols].ffill()
    df = df[~df[level_cols].isna().all(axis=1)]

    # Build node table
    nodes = []
    node_index = {}
    next_id = 1

    for _, row in df[level_cols].iterrows():
        parent_key = None
        for level_pos, col in enumerate(level_cols, start=1):
            val = row[col]
            if pd.isna(val):
                continue
            name = str(val).strip()
            if not name:
                continue

            key = (level_pos, name)

            if key not in node_index:
                node_index[key] = next_id
                parent_id = node_index.get(parent_key) if parent_key else None

                nodes.append(
                    {
                        "id": next_id,
                        "name": name,
                        "level": level_pos,
                        "level_label": col,
                        "parent_id": parent_id,
                    }
                )
                next_id += 1

            parent_key = key

    nodes_df = pd.DataFrame(nodes)
    return mark_leaf_nodes(nodes_df)
