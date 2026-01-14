from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

# What we want to transfer from SOURCE into TARGET
TRANSFER_COLS = ["Type Code", "Type Description", "Coating", "Description_Expanded"]

# Internal join columns (will be removed)
SRC_KEY = "__SRC_KEY__"
TGT_KEY = "__TGT_KEY__"


def _normalize_colname(c: str) -> str:
    # Only normalize column names (NOT values)
    return str(c).strip().replace("\u00A0", " ")


def _make_join_key(series: pd.Series) -> pd.Series:
    """
    Create a join key for matching WITHOUT modifying the original column in the sheet.
    We do not strip/lowercase the values; we only convert to pandas string dtype.
    """
    return series.where(series.notna(), other=pd.NA).astype("string")


def build_source_lookup(source_path: Path, source_id_col: str) -> pd.DataFrame:
    """
    Read all sheets from source workbook, collect Id + TRANSFER_COLS (where present),
    and build a lookup keyed by SRC_KEY.
    """
    xls = pd.ExcelFile(source_path)
    parts: List[pd.DataFrame] = []

    for sheet in xls.sheet_names:
        df = pd.read_excel(source_path, sheet_name=sheet)
        df.columns = [_normalize_colname(c) for c in df.columns]

        if source_id_col not in df.columns:
            continue

        chunk = pd.DataFrame()
        chunk[SRC_KEY] = _make_join_key(df[source_id_col])

        for c in TRANSFER_COLS:
            chunk[c] = df[c] if c in df.columns else pd.NA

        # keep only rows with a key
        chunk = chunk[chunk[SRC_KEY].notna() & (chunk[SRC_KEY] != "")]
        parts.append(chunk)

    if not parts:
        raise ValueError(
            f"No usable sheets found in source workbook with column '{source_id_col}'."
        )

    all_rows = pd.concat(parts, ignore_index=True)

    # Deduplicate: for each Id, take first non-empty value per column
    def first_non_empty(series: pd.Series):
        s = series.dropna()
        s = s[s.astype(str) != ""]
        return s.iloc[0] if len(s) else pd.NA

    lookup = (
        all_rows.groupby(SRC_KEY, as_index=False)
        .agg({c: first_non_empty for c in TRANSFER_COLS})
    )

    return lookup


def merge_into_target(
    target_path: Path,
    lookup: pd.DataFrame,
    output_path: Path,
    target_map_col: str,
    overwrite: bool = False,
) -> None:
    """
    For each sheet in the target workbook:
    - if it contains target_map_col (e.g., 'No.'), enrich it by mapping:
        target[target_map_col]  <->  source[Id]
    - add missing TRANSFER_COLS to target
    - fill values from source where available
    - NEVER modify existing target entries (other than adding/filling the new cols)
    """
    xls = pd.ExcelFile(target_path)
    out_sheets: Dict[str, pd.DataFrame] = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(target_path, sheet_name=sheet)
        df.columns = [_normalize_colname(c) for c in df.columns]

        # If this sheet doesn't have the mapping column, keep unchanged
        if target_map_col not in df.columns:
            out_sheets[sheet] = df
            continue

        df = df.copy()

        # internal join key based on target mapping column (No.)
        df[TGT_KEY] = _make_join_key(df[target_map_col])

        # ensure new columns exist
        for c in TRANSFER_COLS:
            if c not in df.columns:
                df[c] = pd.NA

        merged = df.merge(
            lookup,
            left_on=TGT_KEY,
            right_on=SRC_KEY,
            how="left",
            suffixes=("", "__src"),
        )

        # Fill per column
        for c in TRANSFER_COLS:
            src_c = f"{c}__src"
            if src_c not in merged.columns:
                continue

            if overwrite:
                # overwrite target only where source has value
                merged[c] = merged[src_c].combine_first(merged[c])
            else:
                # fill missing target only
                merged[c] = merged[c].combine_first(merged[src_c])

            merged.drop(columns=[src_c], inplace=True)

        # clean internal cols
        merged.drop(columns=[TGT_KEY, SRC_KEY], inplace=True)

        out_sheets[sheet] = merged

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, sdf in out_sheets.items():
            sdf.to_excel(writer, sheet_name=sheet, index=False)


def main():
    p = argparse.ArgumentParser(
        description="Enrich a target Excel by mapping Target[No.] -> Source[Id] across all source sheets."
    )
    p.add_argument("--source", required=True, help="Source workbook (multi-sheet), e.g. items_expanded.xlsx")
    p.add_argument("--target", required=True, help="Target workbook to enrich")
    p.add_argument("--source-id-col", default="Id", help="Column name in SOURCE used as key (default: Id)")
    p.add_argument("--target-map-col", default="No.", help="Column name in TARGET used for mapping (default: No.)")
    p.add_argument("--output", default=None, help="Output path (default: <target>_merged.xlsx)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite target values when source has values")
    args = p.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    output = Path(args.output) if args.output else target.with_name(f"{target.stem}_merged{target.suffix}")

    lookup = build_source_lookup(source, source_id_col=args.source_id_col)
    merge_into_target(
        target_path=target,
        lookup=lookup,
        output_path=output,
        target_map_col=args.target_map_col,
        overwrite=args.overwrite,
    )

    print(f"Done. Output written to: {output}")


if __name__ == "__main__":
    main()
