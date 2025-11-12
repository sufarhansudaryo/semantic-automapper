import argparse
from class_hierarchy_builder import extract_node_table

"""
Generate a Class Hierarchy Node Table from Excel
------------------------------------------------

This script provides a command-line interface (CLI) for converting a hierarchical
Excel sheet (e.g. product hierarchy) into a normalized *node table* with
unique IDs and parent–child relationships.

It wraps the `extract_node_table` function from `class_hierarchy_builder` and
allows you to configure:

- The input Excel file and sheet
- Which row contains the column headers
- How to detect hierarchy level columns (by keyword or explicit names)
- The output Excel file for the generated node table

The resulting node table includes columns such as:
- `id`                : unique integer ID for each class/node
- `name`              : class name at that level
- `level`             : numeric hierarchy depth (1 = top level)
- `level_label`       : original column name from the Excel header (e.g. 'Level 2 - Product Groups')
- `parent_id`         : ID of the parent node (or NaN/None for root nodes)
- `lowest_level_class`: boolean flag indicating whether the node is a leaf (no children)

Usage Example
-------------
With custom level detection by keyword:
python generate_class_hierarchy.py \
    --input "/path/to/file.xlsx" \
    --sheet "Hierarchy" \
    --header-row 5 \
    --output "node_table.xlsx" \
    --level-column-keywords Level

Requirements
------------
- A valid Excel file where:
  - One row contains the hierarchy column headers (e.g. 'Level 1 - Activities', ...)
  - Subsequent rows contain the hierarchical data
  - Empty parent cells are allowed; they will be forward-filled by the script
"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a class hierarchy node table from an Excel sheet."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the Excel file containing the hierarchy.",
    )
    parser.add_argument(
        "--sheet",
        required=True,
        help="Name of the sheet that contains the hierarchy (e.g. 'tree' or 'Source').",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        required=True,
        help="Row index (0-based) where the column headers start.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output Excel file for the node table.",
    )
    # Optional: level column detection by keyword(s)
    parser.add_argument(
        "--level-column-keywords",
        nargs="+",
        default=["Level"],
        help=(
            "One or more keywords used to detect hierarchy columns by name. "
            "Example: --level-column-keywords Level Tier"
        ),
    )
    # Optional: explicit level column names
    parser.add_argument(
        "--level-column-names",
        nargs="+",
        help=(
            "Explicit list of column names to treat as hierarchy levels. "
            "If provided, this overrides --level-column-keywords. "
            "Example: --level-column-names 'Level 1 - Activities' "
            "'Level 2 - Product Groups'"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Parsed CLI args:", args)

    level_column_keywords = args.level_column_keywords
    level_column_names = args.level_column_names

    df_nodes = extract_node_table(
        input_path=args.input,
        sheet_name=args.sheet,
        header_row_index=args.header_row,
        level_column_keywords=level_column_keywords,
        level_column_names=level_column_names,
    )

    df_nodes.to_excel(args.output, index=False)
    print(f"Node table saved to: {args.output}")
    print(df_nodes.head())


if __name__ == "__main__":
    main()
