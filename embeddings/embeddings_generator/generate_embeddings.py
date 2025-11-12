import argparse
import os
import pickle
from tqdm import tqdm
from get_embeddings import generate_embeddings

"""
Generate Embeddings from Excel Data
------------------------------------------

This script serves as a command-line interface (CLI) for generating text embeddings
from an Excel file using the Gemini API via Google Vertex AI or Google AI API. Users
can specify the Excel file, sheet name, and the column containing text data for which
embeddings are to be generated.

Usage Example:
--------------
python generate_embeddings.py \
    "/path/to/data.xlsx" \
    "Sheet1" \
    "text_column_name" \
    --id "unique_id_column" \
    --workers 10 \
    --output "output_embeddings.pkl"

Requirements:
-------------
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
- Environment variables set in `.env.local` in project root:
    GOOGLE_API_KEY=...  or
    GOOGLE_CLOUD_PROJECT=...
    GOOGLE_CLOUD_LOCATION=...
    GOOGLE_GENAI_USE_VERTEXAI=True

Output:
-------
A `.pkl` file containing embeddings for each row, stored as a dictionary.

Example output structure:
{
    101: np.array([...]),
    102: np.array([...]),
    ...
}
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Gemini embeddings for Excel data"
    )
    parser.add_argument(
        "file", 
        help="Path to Excel file",
    )
    parser.add_argument(
        "sheet", 
        help="Sheet name in Excel file",
    )
    parser.add_argument(
        "text_column", 
        help="Column name containing text",
    )
    parser.add_argument(
        "--id", 
        default=None, 
        help="Column name for unique IDs (optional)",
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=10, 
        help="Number of parallel workers (default 10)",
    )
    parser.add_argument(
        "--output", 
        "-o", 
        default=None, 
        help="Output .pkl file (default: <input>_embeddings.pkl)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Parsed CLI args:", args)
    embeddings = generate_embeddings(
        args.file,
        args.sheet,
        args.text_column,
        id_column=args.id,
        max_workers=args.workers,
    )

    if embeddings:
        output_path = args.output or f"{os.path.splitext(os.path.basename(args.file))[0]}_embeddings.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)
        tqdm.write(f"Saved {len(embeddings)} embeddings to {output_path}")
    else:
        tqdm.write("No embeddings generated.")


if __name__ == "__main__":
    main()
