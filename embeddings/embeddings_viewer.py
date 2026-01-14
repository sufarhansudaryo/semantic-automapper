import pickle
import json
import argparse
from pathlib import Path

"""
View and Convert Pickle Files to JSON
-------------------------------------

This script allows you to inspect the contents of a Python pickle (.pkl) file 
and optionally convert it into a human-readable JSON format.

Features:
---------
- Loads and previews data stored in pickle files (e.g., dictionaries or lists).
- Displays basic information such as data type, item count, and shapes of arrays.
- Converts NumPy arrays and other non-serializable objects into JSON-friendly types.
- Optionally saves the converted data to a JSON file.

Usage Example:
--------------
python embedding_viewer.py path/to/file.pkl --preview 5 --output output.json
"""


def load_pkl(file_path):
    """Load pickle file and return the data."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def convert_to_json_serializable(obj):
    """Convert numpy arrays and other objects to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def main():
    parser = argparse.ArgumentParser(description="View pickle file content")
    parser.add_argument("file", help="Path to .pkl file")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file (optional)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=3,
        help="Number of items to preview (default: 3)",
    )

    args = parser.parse_args()

    # Load pickle file
    data = load_pkl(args.file)
    if data is None:
        return

    print(f"Loaded data type: {type(data)}")
    print(f"Total items: {len(data) if isinstance(data, (dict, list)) else 'N/A'}")
    print("\n" + "="*50)

    # Preview data
    if isinstance(data, dict):
        print(f"\nPreview (first {args.preview} items):")
        for i, (key, value) in enumerate(data.items()):
            if i >= args.preview:
                break
            if hasattr(value, 'shape'):  # numpy array
                print(f"  {key}: array with shape {value.shape}")
            else:
                print(f"  {key}: {type(value).__name__}")
    elif isinstance(data, list):
        print(f"\nPreview (first {args.preview} items):")
        for i, item in enumerate(data[:args.preview]):
            print(f"  [{i}]: {type(item).__name__}")

    # Convert to JSON if output specified
    if args.output:
        try:
            json_data = convert_to_json_serializable(data)
            with open(args.output, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"\nConverted to JSON and saved to: {args.output}")
        except Exception as e:
            print(f"Error converting to JSON: {e}")

if __name__ == "__main__":
    
    main()