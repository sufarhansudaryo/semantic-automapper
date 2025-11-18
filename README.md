# semantic-automapper

A small toolkit to generate text embeddings from Excel data and automatically map items to a class hierarchy using a hybrid fuzzy + embeddings approach.

This repository focuses on two main features:
- Embeddings generation for Excel data (threaded + retry) — see [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py) and [`get_embeddings.generate_embeddings`](embeddings/embeddings_generator/get_embeddings.py).
- Automated item → class assignment (Top-3 candidates using fuzzy + cosine similarity) — see [automapper/semantic_automapper.py](automapper/semantic_automapper.py) and the core function [`semantic_automapper.assign_classes`](automapper/semantic_automapper.py).

Project layout
- [.env.local](.env.local) — local environment variables (project, location, credentials flags)
- [.gitignore](.gitignore)
- [requirements.txt](requirements.txt)
- [automapper/semantic_automapper.py](automapper/semantic_automapper.py) — CLI and algorithm (`assign_classes`)
- [automapper/validator.py](automapper/validator.py) — Excel color-coding and summary for Top-1/2/3
- [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py) — preview / convert .pkl embeddings to JSON
- embeddings/embeddings_generator/
  - [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py) — CLI wrapper
  - [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py) — API client + threading (`get_embedding`, `generate_embeddings`)
- [utilities/excel_extractor.py](utilities/excel_extractor.py) — small Tk GUI to extract unique rows and assign IDs
- utilities/hierarchy_builder/
  - [utilities/hierarchy_builder/generate_class_hierarchy.py](utilities/hierarchy_builder/generate_class_hierarchy.py) — CLI wrapper
  - [utilities/hierarchy_builder/class_hierarchy_builder.py](utilities/hierarchy_builder/class_hierarchy_builder.py) — core builder (`class_hierarchy_builder.extract_node_table`)
- Data & outputs:
  - [data/](data/) (excel_files, generated_embeddings)
  - [results/](results/)
- [credentials/](credentials/) — service account JSONs (ignored by .gitignore)

Core components (quick overview)

- Embeddings
  - Use [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py) to read an Excel sheet and produce a `.pkl` mapping of IDs → NumPy embeddings.
  - The implementation uses [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py) which handles:
    - API calls to Google GenAI / Vertex AI via `google-genai`
    - Exponential backoff (`get_embedding`)
    - ThreadPool parallelism (`generate_embeddings`)
  - Inspect or convert `.pkl` files with [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py).

- Automapper (item → class mapping)
  - Algorithm lives in [automapper/semantic_automapper.py](automapper/semantic_automapper.py). The key function [`semantic_automapper.assign_classes`](automapper/semantic_automapper.py) implements:
    - Fuzzy matching (RapidFuzz) to shortlist candidate classes for each item
    - Cosine similarity (scikit-learn) between item and class embeddings to compute Top-3 matches
    - Optional filtering to lowest-level classes in the hierarchy
    - Batched similarity computation and output merging with original Excel
  - Validate and color-code results using [automapper/validator.py](automapper/validator.py).

- Hierarchy utilities
  - Create a normalized node table from a hierarchical Excel sheet via [utilities/hierarchy_builder/generate_class_hierarchy.py](utilities/hierarchy_builder/generate_class_hierarchy.py), which calls [`class_hierarchy_builder.extract_node_table`](utilities/hierarchy_builder/class_hierarchy_builder.py).
  - The node table includes unique `id`, `name`, `level`, `parent_id`, and `lowest_level_class` flags used by the automapper.

Quickstart

1. Install dependencies:
```sh
pip install -r requirements.txt
```

2. Configure authentication and settings:
- Create / edit `.env.local` in the project root (this repo loads it in [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py)):
  - GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
  - GOOGLE_CLOUD_PROJECT=your-project-id
  - GOOGLE_CLOUD_LOCATION=your-location
  - GOOGLE_GENAI_USE_VERTEXAI=True/False

Note: Credentials files under [credentials/](credentials/) are ignored by [.gitignore](.gitignore). Do not commit secrets.

3. Generate embeddings from Excel:
```sh
python embeddings/embeddings_generator/generate_embeddings.py \
  "/path/to/data.xlsx" "Sheet1" "text_column_name" \
  --id "id_column" --workers 8 --output data/generated_embeddings/data_embeddings.pkl
```
- CLI calls [`get_embeddings.generate_embeddings`](embeddings/embeddings_generator/get_embeddings.py) which concurrently fetches embeddings using [`get_embeddings.get_embedding`](embeddings/embeddings_generator/get_embeddings.py).
- Tune `--workers` and `max_workers` in [get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py) for throughput vs. quota.

4. Build class node table (if you have a hierarchical sheet):
```sh
python utilities/hierarchy_builder/generate_class_hierarchy.py \
  --input "/path/to/hierarchy.xlsx" --sheet "Hierarchy" --header-row 4 \
  --output node_table.xlsx --level-column-keywords Level
```
- This script uses [`class_hierarchy_builder.extract_node_table`](utilities/hierarchy_builder/class_hierarchy_builder.py).

5. Run the automapper:
```sh
python automapper/semantic_automapper.py \
  --items_embeddings data/generated_embeddings/items_embeddings.pkl \
  --classes_embeddings data/generated_embeddings/classes_embeddings.pkl \
  --items_excel data/excel_files/items.xlsx \
  --classes_excel data/excel_files/classes.xlsx \
  --item_id_col ID --item_name_col Description \
  --class_id_col ID --class_name_col name \
  --lowest_level_col lowest_level_class \
  --fuzzy_threshold 0.65 --batch_size 256 \
  --output results/item_class_assignments_merged.xlsx
```
- Core function: [`semantic_automapper.assign_classes`](automapper/semantic_automapper.py)
- Tweak `--fuzzy_threshold` and `--batch_size` for performance/precision trade-offs.

6. Review results and validate:
- Use [automapper/validator.py](automapper/validator.py) to color-code Top-1/2/3 matches and create a summary sheet.
- Use [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py) to preview and convert embeddings `.pkl` files.

Tips and notes
- Column normalization: the automapper lowercases and strips column names before matching — ensure CLI column names match the Excel headers (or run the extractor UI).
- Empty / missing text rows are skipped by the embedding generator.
- The hierarchy builder forward-fills parent cells before generating unique node IDs and marks leaves via `lowest_level_class`.
- Keep credentials out of the repo; use `.env.local` and a secure service account key.

Useful links (files & symbols)
- [automapper/semantic_automapper.py](automapper/semantic_automapper.py) — main automapper CLI
- [`semantic_automapper.assign_classes`](automapper/semantic_automapper.py)
- [automapper/validator.py](automapper/validator.py)
- [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py)
- [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py)
- [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py)
- [`get_embeddings.get_embedding`](embeddings/embeddings_generator/get_embeddings.py)
- [`get_embeddings.generate_embeddings`](embeddings/embeddings_generator/get_embeddings.py)
- [utilities/excel_extractor.py](utilities/excel_extractor.py)
- [utilities/hierarchy_builder/generate_class_hierarchy.py](utilities/hierarchy_builder/generate_class_hierarchy.py)
- [utilities/hierarchy_builder/class_hierarchy_builder.py](utilities/hierarchy_builder/class_hierarchy_builder.py)
- [`class_hierarchy_builder.extract_node_table`](utilities/hierarchy_builder/class_hierarchy_builder.py)
- [requirements.txt](requirements.txt)
- [.gitignore](.gitignore)
- [.env.local](.env.local)
- [data/](data/)
- [results/](results/)