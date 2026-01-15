# semantic-automapper

A toolkit for generating text embeddings from Excel data and automatically mapping items to a class hierarchy using a hybrid fuzzy, embedding-based, and optional LLM-assisted approach.

This repository focuses on two main features:
- **Embeddings generation for Excel data** (threaded + retry) — see [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py) and [`get_embeddings.generate_embeddings`](embeddings/embeddings_generator/get_embeddings.py).
- **Automated item-to-class assignment** (Top-3 candidates using fuzzy + cosine similarity, with optional LLM reranking) — see [automapper/automapper_llm.py](automapper/automapper_llm.py) and the core function [`automapper_llm.assign_classes`](automapper/automapper_llm.py).

---

## Project Layout

### Core Files
- **Embeddings**
  - [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py) — CLI for generating embeddings from Excel data.
  - [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py) — API client + threading (`get_embedding`, `generate_embeddings`).
  - [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py) — Preview and convert `.pkl` embeddings to JSON.

- **Automapper**
  - [automapper/automapper_llm.py](automapper/automapper_llm.py) — **Primary automapper script**. Combines fuzzy matching, cosine similarity, and optional LLM reranking for item-to-class assignment.
  - [automapper/validator.py](automapper/validator.py) — Excel color-coding and summary for Top-1/2/3 matches.

- **Hierarchy Utilities**
  - [utilities/hierarchy_builder/generate_class_hierarchy.py](utilities/hierarchy_builder/generate_class_hierarchy.py) — CLI for generating a normalized class hierarchy node table from Excel.
  - [utilities/hierarchy_builder/class_hierarchy_builder.py](utilities/hierarchy_builder/class_hierarchy_builder.py) — Core builder (`class_hierarchy_builder.extract_node_table`).

- **Other Utilities**
  - [utilities/excel_extractor.py](utilities/excel_extractor.py) — GUI tool for extracting unique rows from Excel files and assigning IDs.
  - [utilities/abbreviation/expand_abbreviation.py](utilities/abbreviation/expand_abbreviation.py) — LLM-based abbreviation expansion for Excel data.
  - [utilities/abbreviation/replace_abbreviation.py](utilities/abbreviation/replace_abbreviation.py) — Replace abbreviations in Excel sheets using a mapping file.

### Data & Outputs
- **Data**
  - [data/](data/) — Contains input Excel files (e.g., `excel_files/`) and generated embeddings.
- **Results**
  - [results/](results/) — Contains output files (e.g., merged item-class assignments).
- **Credentials**
  - [credentials/](credentials/) — Service account JSONs (ignored by `.gitignore`).

---

## Core Components

### Embeddings
- Use [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py) to read an Excel sheet and produce a `.pkl` mapping of `ID → NumPy embeddings`.
- The implementation uses [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py), which handles:
  - API calls to Google GenAI / Vertex AI via `google-genai`.
  - Exponential backoff (`get_embedding`).
  - ThreadPool parallelism (`generate_embeddings`).
- Inspect or convert `.pkl` files with [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py).

### Automapper (Item-to-Class Mapping)
- The **primary automapper script** is [automapper/automapper_llm.py](automapper/automapper_llm.py), which implements:
  - **Fuzzy matching** (RapidFuzz) to shortlist candidate classes for each item.
  - **Cosine similarity** (scikit-learn) between item and class embeddings to compute Top-3 matches.
  - **Optional LLM reranking** (Gemini) to refine the Top-K candidates and select the best match.
  - **Autosave and resume** functionality for long LLM runs.
- Validate and color-code results using [automapper/validator.py](automapper/validator.py).

### Hierarchy Utilities
- Create a normalized node table from a hierarchical Excel sheet via [utilities/hierarchy_builder/generate_class_hierarchy.py](utilities/hierarchy_builder/generate_class_hierarchy.py), which calls [`class_hierarchy_builder.extract_node_table`](utilities/hierarchy_builder/class_hierarchy_builder.py).
- The node table includes unique `id`, `name`, `level`, `parent_id`, and `lowest_level_class` flags used by the automapper.

---

## Quickstart

### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

### 2. Configure Authentication and Settings
- Create or edit `.env.local` in the project root (this repo loads it in [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py)):
  ```env
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
  GOOGLE_CLOUD_PROJECT=your-project-id
  GOOGLE_CLOUD_LOCATION=your-location
  GOOGLE_GENAI_USE_VERTEXAI=True
  ```

> **Note:** Credentials files under [credentials/](credentials/) are ignored by [.gitignore](.gitignore). Do not commit secrets.

---

### 3. Generate Embeddings from Excel
```sh
python embeddings/embeddings_generator/generate_embeddings.py \
  "/path/to/data.xlsx" "Sheet1" "text_column_name" \
  --id "id_column" --workers 8 --output data/generated_embeddings/data_embeddings.pkl
```
- CLI calls [`get_embeddings.generate_embeddings`](embeddings/embeddings_generator/get_embeddings.py), which concurrently fetches embeddings using [`get_embeddings.get_embedding`](embeddings/embeddings_generator/get_embeddings.py).
- Tune `--workers` and `max_workers` in [get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py) for throughput vs. quota.

---

### 4. Build Class Node Table (if you have a hierarchical sheet)
```sh
python utilities/hierarchy_builder/generate_class_hierarchy.py \
  --input "/path/to/hierarchy.xlsx" --sheet "Hierarchy" --header-row 4 \
  --output node_table.xlsx --level-column-keywords Level
```
- This script uses [`class_hierarchy_builder.extract_node_table`](utilities/hierarchy_builder/class_hierarchy_builder.py).

---

### 5. Run the Automapper
```sh
python automapper/automapper_llm.py \
  --items_embeddings data/generated_embeddings/items_embeddings.pkl \
  --classes_embeddings data/generated_embeddings/classes_embeddings.pkl \
  --items_excel data/excel_files/items.xlsx \
  --classes_excel data/excel_files/classes.xlsx \
  --item_id_col ID --item_name_col Description \
  --class_id_col ID --class_name_col name \
  --lowest_level_col lowest_level_class \
  --fuzzy_threshold 0.65 --batch_size 256 \
  --use_llm --llm_top_k 5 \
  --output results/item_class_assignments_merged.xlsx
```
- Core function: [`automapper_llm.assign_classes`](automapper/automapper_llm.py).
- Use `--use_llm` to enable LLM reranking (Gemini).
- Autosave partial results with `--autosave_every N` and resume with `--resume_partial`.

---

### 6. Review Results and Validate
- Use [automapper/validator.py](automapper/validator.py) to color-code Top-1/2/3 matches and create a summary sheet:
  ```sh
  python automapper/validator.py \
    --input_file results/item_class_assignments_merged.xlsx \
    --sheet_name Sheet1 \
    --top1_column top_1_class_name \
    --top2_column top_2_class_name \
    --top3_column top_3_class_name \
    --mapping_column mapping_ph \
    --output_file results/item_class_assignments_colored.xlsx
  ```

- Use [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py) to preview and convert `.pkl` files:
  ```sh
  python embeddings/embeddings_viewer.py path/to/embeddings.pkl --preview 5 --output embeddings.json
  ```

---

## Tips and Notes

- **Abbreviations and Embedding Quality:** Industrial item descriptions often contain abbreviations/shorthand (e.g., C/W, GALV, GRD). Since abbreviation usage differs across datasets and depends heavily on context, embeddings may become less accurate, which can reduce mapping quality. Therefore, it is important that we first try to solve the abbreviations issues in the datasets.

A key improvement is to expand abbreviations before generating embeddings / running the automapper, ideally by using an abbreviation list (per dataset or per sheet/category), if there is any.
Example abbreviation utilities in this repo:
utilities/abbreviation/expand_abbreviation.py (Using LLM to expand abbreviation in the items' description by utilizing the item's extra context information)

---