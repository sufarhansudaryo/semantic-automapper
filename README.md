# semantic-automapper

A toolkit for generating text embeddings from Excel data and automatically mapping items to a class hierarchy using a **hybrid** approach:

- **Fuzzy matching** (RapidFuzz)
- **Embedding similarity** (cosine similarity)
- **Optional LLM reranking** (Gemini)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
  - [Embeddings Generation](#embeddings-generation)
  - [Automapper](#automapper)
  - [Hierarchy Utilities](#hierarchy-utilities)
- [Quickstart](#quickstart)
  - [1 Install](#1-install)
  - [2 Configure Authentication](#2-configure-authentication)
  - [3 Generate Embeddings](#3-generate-embeddings)
  - [4 Build Class Node Table Optional](#4-build-class-node-table-optional)
  - [5 Run the Automapper](#5-run-the-automapper)
  - [6 Validate Results](#6-validate-results)
  - [7 Preview Embeddings](#7-preview-embeddings)
- [Tips and Notes](#tips-and-notes)
- [Recommended Workflow](#recommended-workflow)

---

## Overview

This repository focuses on two main features:

1. **Embeddings generation for Excel data** (threaded + retry)
2. **Automated item-to-class assignment** (Top-3 candidates using fuzzy + cosine similarity, with optional LLM reranking)

---

## Key Features

### Embeddings
- Reads Excel data and generates embeddings
- Threaded embedding generation for speed
- Retry + exponential backoff for stability
- Output stored as `.pkl` mapping: `ID → NumPy embedding vector`

### Automapper
- Hybrid matching (fuzzy + embedding cosine similarity)
- Outputs **Top-3 candidates**
- Optional **LLM reranking** (Gemini) on Top-K candidates
- Autosave + resume for long LLM runs

### Hierarchy Utilities
- Generate normalized class hierarchy node table from hierarchical Excel sheets
- Outputs node table with:
  - `id`, `name`, `level`, `parent_id`, `lowest_level_class`

---

## Project Structure

```text
semantic-automapper/
│
├── automapper/
│   ├── automapper_llm.py             # Main automapper script (fuzzy + cosine + optional LLM)
│   └── validator.py                  # Excel color-coding + summary for Top-1/2/3
│
├── embeddings/
│   ├── embeddings_generator/
│   │   ├── generate_embeddings.py    # CLI for generating embeddings from Excel
│   │   └── get_embeddings.py         # API client + threading + retry/backoff
│   └── embeddings_viewer.py          # Preview/convert embeddings (.pkl → JSON)
│
├── utilities/
│   ├── hierarchy_builder/
│   │   ├── generate_class_hierarchy.py
│   │   └── class_hierarchy_builder.py
│   │
│   ├── abbreviation/
│   │   ├── expand_abbreviation.py    # LLM-based abbreviation expansion
│   │   └── replace_abbreviation.py   # Replace abbreviations using mapping file
│   │
│   └── excel_extractor.py            # GUI tool for extracting unique rows from Excel + assign IDs
│
├── data/                             # Input Excel files + generated embeddings
├── results/                          # Output files (merged assignments, colored validations)
├── credentials/                      # Service account JSONs (ignored by .gitignore)
├── requirements.txt
└── README.md
```


## Core components (quick overview)

- **Embeddings**
  - Use [embeddings/embeddings_generator/generate_embeddings.py](embeddings/embeddings_generator/generate_embeddings.py) to read an Excel sheet and produce a `.pkl` mapping of IDs → NumPy embeddings.
  - The implementation uses [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py) which handles:
    - API calls to Google GenAI / Vertex AI via `google-genai`
    - Exponential backoff (`get_embedding`)
    - ThreadPool parallelism (`generate_embeddings`)
  - Inspect or convert `.pkl` files with [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py).

- **Automapper** (item → class mapping)
  - Algorithm lives in [automapper/semantic_automapper.py](automapper/semantic_automapper.py). The key function [`semantic_automapper.assign_classes`](automapper/semantic_automapper.py) implements:
    - Fuzzy matching (RapidFuzz) to shortlist candidate classes for each item
    - Cosine similarity (scikit-learn) between item and class embeddings to compute Top-3 matches
    - Optional filtering to lowest-level classes in the hierarchy
    - Batched similarity computation and output merging with original Excel
  - Validate and color-code results using [automapper/validator.py](automapper/validator.py).

- **Hierarchy utilities**
  - Create a normalized node table from a hierarchical Excel sheet via [utilities/hierarchy_builder/generate_class_hierarchy.py](utilities/hierarchy_builder/generate_class_hierarchy.py), which calls [`class_hierarchy_builder.extract_node_table`](utilities/hierarchy_builder/class_hierarchy_builder.py).
  - The node table includes unique `id`, `name`, `level`, `parent_id`, and `lowest_level_class` flags used by the automapper.

## Quickstart

1. **Install dependencies:**
```sh
pip install -r requirements.txt
```

2. **Configure authentication and settings:**
- Create / edit `.env.local` in the project root (this repo loads it in [embeddings/embeddings_generator/get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py)):
```sh
  - GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
  - GOOGLE_CLOUD_PROJECT=your-project-id
  - GOOGLE_CLOUD_LOCATION=your-location
  - GOOGLE_GENAI_USE_VERTEXAI=True/False
  ```

Note: Credentials files under [credentials/](credentials/) are ignored by [.gitignore](.gitignore). Do not commit secrets.

3. **Generate embeddings from Excel:**
```sh
python embeddings/embeddings_generator/generate_embeddings.py \
  "/path/to/data.xlsx" "Sheet1" "text_column_name" \
  --id "id_column" --workers 8 --output data/generated_embeddings/data_embeddings.pkl
```
- CLI calls [`get_embeddings.generate_embeddings`](embeddings/embeddings_generator/get_embeddings.py) which concurrently fetches embeddings using [`get_embeddings.get_embedding`](embeddings/embeddings_generator/get_embeddings.py).
- Tune `--workers` and `max_workers` in [get_embeddings.py](embeddings/embeddings_generator/get_embeddings.py) for throughput vs. quota.

4. **Build class node table (if you have a hierarchical sheet):**
```sh
python utilities/hierarchy_builder/generate_class_hierarchy.py \
  --input "/path/to/hierarchy.xlsx" --sheet "Hierarchy" --header-row 4 \
  --output node_table.xlsx --level-column-keywords Level
```
- This script uses [`class_hierarchy_builder.extract_node_table`](utilities/hierarchy_builder/class_hierarchy_builder.py).

5. **Run the automapper:**
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

6. **Review results and validate:**
- Use [automapper/validator.py](automapper/validator.py) to color-code Top-1/2/3 matches and create a summary sheet.
- Use [embeddings/embeddings_viewer.py](embeddings/embeddings_viewer.py) to preview and convert embeddings `.pkl` files.

## Tips and Notes

**Abbreviations and Embedding Quality:** 

Industrial item descriptions often contain abbreviations/shorthand (e.g., C/W, GALV, GRD). Since abbreviation usage differs across datasets and depends heavily on context, embeddings may become less accurate, which can reduce mapping quality. Therefore, it is important that we first try to solve the abbreviations issues in the datasets. A key improvement is to expand abbreviations before generating embeddings / running the automapper, ideally by using an abbreviation list (per dataset or per sheet/category), if there is any. 

Example abbreviation utilities in this repo: 
utilities/abbreviation/expand_abbreviation.py 
(Using LLM to expand abbreviation in the items' description by utilizing the item's extra context information)

**Main Item Name Extraction (Description → `Item_Name`):**
Industrial descriptions often include noise (dimensions, units, standards, packaging), which reduces embedding + fuzzy matching quality. A key improvement is to **extract only the main item name**
before generating embeddings / running the automapper.

Example utility in this repo:
- `utilities/extract_main_name.py`  
  (LLM-based extraction using few-shot prompting + threading + retries. Prefers `Description_Expanded`, falls back to `Description`.)


## Useful links (files & symbols)
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