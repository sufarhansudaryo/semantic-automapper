import os
import time
import random
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai.types import EmbedContentConfig


# Dynamically find the project root by looking for .env.local
current = Path(__file__).resolve()
for parent in current.parents:
    env_path = parent / ".env.local"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment from: {env_path}")
        break
else:
    print("Error: .env.local not found in any parent directory.")


project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
use_vertexai = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "False").lower() == "true"

print(f"Using project: {project_id}, location: {location}, VertexAI: {use_vertexai}")

client = genai.Client(
    vertexai=use_vertexai,
    project=project_id,
    location=location
)

model = "gemini-embedding-001"


def get_embedding(text, max_retries=8, initial_delay=1.0):
    """Fetch embedding with exponential backoff retries."""
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model=model,
                contents=text,
                config=EmbedContentConfig(task_type="CLASSIFICATION"),
            )
            return np.array(response.embeddings[0].values)
        except Exception as e:
            tqdm.write(f"[Attempt {attempt}] Error: {e}")
            if attempt == max_retries:
                tqdm.write("Max retries reached. Skipping this entry.")
                return None
            sleep_time = delay * (1 + random.random() * 0.1)
            tqdm.write(f"Retrying in {sleep_time:.1f} s ...")
            time.sleep(sleep_time)
            delay *= 2


def generate_embeddings(
    file_path,
    sheet_name,
    text_column,
    id_column=None,
    max_workers=10,
):
    """Generate embeddings for a column in an Excel sheet."""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        tqdm.write(f"Loaded {len(df)} rows from {file_path}, sheet '{sheet_name}'")
    except Exception as e:
        tqdm.write(f"Error reading Excel file: {e}")
        return None
    
    # Show exact column names from Excel
    print("\nDEBUG: Columns loaded from Excel")
    print(df.columns.tolist())
    print("=======================================\n")

    if text_column not in df.columns:
        tqdm.write(f"Column '{text_column}' not found.")
        return None

    tasks = []
    for idx, row in df.iterrows():
        text = row[text_column]
        if pd.isna(text) or str(text).strip() == "":
            continue
        item_id = row[id_column] if id_column and id_column in df.columns else idx
        tasks.append((item_id, str(text)))

    tqdm.write(f"Prepared {len(tasks)} valid rows for embedding.")

    embeddings = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(get_embedding, t): i for i, t in tasks}
        for future in tqdm(
            as_completed(future_to_id), total=len(future_to_id), desc="Embedding"
        ):
            item_id = future_to_id[future]
            try:
                emb = future.result()
                if emb is not None:
                    embeddings[item_id] = emb
            except Exception as e:
                tqdm.write(f"Error embedding ID {item_id}: {e}")

    return embeddings
