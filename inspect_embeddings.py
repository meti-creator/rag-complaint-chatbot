"""
inspect_embeddings.py

Loads complaint_embeddings.parquet from data/raw and prints its structure
so we can confirm column names, dtypes, and the embedding format before
wiring it into the RAG retriever.

Usage (from your project root, e.g. rag-complaint-chatbot/):
    python inspect_embeddings.py
"""

import pandas as pd
import numpy as np
import os

PARQUET_PATH = os.path.join("data", "raw", "complaint_embeddings.parquet")


def main():
    if not os.path.exists(PARQUET_PATH):
        print(f"File not found at: {PARQUET_PATH}")
        print("Make sure complaint_embeddings.parquet is placed inside data/raw/")
        return

    print(f"Loading: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)

    print("\n=== SHAPE ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(list(df.columns))

    print("\n=== DTYPES ===")
    print(df.dtypes)

    print("\n=== FIRST ROW (truncated) ===")
    row0 = df.iloc[0].to_dict()
    for k, v in row0.items():
        s = str(v)
        print(f"  {k}: {s[:200]}{'...' if len(s) > 200 else ''}")

    # If there's a column that looks like an embedding, show its shape/type
    print("\n=== LIKELY EMBEDDING COLUMN CHECK ===")
    for col in df.columns:
        sample = df.iloc[0][col]
        if isinstance(sample, (list, np.ndarray)):
            length = len(sample)
            print(f"  Column '{col}' looks like a vector: length={length}, type={type(sample)}")


if __name__ == "__main__":
    main()
