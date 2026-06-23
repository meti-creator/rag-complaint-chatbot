"""
build_index.py

Task 3, step 1: "Load the pre-built vector store from the dataset resources."

This builds a FAISS index directly from the pre-computed embeddings in
data/raw/complaint_embeddings.parquet -- no re-embedding happens here, since
the vectors already exist in the 'embedding' column (384-dim, produced by
all-MiniLM-L6-v2, matching Task 2's embedding model).

Schema of complaint_embeddings.parquet (confirmed via inspect_embeddings.py):
    id        : str   -- e.g. "14069121_0"  (complaint_id + chunk_index)
    document  : str   -- the chunk text
    embedding : object -- numpy array, shape (384,)
    metadata  : object -- dict with complaint_id, company, product, product_category,
                           issue, date_received, chunk_index, etc.

Because this dataset is large (~1.37M rows), this script:
  1. Reads the parquet once.
  2. Stacks all embeddings into a single (N, 384) float32 numpy matrix.
  3. Builds a FAISS index over that matrix.
  4. Saves three artifacts to disk:
       - faiss.index           (the FAISS index itself)
       - chunks_meta.parquet   (id, document, metadata -- aligned by row position
                                 with the FAISS index, so index position i
                                 corresponds to row i in this file)

Run once, from the project root:
    python -m src.build_index
"""

import logging
import os

import numpy as np
import pandas as pd
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PARQUET_PATH = os.path.join("data", "raw", "complaint_embeddings.parquet")
INDEX_OUTPUT_PATH = os.path.join("data", "faiss_index", "faiss.index")
META_OUTPUT_PATH = os.path.join("data", "faiss_index", "chunks_meta.parquet")


def build_index(
    parquet_path: str = PARQUET_PATH,
    index_output_path: str = INDEX_OUTPUT_PATH,
    meta_output_path: str = META_OUTPUT_PATH,
):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Could not find {parquet_path}")

    os.makedirs(os.path.dirname(index_output_path), exist_ok=True)

    logger.info(f"Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows.")

    logger.info("Stacking embeddings into a single matrix...")
    # Each row's 'embedding' is already a numpy array of shape (384,).
    # np.vstack assembles them into one (N, 384) matrix.
    embedding_matrix = np.vstack(df["embedding"].to_numpy()).astype("float32")
    n_rows, dim = embedding_matrix.shape
    logger.info(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Normalize vectors so inner product == cosine similarity.
    # all-MiniLM-L6-v2 embeddings are not guaranteed unit-norm out of the box,
    # and cosine similarity (not raw dot product) is the standard choice for
    # sentence-embedding retrieval.
    logger.info("Normalizing vectors for cosine similarity search...")
    faiss.normalize_L2(embedding_matrix)

    logger.info(f"Building FAISS index (IndexFlatIP, dim={dim})...")
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine similarity
    index.add(embedding_matrix)
    logger.info(f"Index built with {index.ntotal:,} vectors.")

    logger.info(f"Saving FAISS index to {index_output_path} ...")
    faiss.write_index(index, index_output_path)

    logger.info(f"Saving aligned metadata/text to {meta_output_path} ...")
    # Keep only what the retriever needs at query time. Row order here MUST
    # match the row order used to build embedding_matrix above, since FAISS
    # returns positional indices, not IDs.
    meta_df = df[["id", "document", "metadata"]].reset_index(drop=True)
    meta_df.to_parquet(meta_output_path)

    logger.info("Done.")
    return index, meta_df


if __name__ == "__main__":
    build_index()
