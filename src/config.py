"""
Project configuration.
Keep this in sync with whatever ingest.py / preprocess.py already expect.
"""

import os

# --- Paths ---
RAW_DATA_PATH = "data/raw/complaints.csv"
PROCESSED_DATA_PATH = "data/processed/cleaned_complaints.csv"

# Pre-built vector store resource for Tasks 3-4 (NOT re-embedded -- only loaded)
EMBEDDINGS_PARQUET_PATH = "data/raw/complaint_embeddings.parquet"

# Built once by `python -m src.build_index` from EMBEDDINGS_PARQUET_PATH
FAISS_INDEX_PATH = "data/faiss_index/faiss.index"
FAISS_META_PATH = "data/faiss_index/chunks_meta.parquet"

# --- Models ---
# Plain model name for direct sentence_transformers.SentenceTransformer use
# (query-time embedding only -- the corpus embeddings in the parquet were
# already produced with this same model).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pick whichever generator you actually have credentials for.
# "anthropic"  -> uses ANTHROPIC_API_KEY env var, model below (needs billing)
# "local_hf"   -> fully local, no API key, no cost
GENERATOR_BACKEND = os.environ.get("RAG_GENERATOR_BACKEND", "local_hf")
ANTHROPIC_MODEL_NAME = "claude-sonnet-4-6"

# Qwen2.5-0.5B-Instruct: free, local, causal LM (not seq2seq like FLAN-T5),
# 32K context window so retrieved chunks don't get truncated, and properly
# instruction-tuned for chat/QA. Using the 0.5B (not 1.5B) variant here
# specifically because of repeated download stalls on a slow connection --
# ~1GB instead of ~3GB. Quality is a step down from 1.5B but still a major
# improvement over FLAN-T5-base, which was too small/context-limited to
# synthesize multi-chunk complaint excerpts into coherent answers.
LOCAL_HF_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# --- Retrieval ---
DEFAULT_TOP_K = 5