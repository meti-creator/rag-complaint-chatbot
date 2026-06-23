import pandas as pd
import numpy as np
import logging
import sys
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import project configuration
from src import config

# Setup robust logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ingest_data(input_csv=config.PROCESSED_DATA_PATH, output_db_path=config.DB_PATH):
    """
    Reads, samples, chunks, and persists complaint data to ChromaDB.
    """
    try:
        # 1. Load Data
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input CSV not found at: {input_csv}")
            
        logger.info(f"Loading dataset from {input_csv}...")
        df = pd.read_csv(input_csv)
        
        # Clean: Drop rows where critical narrative is missing
        df = df.dropna(subset=['cleaned_narrative'])
        
        # 2. Stratified Sampling
        TARGET_SIZE = 12500
        logger.info(f"Performing stratified sampling for {TARGET_SIZE} records...")
        
        # Ensure 'Product' exists for grouping
        if 'Product' not in df.columns:
            raise ValueError("The column 'Product' is missing from the dataset.")

        df_sampled = df.groupby('Product', group_keys=False).apply(
            lambda x: x.sample(n=int(np.rint(TARGET_SIZE * len(x) / len(df))), random_state=42)
        )
        logger.info(f"Sampled {len(df_sampled)} records successfully.")

        # 3. Initialize Embedding Model
        logger.info("Initializing embedding model (all-MiniLM-L6-v2)...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 4. Chunking and Metadata Mapping
        logger.info("Chunking text and preparing documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        
        documents = []
        for _, row in df_sampled.iterrows():
            content = str(row.get('cleaned_narrative', ''))
            splits = text_splitter.split_text(content)
            
            for i, split in enumerate(splits):
                metadata = {
                    "complaint_id": str(row.get('Complaint ID', 'N/A')),
                    "product_category": str(row.get('Product', 'N/A')),
                    "product": str(row.get('Sub-product', 'N/A')),
                    "issue": str(row.get('Issue', 'N/A')),
                    "sub_issue": str(row.get('Sub-issue', 'N/A')),
                    "company": str(row.get('Company', 'N/A')),
                    "state": str(row.get('State', 'N/A')),
                    "date_received": str(row.get('Date received', 'N/A')),
                    "chunk_index": i,
                    "total_chunks": len(splits)
                }
                documents.append(Document(page_content=split, metadata=metadata))

        # 5. Persist to ChromaDB
        logger.info(f"Ingesting {len(documents)} chunks into ChromaDB at {output_db_path}...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=output_db_path
        )
        logger.info("Ingestion successfully completed.")

    except FileNotFoundError as e:
        logger.error(f"File Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Critical System Error during ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    ingest_data()
    
    