import pandas as pd
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import logging

# Set up logging for visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_data(input_csv, output_db_path="./data/chroma_db"):
    """
    Reads data, samples it, chunks text with metadata, and persists to ChromaDB.
    """
    # 1. Load Data
    logger.info("Loading cleaned dataset...")
    df = pd.read_csv(input_csv)
    
    # 2. Stratified Sampling (10,000 - 15,000 target)
    TARGET_SIZE = 12500
    logger.info(f"Performing stratified sampling for {TARGET_SIZE} records...")
    
    df_sampled = df.groupby('Product', group_keys=False).apply(
        lambda x: x.sample(n=int(np.rint(TARGET_SIZE * len(x) / len(df))), random_state=42)
    )
    logger.info(f"Sampled {len(df_sampled)} records successfully.")

    # 3. Initialize Embedding Model
    logger.info("Initializing embedding model: all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 4. Chunking and Metadata Mapping
    logger.info("Chunking text and preparing documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    
    documents = []
    for _, row in df_sampled.iterrows():
        # Clean text
        content = str(row.get('cleaned_narrative', ''))
        if not content or content == 'nan':
            continue
            
        # Split text
        splits = text_splitter.split_text(content)
        
        # Attach Metadata
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
    
    logger.info("Ingestion pipeline complete.")
    return vectorstore

if __name__ == "__main__":
    # Ensure your file path is correct
    INPUT_FILE = '../data/processed/cleaned_complaints.csv'
    ingest_data(INPUT_FILE)
    