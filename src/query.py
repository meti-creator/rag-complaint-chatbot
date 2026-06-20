from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def query_system(query_text, product_filter=None, state_filter=None):
    """
    Queries the vector database with optional metadata filtering.
    """
    # 1. Initialize the embedding model
    # Must match the model used during ingestion!
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Connect to the existing ChromaDB
    db = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)

    # 3. Construct the 'where' filter logic
    # ChromaDB requires explicit operators for multiple filters
    conditions = []
    if product_filter:
        conditions.append({"product_category": product_filter})
    if state_filter:
        conditions.append({"state": state_filter})

    # Prepare the filter dictionary
    where_clause = None
    if len(conditions) > 1:
        where_clause = {"$and": conditions}
    elif len(conditions) == 1:
        where_clause = conditions[0]

    # 4. Perform the Similarity Search
    print(f"\n--- Querying: '{query_text}' ---")
    if where_clause:
        print(f"Applying Filters: {where_clause}")
    
    results = db.similarity_search(
        query=query_text,
        k=3, # Returns the top 3 most relevant chunks
        filter=where_clause
    )

    # 5. Output Results
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    # Test 1: General search (no filters)
    query_system("I was overcharged on my statement.")
    
    # Test 2: Specific search (Filtering for Credit Card complaints in NY)
    # Note: Ensure the values match how they were ingested in your CSV
    query_system("I was overcharged on my statement.", product_filter="Credit card", state_filter="NY")