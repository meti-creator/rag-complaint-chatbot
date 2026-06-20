Task 1: Exploratory Data Analysis (EDA)
This directory contains the analytical exploration of the Consumer Complaint dataset. The objective of this analysis was to move beyond descriptive statistics and identify the systemic trends, data quality bottlenecks, and potential biases that would influence the design and performance of the RAG Chatbot pipeline.

Analytical Scope
The analysis focuses on three critical pillars:

Temporal Evolution: Identifying the growth and volatility of complaint volume over the last decade (2013–2025).

Entities & Sector Distribution: Mapping the landscape of "offending" companies to understand the bias toward credit reporting bureaus.

Data Integrity Audit: Quantifying missing values and assessing the viability of specific features for RAG ingestion.

 Key Findings & Strategic Implications
1. The Recency & Volume Shift
Observation: Complaint volume exhibited steady, linear growth until 2021, followed by an aggressive, near-exponential surge through 2025.

Strategic Impact: The RAG system will exhibit "recency bias." Because the vast majority of our data is concentrated in the 2023–2025 window, the chatbot will be highly adept at answering modern financial queries but may lack context for historical complaint resolution patterns.

2. Credit Reporting Dominance
Observation: Equifax, TransUnion, and Experian occupy the top three spots by a significant margin.

Strategic Impact: Our vector database is effectively a specialized corpus for credit reporting disputes. This informed our decision to implement metadata filtering on the company and product fields, allowing the RAG pipeline to isolate banking-specific issues that would otherwise be "drowned out" by the sheer volume of credit bureau data.

3. Data Quality Audit
Observation: Essential metadata (Product, Issue, Company) is 100% complete. However, the Consumer complaint narrative column contained ~6.6 million missing values.

Strategic Impact: This was the primary driver for our preprocessing strategy. We implemented a strict dropna() filter on the narratives during the ingest.py phase to ensure the RAG system only indexes high-quality, actionable consumer narratives.

 Tools & Stack
Analysis: Python 3.10, Pandas (data manipulation).

Visualization: Matplotlib & Seaborn (statistical plotting).

Environment: Jupyter Notebooks for literate programming and narrative data analysis.

 How to Reproduce
Ensure you have the cleaned dataset in ./data/cleaned_complaints.csv.

Launch Jupyter Lab: jupyter lab.

Open notebooks/eda.ipynb.

Run all cells sequentially to regenerate the visualizations and the data quality report.
# Consumer Complaint RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to ingest, vectorize, and query consumer complaints. It allows for semantic searching of large complaint datasets while enabling precise metadata filtering.

## Features
* **Vector Database:** Uses **ChromaDB** for efficient, persistent storage of document embeddings.
* **Semantic Search:** Powered by the `all-MiniLM-L6-v2` embedding model from HuggingFace.
* **Metadata Filtering:** Supports complex filtering (e.g., "Show me Credit Card complaints in NY") using the `$and` logical operator.
* **Modular Architecture:** Separates data ingestion (`ingest.py`) from the retrieval/query interface (`query.py`).

## Getting Started

### Prerequisites
* **Python 3.10+** installed.
* **Windows Users:** If you encounter DLL load failures (specifically related to PyTorch/`c10.dll`), ensure the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) is installed and your system is restarted.

### Installation
1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
