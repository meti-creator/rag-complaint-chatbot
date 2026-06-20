import pandas as pd
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove boilerplate (simple regex pattern for example)
    text = re.sub(r"i am writing to file a complaint against.*?\.", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s\.,!?]", "", text)
    return text.strip()

def run_pipeline(input_path, output_path):
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    # 1. Filter: Drop empty narratives
    df = df.dropna(subset=['Consumer complaint narrative'])
    
    # 2. Filter: Target products
    target_products = ['Credit card', 'Personal loan', 'Checking or savings account', 'Money transfer, virtual currency, or money service']
    df = df[df['Product'].isin(target_products)]
    
    # 3. Clean: Apply text normalization
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    
    # 4. Save: Final processed data
    df.to_csv(output_path, index=False)
    print(f"Pipeline complete. Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    run_pipeline("data/raw/complaints.csv", "data/processed/cleaned_complaints.csv")