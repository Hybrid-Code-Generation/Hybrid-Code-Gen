import re
import pandas as pd
import numpy as np
import faiss
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

# 1. Load CodeT5+ embedding model
model_name = "Salesforce/codet5p-220m-embedding"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. Cleaning function for code bodies
def clean_body(body: str) -> str:
    # (a) Strip single-line and multi-line comments
    body = re.sub(r'//.*?$|/\*.*?\*/', '', body, flags=re.DOTALL | re.MULTILINE)
    # (b) Remove logging statements
    body = re.sub(r'\blogger\.[a-zA-Z_]+\s*\([^;]*\);\s*', '', body)
    # (c) Template string literals
    body = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', '<STR>', body)
    # (d) Normalize identifiers: split camelCase and snake_case
    def split_id(match):
        text = match.group(0)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # camelCase → camel Case
        return text.replace('_', ' ')                     # snake_case → snake case
    body = re.sub(r'\b[A-Za-z0-9_]+\b', split_id, body)
    # (e) Collapse whitespace
    body = re.sub(r'\s+', ' ', body).strip()
    # (f) Trim to first 200 words
    words = body.split(' ')
    return ' '.join(words[:200])

# 3. Load and preprocess your CSV
csv_path = "methods.csv"  # adjust path as needed
df = pd.read_csv(csv_path)

# Apply cleaning to the Function Body
df['clean_body'] = df['Function Body'].apply(clean_body)

# 4. Construct the embedding input per method
def construct_code_chunk(row):
    return (
        f"Method: {row['Method Name']}\n"
        f"Class: {row['Class']}\n"
        f"Parameters: {row['Parameters']}\n"
        f"Return Type: {row['Return Type']}\n"
        f"Modifiers: {row['Modifiers']}\n\n"
        f"Body:\n{row['clean_body']}"
    )

df['code_chunk'] = df.apply(construct_code_chunk, axis=1)

# 5. Generate embeddings
def get_embedding(text: str) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    # Mean-pool token embeddings
    return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

print("🔄 Generating embeddings for each method…")
df['embedding'] = df['code_chunk'].apply(get_embedding)

# 6. Build FAISS index
embedding_matrix = np.vstack(df['embedding'].values)
dim = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embedding_matrix)
faiss.write_index(index, "code_embeddings.index")
print(f"✅ FAISS index saved to code_embeddings.index (dim={dim}, n={len(df)})")

# 7. Save metadata (without bulky embeddings)
df.drop(columns=['embedding']).to_pickle("code_metadata.pkl")
print("✅ Metadata saved to code_metadata.pkl")
