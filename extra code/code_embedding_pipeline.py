from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
import faiss
import joblib

# Load CodeT5+ embedding model
model_name = "Salesforce/codet5p-220m-embedding"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load CSV
csv_path = "methods.csv"  # Change this if needed
df = pd.read_csv(csv_path)

# Construct code chunks
def construct_code_chunk(row):
    return f"""Method: {row['Method Name']}
Class: {row['Class']}
Parameters: {row['Parameters']}
Return Type: {row['Return Type']}
Modifiers: {row['Modifiers']}
Function Body:
{row['Function Body']}
"""

df["code_chunk"] = df.apply(construct_code_chunk, axis=1)

# Generate embedding
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()

# Apply to all code chunks
print("Generating embeddings...")
df["embedding"] = df["code_chunk"].apply(get_embedding)

# Stack all embeddings
embedding_matrix = np.vstack(df["embedding"].values)
embedding_dim = embedding_matrix.shape[1]

# Create and save FAISS index
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)
faiss.write_index(index, "code_embeddings.index")

# Save metadata
df.drop(columns=["embedding"]).to_pickle("code_metadata.pkl")

print("✅ Embeddings and FAISS index saved.")
