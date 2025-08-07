import re
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, T5EncoderModel

print("🚀 Starting embedding script...")

# Load CodeT5+ 770M with GPU support
model_name = "Salesforce/codet5p-770m"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

print("🔃 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

print("🔃 Loading model...")
model = T5EncoderModel.from_pretrained(model_name).to(device)
#model = AutoModel.from_pretrained(model_name).to(device)
print("✅ Model and tokenizer loaded!")

# Clean code bodies safely
def clean_body(body: str) -> str:
    if not isinstance(body, str):
        return ""
    body = re.sub(r'//.*?$|/\*.*?\*/', '', body, flags=re.DOTALL | re.MULTILINE)
    body = re.sub(r'logger\.[a-zA-Z_]+\s*\([^;]*\);\s*', '', body)
    body = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', '<STR>', body)
    def split_id(match):
        text = match.group(0)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return text.replace('_', ' ')
    body = re.sub(r'\b[A-Za-z0-9_]+\b', split_id, body)
    body = re.sub(r'\s+', ' ', body).strip()
    return ' '.join(body.split(' ')[:200])

# Load CSV
csv_path = "methods.csv"
print(f"📄 Loading CSV file from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"✅ Loaded {len(df)} rows")

print("🧹 Cleaning function bodies...")
df['clean_body'] = df['Function Body'].apply(clean_body)

# Build input chunks
print("🧩 Constructing code chunks...")
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
print("✅ Code chunks created!")

# Generate embeddings with progress
def get_embedding(text: str) -> np.ndarray:
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

print("🔄 Generating embeddings...")
embeddings = []
for i, chunk in enumerate(df['code_chunk']):
    print(f"🔍 Processing method {i+1}/{len(df)}")
    emb = get_embedding(chunk)
    embeddings.append(emb)

df['embedding'] = embeddings
print("✅ Embedding generation complete!")

# Build and save FAISS index
print("💾 Building FAISS index...")
embedding_matrix = np.vstack(df['embedding'].values).astype("float32")
dim = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embedding_matrix)
faiss.write_index(index, "code_embeddings.index")
print(f"✅ FAISS index saved (dim={dim}, n={len(df)})")

# Save metadata
print("💾 Saving metadata (without embeddings)...")
df.drop(columns=['embedding']).to_pickle("code_metadata.pkl")
print("✅ Metadata saved to code_metadata.pkl")

print("🎉 All done!")
