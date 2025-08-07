import pandas as pd
import numpy as np
import torch
import faiss
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ------------------- CONFIG -------------------
MODEL_NAME = "microsoft/codebert-base"
EMBEDDING_DIM = 768
MAX_LENGTH = 256

# ------------------- LOAD CODEBERT -------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ------------------- POOLING FUNCTION -------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of output: last hidden state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)
    return pooled

# ------------------- EMBED METHOD -------------------
def embed_code(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
    return emb[0].cpu().numpy()

# ------------------- BUILD FAISS INDEX -------------------
def build_index(embeddings, ids, index_path="codebert_index.faiss", map_path="method_ids.pkl"):
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(map_path, "wb") as f:
        pickle.dump(ids, f)
    print(f"[✔] Index saved to {index_path} and ID map to {map_path}")

# ------------------- QUERY TOP K -------------------
def query_top_k(prompt, k=3, index_path="codebert_index.faiss", map_path="method_ids.pkl"):
    q_emb = embed_code(prompt).reshape(1, -1)
    faiss.normalize_L2(q_emb)
    index = faiss.read_index(index_path)
    D, I = index.search(q_emb, k)
    with open(map_path, "rb") as f:
        ids = pickle.load(f)
    return [(ids[i], float(D[0, j])) for j, i in enumerate(I[0])]

# ------------------- MAIN -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to your method-level CSV")
    parser.add_argument("--build", action="store_true", help="Build embeddings + index")
    parser.add_argument("--query", help="Enter a prompt to get top 3 similar methods")
    args = parser.parse_args()

    if args.build:
        df = pd.read_csv(args.csv).dropna(subset=["Function Body"])
        embeddings = []
        ids = []

        print("[📦] Generating embeddings...")
        for idx, code in tqdm(enumerate(df["Function Body"]), total=len(df)):
            try:
                vec = embed_code(code)
                embeddings.append(vec)
                ids.append({
                    "method_id": idx,
                    "method_name": df.loc[idx, "Method Name"],
                    "class": df.loc[idx, "Class"],
                    "file": df.loc[idx, "FilePath"],
                    "code": code
                })
            except Exception as e:
                print(f"[!] Skipped method {idx}: {e}")

        embeddings = np.vstack(embeddings)
        build_index(embeddings, ids)

    if args.query:
        top3 = query_top_k(args.query)
        print("\n[🔍] Top 3 similar methods:")
        for method, score in top3:
            print(f"\n🔹 Score: {score:.4f}")
            print(f"📄 {method['file']} :: {method['class']} :: {method['method_name']}")
            print(f"```java\n{method['code'][:500]}...\n```")
