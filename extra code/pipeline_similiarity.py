import os
import pandas as pd
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIGURATION ===
CSV_PATH =  r"C:\Users\9idiv\OneDrive\Desktop\Hybrid-Code-Gen\AST\java_parsed.csv"           # Your AST CSV file
EMBEDDING_PKL_PATH = "method_embeddings.pkl"
OPENAI_MODEL = "text-embedding-3-small"

# === SET YOUR OPENAI API KEY ===
openai.api_key = ""  # Replace with your key or use environment variables

# === Prepare text for each method ===
def prepare_method_text(row):
    return f"Method Name: {row['Method Name']}\nParameters: {row['Parameters']}\nFunction Body:\n{row['Function Body']}"

# === Get embedding from OpenAI ===
def get_embedding(text, model=OPENAI_MODEL):
    text = text.replace("\n", " ")[:8191]  # OpenAI recommends removing line breaks and truncating if needed
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

# === Load or create embeddings ===
if os.path.exists(EMBEDDING_PKL_PATH):
    print("🔁 Loading saved method embeddings...")
    df = pd.read_pickle(EMBEDDING_PKL_PATH)
else:
    print("🚀 No saved embeddings found. Generating...")
    df = pd.read_csv(CSV_PATH)
    df["combined_text"] = df.apply(prepare_method_text, axis=1)
    df["embedding"] = df["combined_text"].apply(get_embedding)
    df.to_pickle(EMBEDDING_PKL_PATH)
    print("✅ Embeddings saved to:", EMBEDDING_PKL_PATH)

# === Prepare embedding matrix for similarity search ===
embedding_matrix = np.vstack(df["embedding"].values)

# === Query the system ===
while True:
    query = input("\n🧠 Enter your natural language query (or type 'exit'): ")
    if query.lower() in {"exit", "quit"}:
        break

    query_embedding = get_embedding(query)
    similarities = cosine_similarity([query_embedding], embedding_matrix)[0]
    df["similarity"] = similarities
    top_matches = df.sort_values("similarity", ascending=False).head(3)

    print("\n🎯 Top 3 matching methods:\n")
    for idx, row in top_matches.iterrows():
        print(f"🔹 Method Name: {row['Method Name']}")
        print(f"   Class: {row['Class']} | File: {row['FilePath']}")
        print(f"   Similarity Score: {row['similarity']:.4f}")
        print("-" * 60)
