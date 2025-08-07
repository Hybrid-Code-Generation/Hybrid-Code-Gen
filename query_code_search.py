
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load CSV
df = pd.read_csv("extracted_code_data.csv")

# Preprocess code chunks for similarity search
def clean_code(row):
    method_sig = f"{row['Modifiers'] or ''} {row['Return Type']} {row['Method Name']}({row['Parameters'] or ''})".strip()
    method_sig = " ".join(method_sig.split())
    function_body = row["Function Body"].replace("\n", " ").replace("\t", " ").strip()
    function_body = " ".join(function_body.split())
    return f"{method_sig} {{ {function_body} }}"

df["code_chunk"] = df.apply(clean_code, axis=1)

# Load CodeT5+ embedding model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Salesforce/codet5p-220m-embedding"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

# Generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy()[0]

df["embedding"] = df["code_chunk"].map(get_embedding)

# Similarity search
def search(query, top_k=3):
    query_emb = get_embedding(query).reshape(1, -1)
    code_embeddings = np.vstack(df["embedding"].to_numpy())
    similarities = cosine_similarity(query_emb, code_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][["Method Name", "code_chunk", "FilePath"]]

# Example query
if __name__ == "__main__":
    query = "how to show an error alert in JavaFX"
    results = search(query)
    print("Top matches:")
    for idx, row in results.iterrows():
        print(f"\nMethod: {row['Method Name']}\nPath: {row['FilePath']}\nCode:\n{row['code_chunk']}")
