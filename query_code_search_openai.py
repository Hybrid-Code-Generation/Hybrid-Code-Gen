import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI

print("🚀 Starting script with Azure OpenAI embeddings...")

# 🔑 Azure OpenAI configuration
endpoint = "https://azure-ai-hackthon.openai.azure.com/"
deployment = "text-embedding-3-large"
api_version = "2024-12-01-preview"
key = ""

client = AzureOpenAI(
    api_key=key,
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_deployment=deployment,
)

print("✅ Azure OpenAI client initialized!")

# 📄 Load CSV
df = pd.read_csv("extracted_code_data.csv")

# 🧹 Preprocess code chunks
def clean_code(row):
    method_sig = f"{row['Modifiers'] or ''} {row['Return Type']} {row['Method Name']}({row['Parameters'] or ''})".strip()
    method_sig = " ".join(method_sig.split())
    function_body = row["Function Body"].replace("\n", " ").replace("\t", " ").strip()
    function_body = " ".join(function_body.split())
    return f"{method_sig} {{ {function_body} }}"

df["code_chunk"] = df.apply(clean_code, axis=1)

# 🔄 Generate embeddings
def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=deployment,
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

print("🔄 Generating embeddings for code chunks...")
df["embedding"] = df["code_chunk"].map(get_embedding)
print("✅ Embeddings generated!")

# 🔎 Similarity search
def search(query: str, top_k: int = 3):
    query_emb = get_embedding(query).reshape(1, -1)
    code_embeddings = np.vstack(df["embedding"].to_numpy())
    similarities = cosine_similarity(query_emb, code_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][["Method Name", "code_chunk", "FilePath"]]

# ▶️ Example query
if __name__ == "__main__":
    query = "how to show an error alert in JavaFX"
    results = search(query)
    print("Top matches:")
    for idx, row in results.iterrows():
        print(f"\nMethod: {row['Method Name']}\nPath: {row['FilePath']}\nCode:\n{row['code_chunk']}")
