import re
import pandas as pd
import numpy as np
import faiss
from openai import AzureOpenAI

def generate_embeddings_and_save(csv_path="methods.csv", batch_size=20):
    """
    Generate embeddings for code chunks and save FAISS index and metadata.
    
    Args:
        csv_path: Path to the CSV file
        batch_size: Number of items to process per batch
    
    Returns:
        tuple: (faiss_index, df_with_embeddings)
    """
    print("🚀 Starting embedding script with Azure OpenAI...")

    # 🔑 Azure OpenAI configuration
    endpoint = "https://azure-ai-hackthon.openai.azure.com/"
    deployment = "text-embedding-3-large"
    api_version = "2024-12-01-preview"
    key = ""

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_key=key
    )

    print("✅ Azure OpenAI client initialized!")

    # 🧹 Clean code bodies safely
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

    # 📄 Load CSV
    print(f"📄 Loading CSV file from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")

    print("🧹 Cleaning function bodies...")
    df['clean_body'] = df['Function Body'].apply(clean_body)

    # 🧩 Build input chunks
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

    # 🔄 Generate embeddings in batches
    def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
        """Request embeddings for a batch of texts, preserving order."""
        response = client.embeddings.create(
            model=deployment,
            input=texts
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    print("🔄 Generating embeddings in batches...")
    embeddings = []

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_texts = df['code_chunk'].iloc[start:end].tolist()
        print(f"🔍 Processing methods {start+1} → {end}/{len(df)}")
        
        batch_embeddings = get_embeddings_batch(batch_texts)
        embeddings.extend(batch_embeddings)

    # Attach back to DataFrame
    df['embedding'] = embeddings
    print("✅ Embedding generation complete!")

    # 💾 Build and save FAISS index
    print("💾 Building FAISS index...")
    embedding_matrix = np.vstack(df['embedding'].values).astype("float32")
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embedding_matrix)
    faiss.write_index(index, "code_embeddings.index")
    print(f"✅ FAISS index saved (dim={dim}, n={len(df)})")

    # 💾 Save metadata
    print("💾 Saving metadata (without embeddings)...")
    df.drop(columns=['embedding']).to_pickle("code_metadata.pkl")
    print("✅ Metadata saved to code_metadata.pkl")

    print("🎉 All done with Azure OpenAI embeddings (batched)!")
    
    return index, df


# Call the method
# if __name__ == "__main__":
#     index, df_with_embeddings = generate_embeddings_and_save()
