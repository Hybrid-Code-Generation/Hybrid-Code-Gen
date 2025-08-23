import re
import pandas as pd
import numpy as np
import faiss
from openai import AzureOpenAI
import nltk
from nltk.stem import WordNetLemmatizer

# -------------------------------
# 1️⃣ Azure OpenAI Configuration
# -------------------------------
endpoint = "https://azure-ai-hackthon.openai.azure.com/"
deployment = "text-embedding-3-large"
api_version = "2024-12-01-preview"
key = ""  # Your Azure OpenAI key here

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_deployment=deployment,
    api_key=key
)
print("✅ Azure OpenAI client initialized!")

# -------------------------------
# 2️⃣ Load FAISS index + metadata
# -------------------------------
print("📄 Loading metadata and FAISS index...")
df = pd.read_pickle("code_metadata.pkl")
index = faiss.read_index("code_embeddings.index")
print(f"✅ Loaded {len(df)} methods and FAISS index.")

# -------------------------------
# 3️⃣ Advanced Query Cleaning
# -------------------------------
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stopwords = {"the", "a", "an", "is", "to", "for", "of", "and", "in", "on", "by", "with", "from"}

def clean_query_advanced(query: str) -> str:
    query = query.lower()
    query = re.sub(r'[^a-z0-9_ ]', ' ', query)  # keep underscores for identifiers
    words = query.split()
    # Remove stopwords
    words = [w for w in words if w not in stopwords]
    # Split camelCase identifiers
    words = [re.sub(r'([a-z])([A-Z])', r'\1 \2', w) for w in words]
    # Convert snake_case to space
    words = [w.replace('_', ' ') for w in words]
    # Lemmatization
    words = [lemmatizer.lemmatize(w) for w in words]
    # Limit to first 30 tokens
    return " ".join(words[:30])


# we could also use a simpler cleaning function if preferred
# currently not used
def clean_query(query: str) -> str:
    # Lowercase, remove extra spaces and punctuation (basic cleaning)
    query = query.lower()
    query = re.sub(r'[^a-z0-9\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query



# -------------------------------
# 4️⃣ Generate embedding for query
# -------------------------------
def get_query_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model=deployment,
        input=[text]
    )
    return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

# -------------------------------
# 5️⃣ Search top-k using FAISS
# -------------------------------
def search_top_k(query: str, k: int = 3):
    cleaned_query = clean_query_advanced(query)
    query_embedding = get_query_embedding(cleaned_query)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    # Note: If your index embeddings were not normalized originally, L2 distance is a rough approx
    
    distances, indices = index.search(query_embedding, k)
    top_methods = df.iloc[indices[0]]['Method Name'].tolist()
    
    return top_methods

# -------------------------------
# 6️⃣ Example usage
# -------------------------------
if __name__ == "__main__":
    user_query = input("Enter your code search query: ")
    top_matches = search_top_k(user_query, k=3)
    print("\nTop 3 matching methods:")
    for i, method in enumerate(top_matches, 1):
        print(f"{i}. {method}")
