import re
import pandas as pd
import numpy as np
import faiss
from openai import AzureOpenAI
import nltk
from nltk.stem import WordNetLemmatizer

class CodeSearcher:
    def get_code_embedding(self, code_text: str) -> np.ndarray:
        """
        Generate an embedding for a code snippet or method body using Azure OpenAI.
        """
        response = self.client.embeddings.create(
            model=self.deployment,
            input=[code_text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    def __init__(self):
        self.client = None
        self.df = None
        self.index = None
        self.lemmatizer = None
        self.stopwords = {"the", "a", "an", "is", "to", "for", "of", "and", "in", "on", "by", "with", "from"}
        self.deployment = "text-embedding-3-large"
        
    def initialize(self):
        """Initialize Azure OpenAI client, load data, and setup NLP tools"""
        # Azure OpenAI Configuration
        endpoint = "https://azure-ai-hackthon.openai.azure.com/"
        api_version = "2024-12-01-preview"
        key = ""
        
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_deployment=self.deployment,
            api_key=key
        )
        print("✅ Azure OpenAI client initialized!")
        
        # Load FAISS index + metadata
        print("📄 Loading metadata and FAISS index...")
        self.df = pd.read_pickle("code_metadata.pkl")
        self.index = faiss.read_index("code_embeddings.index")
        print(f"✅ Loaded {len(self.df)} methods and FAISS index.")
        
        # Setup NLP tools
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_query_advanced(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r'[^a-z0-9_ ]', ' ', query)
        words = query.split()
        words = [w for w in words if w not in self.stopwords]
        words = [re.sub(r'([a-z])([A-Z])', r'\1 \2', w) for w in words]
        words = [w.replace('_', ' ') for w in words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return " ".join(words[:30])
    
    def clean_query(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r'[^a-z0-9\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query
    
    def get_query_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.deployment,
            input=[text]
        )
        return np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    
    def search_top_k(self, query: str, k: int = 3):
        cleaned_query = self.clean_query_advanced(query)
        query_embedding = self.get_query_embedding(cleaned_query)
        
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            method_info = {
                'Class': self.df.iloc[idx]['Class'],
                'Method Name': self.df.iloc[idx]['Method Name'],
                'Return Type': self.df.iloc[idx]['Return Type'],
                'Parameters': self.df.iloc[idx]['Parameters']
            }
            results.append(method_info)
        
        return results
    
def main():
    """Main function for standalone execution"""
    searcher = CodeSearcher()
    searcher.initialize()
    
    user_query = input("Enter your code search query: ")
    top_matches = searcher.search_top_k(user_query, k=3)
    print("\nTop 3 matching methods:")
    for i, method in enumerate(top_matches, 1):
        print(f"{i}. {method}")

# Usage example:
# if __name__ == "__main__":
#     main()
