# Hybrid-Code-Gen

This is Hybrid code gen.

# Hybrid AST+KG RAG Pipeline

This project extracts method‑level AST metadata from Java source code, indexes the embeddings with FAISS, and allows real‑time retrieval of the top‑3 most similar methods for any text query.

---

## 1. AST Extraction & Embedding Generation

```bash
# Generate AST metadata, embeddings, and build the FAISS index
python main.py
```

## 2. Real‑time Query

```bash
# Run a single query against your index
python query.py
```

---

## 3. Java Code Repo KG Builder

To get all the information about Java code repo parser, [Read here](javarepoparser/README.md)
