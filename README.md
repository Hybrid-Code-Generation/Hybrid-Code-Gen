# Hybrid-Code-Gen
This is Hybrid code gen.


# Hybrid AST+KG RAG Pipeline

This project extracts method‑level AST metadata from Java source code, trains a GNN on the resulting graphs, indexes the embeddings with FAISS, and allows real‑time retrieval of the top‑3 most similar methods for any text query.

---

## 1. AST Extraction

```bash
# 1.1 Compile the Java AST extractor
javac -cp javaparser-core-3.x.jar AstExtractor.java

# 1.2 Run against your source directory, outputting methods.csv
java -cp .:javaparser-core-3.x.jar AstExtractor /path/to/your/java/src methods.csv
````

* **Input:** Java source directory
* **Output:** `methods.csv` (one row per method with metadata: file path, package, class, signature, body, LOC, etc.)

---

## 2. GNN Embedding & FAISS Index Construction

```bash
# 2.1 Install Python dependencies
pip install torch torch-geometric faiss-cpu pandas

# 2.2 Train the GNN and build your FAISS index
python gnnpipeline.py methods.csv --build
```

This step will:

1. Parse `methods.csv` into PyG graphs
2. Train—or load—a GraphSAGE model
3. Compute and normalize a fixed-size embedding for each method
4. Build and persist:

   * `method_index.faiss` (FAISS index)
   * `method_ids.pkl`  (mapping from index positions back to method IDs)

---

## 3. Real‑time Query

```bash
# Run a single query against your index
python gnnpipeline.py methods.csv --query "hash passwords securely"
```

This will:

* Convert your text prompt into a minimal AST‑style query graph
* Encode it via the trained GNN
* Normalize the resulting vector
* Perform a top‑3 nearest‑neighbor search in FAISS
* Print each method’s ID and similarity score

---

## Artifacts Produced

* `methods.csv`          – Flat CSV of extracted AST metadata
* `method_index.faiss`   – FAISS index of 256‑dim method embeddings
* `method_ids.pkl`       – Python pickle mapping FAISS indices → method IDs

> *Tip: Tweak the Java graph construction, GNN architecture, and training objective to best reflect your codebase’s actual AST patterns and semantics.*

```
```
