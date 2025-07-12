# File: gnnpipeline.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
import faiss
import pickle
import argparse

# 1. Load CSV of methods into DataFrame
def load_methods(csv_path):
    df = pd.read_csv(csv_path)
    # assign a unique numeric ID per row
    df['method_id'] = np.arange(len(df))
    return df

# 2. Build PyG Data objects (simple example: nodes=lines, edges sequential)
def build_graphs(df):
    graphs = []
    for _, row in df.iterrows():
        lines = row['Body'].split('\\n')
        num_nodes = len(lines)
        # node features could be simple bag‑of‑words or length
        x = torch.tensor([[len(l)] for l in lines], dtype=torch.float)
        # linear chain edges
        edge_index = torch.tensor([list(range(num_nodes-1)) + list(range(1, num_nodes)),
                                   list(range(1, num_nodes)) + list(range(num_nodes-1))],
                                  dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=None)
        data.method_id = row['method_id']
        graphs.append(data)
    return graphs

# 3. Define GraphSAGE encoder
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats=1, hidden=64, out_feats=256):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden)
        self.conv2 = SAGEConv(hidden, out_feats)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return global_mean_pool(x, batch)  # (batch_size, out_feats)

# 4. Train or load pretrained model (omitted: contrastive training loop)
def train_gnn(graphs):
    loader = DataLoader(graphs, batch_size=32, shuffle=True)
    model = GraphSAGE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # ... implement contrastive/supervised training here ...
    return model

# 5. Compute embeddings & build FAISS index
def build_index(model, graphs, dim=256, index_path="method_index.faiss", map_path="method_ids.pkl"):
    model.eval()
    emb_list, id_list = [], []
    loader = DataLoader(graphs, batch_size=64)
    with torch.no_grad():
        for batch in loader:
            ev = model(batch).cpu().numpy()
            emb_list.append(ev)
            id_list.extend(batch.method_id.tolist())
    embeddings = np.vstack(emb_list)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(map_path, "wb") as f:
        pickle.dump(id_list, f)

# 6. Query-time: encode user prompt into a dummy graph then retrieve
def query_top_k(model, prompt, k=3, index_path="method_index.faiss", map_path="method_ids.pkl"):
    # for demo: represent prompt as single‑node graph with feature = length
    node_feat = torch.tensor([[len(prompt)]], dtype=torch.float)
    data = Data(x=node_feat, edge_index=torch.empty((2,0),dtype=torch.long), batch=torch.zeros(1,dtype=torch.long))
    model.eval()
    with torch.no_grad():
        q_emb = model(data).cpu().numpy()
    faiss.normalize_L2(q_emb)
    index = faiss.read_index(index_path)
    D, I = index.search(q_emb, k)
    with open(map_path,"rb") as f:
        ids = pickle.load(f)
    return [(ids[i], float(D[0,j])) for j,i in enumerate(I[0])]

# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="methods.csv")
    parser.add_argument("--build", action="store_true", help="train+build index")
    parser.add_argument("--query", help="prompt to query top‑3")
    args = parser.parse_args()

    df      = load_methods(args.csv)
    graphs  = build_graphs(df)
    model   = train_gnn(graphs)

    if args.build:
        build_index(model, graphs)

    if args.query:
        top3 = query_top_k(model, args.query)
        print("Top‑3 methods:", top3)
