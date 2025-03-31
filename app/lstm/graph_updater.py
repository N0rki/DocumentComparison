import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from model_lstm import LinkPredictorLSTM
import chromadb

def predict_link_score(emb1, emb2, model, device="cpu"):
    diff = np.abs(emb1 - emb2)
    x = np.concatenate([emb1, emb2, diff])
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(x_tensor).item()
    return score

def build_dynamic_graph(collection_name, model_path="link_predictor_lstm.pt", threshold=0.8):
    # Connect to Chroma
    client = chromadb.HttpClient(host="localhost", port=8000)
    collection = client.get_collection(collection_name)
    records = collection.get(include=["embeddings", "metadatas"])

    embeddings = np.array(records["embeddings"])
    metadatas = records["metadatas"]
    ids = records["ids"]

    # Load model
    model = LinkPredictorLSTM()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Build graph
    G = nx.Graph()

    # Add nodes
    for i, meta in enumerate(metadatas):
        G.add_node(ids[i], title=meta.get("title", f"Doc {i}"), metadata=meta)

    # Add predicted edges
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = predict_link_score(embeddings[i], embeddings[j], model)
            if score >= threshold:
                G.add_edge(ids[i], ids[j], weight=score)

    print(f"✅ Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G
