import networkx as nx
import json
import numpy as np

def load_graph(path):
    G = nx.read_gexf(path)
    return G

def compare_edges(G_lstm, G_cosine):
    edges_lstm = set(G_lstm.edges())
    edges_cosine = set(G_cosine.edges())

    intersection = edges_lstm & edges_cosine
    union = edges_lstm | edges_cosine

    gained = edges_lstm - edges_cosine
    lost = edges_cosine - edges_lstm
    jaccard = len(intersection) / len(union) if union else 0

    return {
        "lstm_edges": len(edges_lstm),
        "cosine_edges": len(edges_cosine),
        "overlap": len(intersection),
        "jaccard": jaccard,
        "gained": list(gained),
        "lost": list(lost),
    }

def compute_graph_metrics(G):
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "components": nx.number_connected_components(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "avg_clustering": nx.average_clustering(G)
    }

def degree_shift(G1, G2):
    d1 = dict(G1.degree())
    d2 = dict(G2.degree())
    all_nodes = set(d1) | set(d2)
    return {
        node: d2.get(node, 0) - d1.get(node, 0)
        for node in all_nodes
    }

if __name__ == "__main__":
    # Path to your two graphs
    PATH_LSTM = "graph_lstm.gexf"
    PATH_COSINE = "graph_cosine.gexf"

    print("Loading graphs...")
    G_lstm = load_graph(PATH_LSTM)
    G_cosine = load_graph(PATH_COSINE)

    print("\n📊 Comparing Edges...")
    edge_stats = compare_edges(G_lstm, G_cosine)
    for k, v in edge_stats.items():
        if isinstance(v, list):
            print(f"{k}: {len(v)}")
        else:
            print(f"{k}: {v}")

    print("\n📐 Graph Metrics:")
    metrics_lstm = compute_graph_metrics(G_lstm)
    metrics_cosine = compute_graph_metrics(G_cosine)

    print("\nLSTM Graph:")
    for k, v in metrics_lstm.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    print("\nCosine Graph:")
    for k, v in metrics_cosine.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n📈 Degree Shift (Top 10 nodes):")
    delta_deg = degree_shift(G_cosine, G_lstm)
    top_shift = sorted(delta_deg.items(), key=lambda x: -abs(x[1]))[:10]
    for node, delta in top_shift:
        print(f"{node[:50]}...  Δ degree: {delta}")
