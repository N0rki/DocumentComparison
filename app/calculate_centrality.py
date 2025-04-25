import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def calculate_centrality(embedding_data):
    centrality_results = defaultdict(dict)

    for year, clusters in embedding_data.items():
        for cluster, embeddings in clusters.items():
            embeddings_np = np.array(embeddings)
            if len(embeddings_np) < 2:
                continue

            similarity_matrix = cosine_similarity(embeddings_np)
            avg_similarity = similarity_matrix.mean(axis=1)

            centrality_results[year][cluster] = avg_similarity.tolist()

    return centrality_results

if __name__ == "__main__":
    with open("chroma_viewer/app/group_embeddings/group_embeddings_by_year.json", "r", encoding="utf-8") as f:
        embedding_data = json.load(f)

    centrality = calculate_centrality(embedding_data)

    with open("group_embeddings/centrality_by_year.json", "w", encoding="utf-8") as f:
        json.dump(centrality, f, indent=2)
