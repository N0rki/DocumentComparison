import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

def compute_cluster_centroids(group_embeddings_by_year):
    centroids = defaultdict(dict)  # {year: {cluster: centroid vector}}
    for year, clusters in group_embeddings_by_year.items():
        for cluster, embeddings in clusters.items():
            arr = np.array(embeddings)
            if len(arr) > 0:
                centroids[year][cluster] = np.mean(arr, axis=0)
    return centroids

def compute_influence_scores(docs_by_year, centroids_by_year, years_sorted):
    influence_scores = []
    for year_idx, year in enumerate(years_sorted[:-1]):
        year_docs = docs_by_year[year]
        doc_idx_map = 0
        for doc in year_docs:
            doc_embedding = np.array(doc["embedding"]).reshape(1, -1)
            score = 0
            for future_year in years_sorted[year_idx+1:]:
                max_sim = 0
                for cluster_center in centroids_by_year[future_year].values():
                    sim = cosine_similarity(doc_embedding, cluster_center.reshape(1, -1))[0][0]
                    max_sim = max(max_sim, sim)
                score += max_sim
            influence_scores.append({
                "title": doc["title"],
                "year": year,
                "cluster": doc["cluster"],
                "influence_score": score,
                "centrality": doc["centrality"]
            })
    return sorted(influence_scores, key=lambda x: x["influence_score"], reverse=True)

if __name__ == "__main__":
    with open("chroma_viewer/app/group_embeddings/group_embeddings_by_year.json", "r", encoding="utf-8") as f:
        group_embeddings = json.load(f)

    with open("group_embeddings/centrality_by_year.json", "r", encoding="utf-8") as f:
        centrality_scores = json.load(f)

    with open("fused_documents.json", "r", encoding="utf-8") as f:
        all_docs = json.load(f)

    docs_by_year = defaultdict(list)
    for doc in all_docs:
        year = str(doc["year"])
        if "specter_embedding" not in doc:
            continue

        doc_emb = np.array(doc["specter_embedding"]).reshape(1, -1)
        best_cluster = None
        best_similarity = -1

        if year in group_embeddings:
            for cluster, emb_list in group_embeddings[year].items():
                if not emb_list:
                    continue
                cluster_embs = np.array(emb_list)
                sims = cosine_similarity(doc_emb, cluster_embs)
                max_sim = np.max(sims)

                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_cluster = cluster

        if best_cluster is None:
            continue

        docs_by_year[year].append({
            "title": doc["title"],
            "year": year,
            "cluster": best_cluster,
            "embedding": doc["specter_embedding"],
            "centrality": None
        })

    for year in docs_by_year:
        if year in centrality_scores:
            for i, doc in enumerate(docs_by_year[year]):
                cluster = doc["cluster"]
                if cluster in centrality_scores[year]:
                    if i < len(centrality_scores[year][cluster]):
                        doc["centrality"] = centrality_scores[year][cluster][i]

    cluster_centroids = compute_cluster_centroids(group_embeddings)
    years_sorted = sorted(docs_by_year.keys())
    influence = compute_influence_scores(docs_by_year, cluster_centroids, years_sorted)

    with open("semantic_influence_scores.json", "w", encoding="utf-8") as f:
        json.dump(influence, f, indent=2, ensure_ascii=False)

    print("✅ Top influential documents saved to app/semantic_influence_scores.json")
