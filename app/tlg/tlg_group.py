import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_documents_by_group_and_year(json_path, group_field="category"):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    grouped = defaultdict(lambda: defaultdict(list))  # grouped[group][year] = [vectors]
    for doc in docs:
        year = str(doc.get("year"))
        group = doc.get(group_field)
        vec = doc.get("specter_embedding")
        if not (year and group and vec):
            continue
        grouped[group][year].append(np.array(vec))
    return grouped

def compute_centroid(vectors):
    return np.mean(np.vstack(vectors), axis=0)

def compute_group_drift(embedding_data):
    drift_entries = []
    years = sorted(int(y) for y in embedding_data.keys())

    for i in range(len(years) - 1):
        y1, y2 = str(years[i]), str(years[i + 1])
        groups1, groups2 = embedding_data[y1], embedding_data[y2]

        common_groups = set(groups1.keys()).intersection(groups2.keys())

        for group in common_groups:
            vecs1 = np.array(groups1[group])
            vecs2 = np.array(groups2[group])

            if len(vecs1) == 0 or len(vecs2) == 0:
                continue

            avg1 = np.mean(vecs1, axis=0).reshape(1, -1)
            avg2 = np.mean(vecs2, axis=0).reshape(1, -1)

            sim = float(cosine_similarity(avg1, avg2)[0][0])
            drift_entries.append({
                "group": group,
                "from": y1,
                "to": y2,
                "similarity": sim,
                "drift": sim - 1.0
            })

    return drift_entries

def save_drift(results, path="group_tlg_scores.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved {len(results)} group drift scores to {path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compute_group_tlg.py fused_documents.json [field=category|cluster]")
    else:
        group_field = sys.argv[2] if len(sys.argv) > 2 else "category"
        data = load_documents_by_group_and_year(sys.argv[1], group_field=group_field)
        drift_scores = compute_group_drift(data)
        save_drift(drift_scores)