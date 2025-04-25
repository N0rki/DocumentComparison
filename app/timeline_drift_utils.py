import json
import numpy as np

def load_group_drift(path="app/group_embeddings/group_embeddings_by_year.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("⚠️ Could not load group drift scores:", e)
        return []

def add_group_drift_to_timeline(timeline_net, drift_data, group_type="category"):
    for entry in drift_data:
        group = entry["group"]
        y1 = entry["from"]
        y2 = entry["to"]
        drift = entry["drift"]
        sim = entry["similarity"]
        node1 = f"{group}_{y1}"
        node2 = f"{group}_{y2}"

        color = "green" if drift >= 0 else "red"
        label = f"Δ: {drift:.2f}"

        timeline_net.add_node(node1, label=f"{group} ({y1})", shape="ellipse", group=group)
        timeline_net.add_node(node2, label=f"{group} ({y2})", shape="ellipse", group=group)
        timeline_net.add_edge(
            node1,
            node2,
            title=label,
            value=abs(drift),
            color=color,
            arrows="to",
            font={"align": "middle"},
        )

from sklearn.metrics.pairwise import cosine_similarity

def compute_group_drift(embedding_data):
    drift_entries = []
    years = sorted(int(y) for y in embedding_data.keys())

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        groups1 = embedding_data[str(y1)]
        groups2 = embedding_data[str(y2)]

        common_groups = set(groups1.keys()) & set(groups2.keys())

        for group in common_groups:
            emb1 = np.array(groups1[group])
            emb2 = np.array(groups2[group])

            if len(emb1) == 0 or len(emb2) == 0:
                continue

            avg1 = np.mean(emb1, axis=0).reshape(1, -1)
            avg2 = np.mean(emb2, axis=0).reshape(1, -1)

            sim = cosine_similarity(avg1, avg2)[0][0]
            drift = sim - 1.0  # deviation from perfect similarity

            drift_entries.append({
                "group": group,
                "from": y1,
                "to": y2,
                "similarity": float(sim),
                "drift": float(drift)
            })

    return drift_entries