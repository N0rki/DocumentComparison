import numpy as np
import json
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import joblib

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return docs

def cosine(a, b):
    return float(cosine_similarity([a], [b])[0][0])

def sample_pairs(docs, num_positive=1000, num_negative=1000):
    pairs = []
    labels = []

    indices = list(range(len(docs)))

    # Positive pairs: close in cluster or same authors
    for _ in range(num_positive):
        d1 = random.choice(docs)
        candidates = [d for d in docs if d != d1 and (
            d.get("authors") == d1.get("authors") or d.get("cluster") == d1.get("cluster"))]
        if not candidates:
            continue
        d2 = random.choice(candidates)
        sim1 = cosine(d1["specter_embedding"], d2["specter_embedding"])
        sim2 = cosine(d1["sbert_embedding"], d2["sbert_embedding"])
        sim3 = cosine(d1["metadata_vector"], d2["metadata_vector"])
        pairs.append([sim1, sim2, sim3])
        labels.append(1)

    # Negative pairs: randomly selected
    for _ in range(num_negative):
        d1, d2 = random.sample(docs, 2)
        sim1 = cosine(d1["specter_embedding"], d2["specter_embedding"])
        sim2 = cosine(d1["sbert_embedding"], d2["sbert_embedding"])
        sim3 = cosine(d1["metadata_vector"], d2["metadata_vector"])
        pairs.append([sim1, sim2, sim3])
        labels.append(0)

    return np.array(pairs), np.array(labels)

def train_mv_fusion_model(docs, save_path="mv_fusion_model.pkl"):
    X, y = sample_pairs(docs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"✅ Model trained with accuracy: {acc:.4f}")
    print("Learned α weights:", model.coef_[0])
    joblib.dump(model, save_path)
    print(f"📦 Saved to {save_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python learn_mv_fusion.py path/to/fused_documents.json")
    else:
        docs = load_data(sys.argv[1])
        train_mv_fusion_model(docs)