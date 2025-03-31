import numpy as np
import random

def generate_training_pairs(chroma_collection, threshold=0.85, negative_ratio=1.0):
    """
    Generate positive and negative training pairs from ChromaDB collection.
    Positive = cosine_sim > threshold.
    Negative = random pairs with sim < threshold (approx).
    """
    records = chroma_collection.get(include=["embeddings", "metadatas", "documents"])
    embeddings = np.array(records["embeddings"])
    ids = records["ids"]

    pos_pairs, neg_pairs = [], []
    used = set()

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            id_pair = (ids[i], ids[j])
            if id_pair in used:
                continue
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            pair_data = {
                "x": np.concatenate([
                    embeddings[i], embeddings[j], np.abs(embeddings[i] - embeddings[j])
                ]),
                "y": 1 if sim >= threshold else 0
            }

            if sim >= threshold:
                pos_pairs.append(pair_data)
            elif len(neg_pairs) < len(pos_pairs) * negative_ratio:
                neg_pairs.append(pair_data)

            used.add(id_pair)

    all_pairs = pos_pairs + neg_pairs
    random.shuffle(all_pairs)
    return all_pairs
