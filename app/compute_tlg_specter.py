import json
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from itertools import combinations

def load_documents_by_year(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    timeline = defaultdict(dict)
    for doc in docs:
        year = str(doc.get("year"))
        doc_id = doc.get("id") or doc.get("title")
        if not year or "specter_embedding" not in doc:
            continue
        timeline[doc_id][year] = np.array(doc["specter_embedding"])

    return timeline

def compute_tlg(vecs1, vecs2):
    common_years = sorted(set(vecs1.keys()) & set(vecs2.keys()))
    if len(common_years) < 3:
        return None

    sims = []
    for year in common_years:
        sims.append(cosine_similarity([vecs1[year]], [vecs2[year]])[0][0])

    gradients = np.diff(sims)
    return gradients[-1]

def compute_all_tlgs(timeline):
    doc_ids = list(timeline.keys())
    results = []

    for id1, id2 in tqdm(combinations(doc_ids, 2), total=(len(doc_ids)*(len(doc_ids)-1))//2):
        vecs1 = timeline[id1]
        vecs2 = timeline[id2]
        tlg = compute_tlg(vecs1, vecs2)
        if tlg is not None:
            results.append({"doc1": id1, "doc2": id2, "tlg": tlg})
    return results

def save_tlg_scores(tlg_scores, output="tlg_scores.json"):
    with open(output, "w", encoding="utf-8") as f:
        json.dump(tlg_scores, f, indent=2)
    print(f"✅ Saved {len(tlg_scores)} TLG scores to {output}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compute_tlg_specter_only.py path/to/fused_documents.json")
    else:
        timeline = load_documents_by_year(sys.argv[1])
        scores = compute_all_tlgs(timeline)
        save_tlg_scores(scores)