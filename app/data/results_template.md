results_template = """
# 📊 Evaluation Results

This document summarizes the results of empirical evaluations conducted on the document analysis system.

---

## 1. Embedding Comparison

| Embedding Method | Silhouette Score | Davies-Bouldin Index |
|------------------|------------------|-----------------------|
| SPECTER          | (to fill)        | (to fill)             |
| SBERT            | (to fill)        | (to fill)             |
| TF-IDF           | (to fill)        | (to fill)             |

📝 *These results evaluate how well different embedding strategies enable meaningful clustering.*

---

## 2. Query Retrieval Evaluation

| Query Term              | Precision@5 | Recall@5 |
|-------------------------|-------------|----------|
| graph neural networks   | (to fill)   | (to fill)|
| protein folding         | (to fill)   | (to fill)|

📝 *These values indicate the accuracy of semantic document retrieval using cosine similarity.*

---

## 3. Clustering Algorithm Comparison

| Clustering Algorithm | Silhouette Score |
|----------------------|------------------|
| KMeans               | (to fill)        |
| DBSCAN               | (to fill)        |
| Agglomerative        | (to fill)        |

📝 *Higher scores suggest tighter, more separated clusters.*

---

## 4. Summary Quality (Human Evaluation)

| Paper Title          | Summary Rating (1-5) | Notes |
|----------------------|----------------------|-------|
| (paper title)        | (to fill)            |       |
| (paper title)        | (to fill)            |       |

📝 *These summaries were reviewed for coherence, coverage, and clarity.*

---

## Notes

- Silhouette score ∈ [-1, 1]; closer to 1 is better.
- DBI (Davies-Bouldin Index): lower is better.
- Retrieval metrics require relevance ground truth (manual or heuristic).
- Summary ratings should reflect user feedback if available.

---

✅ *All evaluations were generated using the scripts in the `evaluation_scripts/` directory.*
"""

with open("/mnt/data/evaluation_scripts/results_template.md", "w") as f:
    f.write(results_template)

"/mnt/data/evaluation_scripts/results_template.md"
