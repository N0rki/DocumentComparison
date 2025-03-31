
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from extract_data import process_pdf
from vectorization import vectorize_text_specter

PDF_DIR = os.path.join(BASE_DIR, 'documents', 'three_categories')

def load_pdf_texts(pdf_dir=PDF_DIR):
    texts, filenames = [], []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            try:
                path = os.path.join(pdf_dir, filename)
                result = process_pdf(path, filename)
                title = result.get("title", "").strip()
                abstract = result.get("abstract", "").strip()
                if title and abstract and "Nothing was found" not in title and "Nothing was found" not in abstract:
                    combined = f"{title} {abstract}"
                    texts.append(combined)
                    filenames.append(filename)
                else:
                    print(f"Skipped (missing/invalid title/abstract): {filename}")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return texts, filenames

def best_kmeans_clustering(embeddings, k_range=(2, 10)):
    best_k = None
    best_score = -1
    best_labels = None
    for k in range(k_range[0], k_range[1] + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            print(f"K={k}, Silhouette={score:.4f}")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception as e:
            print(f"Clustering failed for K={k}: {e}")
    return best_k, best_score, best_labels

def evaluate_embeddings(embeddings, name="Model"):
    k, silhouette, labels = best_kmeans_clustering(embeddings)
    dbi = davies_bouldin_score(embeddings, labels)
    print(f"[{name}] Best K: {k}, Silhouette: {silhouette:.4f}, DBI: {dbi:.4f}")

def specter2_encode(texts):
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoModel.from_pretrained("allenai/specter2_base")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
            cls_embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(cls_embedding.squeeze().cpu().numpy())
    return np.stack(embeddings)

def main():
    texts, filenames = load_pdf_texts()
    if not texts:
        print("No valid documents found.")
        return

    print("\n--- Evaluating SPECTER ---")
    specter_embs = [vectorize_text_specter(text) for text in texts]
    evaluate_embeddings(np.array(specter_embs), "SPECTER")

    print("\n--- Evaluating SPECTER2 ---")
    specter2_embs = specter2_encode(texts)
    evaluate_embeddings(np.array(specter2_embs), "SPECTER2")

    print("\n--- Evaluating SBERT ---")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    sbert_embs = sbert.encode(texts, show_progress_bar=True)
    evaluate_embeddings(np.array(sbert_embs), "SBERT")

    print("\n--- Evaluating TF-IDF ---")
    tfidf = TfidfVectorizer(max_features=1000)
    tfidf_embs = tfidf.fit_transform(texts).toarray()
    evaluate_embeddings(tfidf_embs, "TF-IDF")

if __name__ == "__main__":
    main()
