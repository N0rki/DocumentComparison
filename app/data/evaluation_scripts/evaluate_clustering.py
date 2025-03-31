
import sys
import os
import fitz  # PyMuPDF

# Path fix to allow local imports if needed
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

PDF_DIR = os.path.join(BASE_DIR, 'documents', 'three_categories')

def load_pdf_texts(pdf_dir=PDF_DIR):
    texts, filenames = [], []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            try:
                path = os.path.join(pdf_dir, filename)
                doc = fitz.open(path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                doc.close()
                texts.append(full_text.strip())
                filenames.append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return texts, filenames

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from vectorization import vectorize_text_specter

def evaluate(embeddings, labels, name):
    score = silhouette_score(embeddings, labels)
    print(f"{name} Silhouette Score: {score:.4f}")

def main():
    texts, _ = load_pdf_texts()
    embeddings = np.array([vectorize_text_specter(text) for text in texts])

    km = KMeans(n_clusters=5, random_state=42).fit_predict(embeddings)
    evaluate(embeddings, km, "KMeans")

    db = DBSCAN(eps=0.5, min_samples=5).fit_predict(embeddings)
    evaluate(embeddings, db, "DBSCAN")

    agg = AgglomerativeClustering(n_clusters=5).fit_predict(embeddings)
    evaluate(embeddings, agg, "Agglomerative")

if __name__ == "__main__":
    main()
