import os
import time
import fitz
import torch
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import hdbscan

# --- CONFIG ---
FOLDER_PATH = r"C:\\Users\\Polymer\\PycharmProjects\\DocumentComparison\\app\\documents\\pdfs_2025-03-31_03-06-34"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "SBERT": "all-MiniLM-L6-v2",
    "SBERT_L12": "sentence-transformers/all-MiniLM-L12-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "RoBERTa": "sentence-transformers/nli-roberta-base-v2",
    "GTR": "sentence-transformers/gtr-t5-base",
    "SPECTER": "allenai/specter"
}

CLUSTERERS = {
    "KMeans": lambda X: KMeans(n_clusters=5, random_state=42).fit_predict(X),
    "Agglomerative": lambda X: AgglomerativeClustering(n_clusters=5).fit_predict(X),
    "Spectral": lambda X: SpectralClustering(n_clusters=5, affinity="nearest_neighbors", assign_labels="kmeans").fit_predict(X),
    "HDBSCAN": lambda X: hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(X)
}

def extract_text_from_pdf(pdf_path, max_pages=2):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:max_pages]:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def load_texts_from_folder(folder_path):
    docs = []
    filenames = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pdf"):
            full_path = os.path.join(folder_path, fname)
            text = extract_text_from_pdf(full_path)
            if text:
                docs.append(text)
                filenames.append(fname)
    return docs, filenames

def encode_with_huggingface(model_id, docs):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(DEVICE)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for doc in docs:
            inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
            embeddings.append(cls_embedding.cpu().numpy())

    return np.vstack(embeddings)

def encode_with_sbert(model_id, docs):
    model = SentenceTransformer(model_id)
    return model.encode(docs, show_progress_bar=True)

def track_performance(func, *args, **kwargs):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    cpu_samples = []

    result = func(*args, **kwargs)

    for _ in range(10):
        cpu_samples.append(process.cpu_percent(interval=0.1))

    end_time = time.time()
    return result, end_time - start_time, max(cpu_samples), np.mean(cpu_samples)

def clustering_metrics(embeddings, labels):
    metrics = {}
    try:
        metrics["Silhouette"] = silhouette_score(embeddings, labels)
    except:
        metrics["Silhouette"] = np.nan
    try:
        metrics["Davies-Bouldin"] = davies_bouldin_score(embeddings, labels)
    except:
        metrics["Davies-Bouldin"] = np.nan
    try:
        metrics["Calinski-Harabasz"] = calinski_harabasz_score(embeddings, labels)
    except:
        metrics["Calinski-Harabasz"] = np.nan

    metrics["Num Clusters"] = len(set(labels)) - (1 if -1 in labels else 0)
    metrics["Num Outliers"] = sum(1 for x in labels if x == -1)

    return metrics

def evaluate_model_with_all_clusterers(embeddings, model_name):
    results = []
    for algo_name, cluster_func in CLUSTERERS.items():
        labels = cluster_func(embeddings)
        metrics = clustering_metrics(embeddings, labels)
        metrics["Model"] = model_name
        metrics["Algorithm"] = algo_name
        results.append(metrics)
    return results

def main():
    texts, filenames = load_texts_from_folder(FOLDER_PATH)
    print(f"Loaded {len(texts)} documents.")

    all_results = []

    for model_name, model_id in MODELS.items():
        print(f"\\nEncoding using {model_name}...")
        if "specter" in model_id:
            embeddings, time_sec, cpu_peak, cpu_avg = track_performance(encode_with_huggingface, model_id, texts)
        else:
            embeddings, time_sec, cpu_peak, cpu_avg = track_performance(encode_with_sbert, model_id, texts)

        print(f"{model_name}: Time = {time_sec:.2f}s, CPU Avg = {cpu_avg:.2f}%, CPU Peak = {cpu_peak:.2f}%")

        cluster_results = evaluate_model_with_all_clusterers(embeddings, model_name)
        for res in cluster_results:
            res["Time (s)"] = time_sec
            res["CPU Avg (%)"] = cpu_avg
            res["CPU Peak (%)"] = cpu_peak
        all_results.extend(cluster_results)

    df = pd.DataFrame(all_results)
    df.to_csv("evaluation_data/clustering_evaluations/clustering_algorithm_comparison.csv", index=False)
    print("\\nSaved clustering_algorithm_comparison.csv")

if __name__ == "__main__":
    main()