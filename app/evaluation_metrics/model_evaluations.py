import os
import time
import json
import fitz
import torch
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances, pairwise
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import cdist

# --- CONFIG ---
FOLDER_PATH = r"../../app/documents/pdfs_2025-03-30_21-44-14"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5

MODELS = {
    "SBERT": "all-MiniLM-L6-v2",
    "SBERT_L12": "sentence-transformers/all-MiniLM-L12-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "RoBERTa": "sentence-transformers/nli-roberta-base-v2",
    "GTR": "sentence-transformers/gtr-t5-base",
    "SPECTER": "allenai/specter"
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
    start_cpu = psutil.cpu_percent(interval=None)
    cpu_samples = []
    start_time = time.time()
    process = psutil.Process(os.getpid())

    result = None
    while True:
        start = time.time()
        if result is None:
            result = func(*args, **kwargs)
        cpu_samples.append(process.cpu_percent(interval=0.1))
        if time.time() - start_time > 1.0:
            break

    end_time = time.time()
    return result, end_time - start_time, max(cpu_samples), np.mean(cpu_samples)


def top_k_neighbor_consistency(embeddings, labels, k=5):
    similarities = pairwise.cosine_similarity(embeddings)
    np.fill_diagonal(similarities, -np.inf)
    consistent = []

    for i, row in enumerate(similarities):
        top_k = np.argsort(row)[-k:]
        same_cluster = [1 if labels[i] == labels[j] else 0 for j in top_k]
        consistent.append(np.mean(same_cluster))

    return np.mean(consistent)


def evaluate_embeddings(embeddings, filenames, model_name):
    results = {}

    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Basic clustering metrics
    silhouette = silhouette_score(embeddings, labels)
    dbi = davies_bouldin_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)

    # Cosine similarity matrix
    cosine_sim = pairwise.cosine_similarity(embeddings)
    cosine_var = np.var(cosine_sim)

    # Intra-cluster Distance
    intra_dists = []
    for cluster_id in np.unique(labels):
        cluster_points = embeddings[labels == cluster_id]
        if len(cluster_points) > 1:
            distances = pairwise_distances(cluster_points)
            intra = np.mean(distances)
            intra_dists.append(intra)
    intra_cluster_distance = np.mean(intra_dists)

    # Inter-cluster Distance
    cluster_centers = kmeans.cluster_centers_
    inter_dists = cdist(cluster_centers, cluster_centers)
    inter_cluster_distance = np.mean(inter_dists[np.triu_indices_from(inter_dists, k=1)])
    intra_inter_ratio = intra_cluster_distance / inter_cluster_distance

    # Save cosine matrix
    pd.DataFrame(cosine_sim, index=filenames, columns=filenames).to_csv(
        f"evaluation_data/model_evaluations/{model_name}_similarity_matrix.csv")

    # Silhouette histogram
    plt.hist(silhouette if isinstance(silhouette, np.ndarray) else [silhouette], bins=20)
    plt.title(f"{model_name} Silhouette Score Distribution")
    plt.xlabel("Silhouette Score")
    plt.ylabel("Documents")
    plt.savefig(f"evaluation_data/model_evaluations/{model_name}_silhouette_hist.png")
    plt.close()

    # Top-K cluster consistency
    topk_consistency = top_k_neighbor_consistency(embeddings, labels, k=5)

    results.update({
        "Silhouette": silhouette,
        "Davies-Bouldin": dbi,
        "Calinski-Harabasz": calinski,
        "Cosine Mean": np.mean(cosine_sim),
        "Cosine Median": np.median(cosine_sim),
        "Cosine Min": np.min(cosine_sim),
        "Cosine Max": np.max(cosine_sim),
        "Cosine Variance": cosine_var,
        "Top-K Cluster Consistency": topk_consistency,
        "Embedding Dim": embeddings.shape[1],
        "Intra-cluster Distance": intra_cluster_distance,
        "Inter-cluster Distance": inter_cluster_distance,
        "Intra/Inter Ratio": intra_inter_ratio
    })

    return results


def main():
    print("Loading PDF documents...")
    texts, filenames = load_texts_from_folder(FOLDER_PATH)
    print(f"Loaded {len(texts)} documents.")

    summary = []

    for name, model_id in MODELS.items():
        print(f"\\n--- Evaluating {name} ---")
        if "specter" in model_id:
            embeddings, time_sec, cpu_peak, cpu_avg = track_performance(encode_with_huggingface, model_id, texts)
        else:
            embeddings, time_sec, cpu_peak, cpu_avg = track_performance(encode_with_sbert, model_id, texts)

        print(f"{name} Time: {time_sec:.2f}s | CPU Avg: {cpu_avg:.2f}% | CPU Peak: {cpu_peak:.2f}%")
        metrics = evaluate_embeddings(embeddings, filenames, name)
        metrics.update({
            "Model": name,
            "Time (s)": time_sec,
            "CPU Avg (%)": cpu_avg,
            "CPU Peak (%)": cpu_peak
        })
        summary.append(metrics)

    df_summary = pd.DataFrame(summary)
    df_summary.set_index("Model", inplace=True)
    df_summary.to_csv("evaluation_data/model_evaluations/model_comparison_summary.csv")
    print("\\nSaved model_comparison_summary.csv")


if __name__ == "__main__":
    main()
