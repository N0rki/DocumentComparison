import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise
from sklearn.cluster import KMeans
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch

# Hardcoded PDF folder
FOLDER_PATH = r"/app/documents/pdfs_2025-03-31_03-06-34"

# Load text content
from extract_pdf import load_texts_from_pdfs_batched

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models to compare
MODELS = {
    "SBERT": "all-MiniLM-L6-v2",
    "SPECTER": "allenai/specter",
    "SPECTER2": "allenai/specter2"
}


# Hugging Face model-based encoding
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


# SBERT-style encoding
def encode_with_sbert(model_id, docs):
    model = SentenceTransformer(model_id)
    return model.encode(docs, show_progress_bar=True)


# Performance wrapper
def track_performance(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end_time - start_time, peak / 10 ** 6


# Core evaluation
def evaluate_embeddings(embeddings, filenames, model_name):
    results = {}

    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    results["Silhouette"] = silhouette_score(embeddings, labels)
    results["Davies-Bouldin"] = davies_bouldin_score(embeddings, labels)

    sim_matrix = pairwise.cosine_similarity(embeddings)
    sim_stats = {
        "Cosine Mean": np.mean(sim_matrix),
        "Cosine Median": np.median(sim_matrix),
        "Cosine Min": np.min(sim_matrix),
        "Cosine Max": np.max(sim_matrix),
    }
    results.update(sim_stats)

    # Contextual shift
    context_vector = np.random.normal(0, 0.01, size=embeddings.shape[1])
    embeddings_contextual = embeddings + context_vector
    sim_matrix_contextual = pairwise.cosine_similarity(embeddings_contextual)
    delta = sim_matrix_contextual - sim_matrix

    # Save matrices
    pd.DataFrame(sim_matrix, index=filenames, columns=filenames).to_csv(f"{model_name}_similarity_matrix.csv")
    pd.DataFrame(sim_matrix_contextual, index=filenames, columns=filenames).to_csv(
        f"{model_name}_similarity_matrix_contextual.csv")

    # Save heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(delta, cmap='coolwarm')
    plt.title(f"{model_name}: Contextual Similarity Δ")
    plt.colorbar()
    plt.savefig(f"{model_name}_similarity_delta_heatmap.png")
    plt.close()

    return results


def main():
    print("Loading PDF documents...")
    texts, filenames = load_texts_from_pdfs_batched(batch_size=8)
    print(f"Loaded {len(texts)} documents.")

    summary = []

    for name, model_id in MODELS.items():
        print(f"\n--- Evaluating {name} ---")
        if "specter" in model_id:
            embeddings, vec_time, mem = track_performance(encode_with_huggingface, model_id, texts)
        else:
            embeddings, vec_time, mem = track_performance(encode_with_sbert, model_id, texts)

        print(f"{name} Vectorization Time: {vec_time:.2f}s, Peak Memory: {mem:.2f}MB")
        metrics = evaluate_embeddings(embeddings, filenames, name)
        metrics["Model"] = name
        metrics["Time (s)"] = vec_time
        metrics["Memory (MB)"] = mem
        summary.append(metrics)

    df_summary = pd.DataFrame(summary)
    df_summary.set_index("Model", inplace=True)
    df_summary.to_csv("model_comparison_summary.csv")
    print("\nSaved model_comparison_summary.csv")


if __name__ == "__main__":
    main()
