import os
import fitz
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# CONFIG
PDF_FOLDER = r"C:\\Users\\Polymer\\PycharmProjects\\DocumentComparison\\app\\documents\\pdfs_2025-03-30_21-44-14"
MODEL_ID = "allenai/specter"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_CSV = "evaluation_data/reducer_evaluations/dimensionality_reduction_evaluation.csv"


# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


def run_autoencoder(X, epochs=100, batch_size=8, lr=0.001):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model = Autoencoder(X.shape[1])
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        reduced = model.encode(X_tensor).numpy()
    return reduced


def extract_text_from_pdf(pdf_path, max_pages=2):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:max_pages]:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Failed to extract {pdf_path}: {e}")
        return ""


def load_pdfs(folder_path):
    docs = []
    filenames = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".pdf"):
            path = os.path.join(folder_path, fname)
            content = extract_text_from_pdf(path)
            if content:
                docs.append(content)
                filenames.append(fname)
    return docs, filenames


def encode_with_specter(docs):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    embeddings = []

    with torch.no_grad():
        for doc in docs:
            inputs = tokenizer(doc, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
            output = model(**inputs)
            cls_emb = output.last_hidden_state[:, 0, :].squeeze()
            embeddings.append(cls_emb.cpu().numpy())

    return np.vstack(embeddings)


def reduce_and_evaluate(X, model_name="SPECTER"):
    reducers = {
        "PCA": PCA(n_components=2),
        "tSNE": TSNE(n_components=2, perplexity=min(30, (len(X) - 1) // 3), random_state=42),
        "UMAP": umap.UMAP(n_components=2, random_state=42),
        "Autoencoder": "custom"
    }

    results = []

    for name, reducer in reducers.items():
        print(f"Reducing with {name}...")
        if name == "Autoencoder":
            X_2d = run_autoencoder(X)
        else:
            X_2d = reducer.fit_transform(X)

        clustering = AgglomerativeClustering(n_clusters=5)
        labels = clustering.fit_predict(X_2d)

        try:
            silhouette = silhouette_score(X_2d, labels)
        except:
            silhouette = np.nan
        try:
            db = davies_bouldin_score(X_2d, labels)
        except:
            db = np.nan
        try:
            ch = calinski_harabasz_score(X_2d, labels)
        except:
            ch = np.nan

        results.append({
            "Model": model_name,
            "Reducer": name,
            "Silhouette_2D": silhouette,
            "Davies-Bouldin_2D": db,
            "Calinski-Harabasz_2D": ch
        })

        plt.figure(figsize=(6, 5))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=10)
        plt.title(f"{model_name} - {name}")
        plt.savefig(f"evaluation_data/reducer_evaluations/{model_name}_{name}_2D_plot.png")
        plt.close()

    return results


def main():
    print("Loading documents...")
    docs, filenames = load_pdfs(PDF_FOLDER)
    print(f"Loaded {len(docs)} PDFs.")

    print("Encoding with SPECTER...")
    embeddings = encode_with_specter(docs)
    np.save("specter_embeddings.npy", embeddings)

    print("Reducing and evaluating...")
    X = StandardScaler().fit_transform(embeddings)
    results = reduce_and_evaluate(X)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dimensionality reduction metrics to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
