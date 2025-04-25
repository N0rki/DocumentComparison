import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import feedparser
from ast import literal_eval
import torch
import random
import matplotlib.pyplot as plt
import io
import zipfile
import json
import shap
from collections import defaultdict
import torch.nn as nn
import networkx as nx
from lstm.model_lstm import LinkPredictorLSTM
from scholarly import scholarly
import time
from sklearn.metrics import silhouette_score
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
from database_connection import connect_to_chromadb
from config.science_categories import PREDEFINED_CATEGORIES
from vectorization import vectorize_text_specter
import matplotlib.colors as mcolors
import networkx as nx
from nltk.corpus import wordnet
from transformers import pipeline
from pyvis.network import Network
from spellchecker import SpellChecker
import re
from PyPDF2 import PdfReader
from summarizer import summarize_text
from summarizer import chunk_and_summarize
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean, cityblock
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from similarity_utils import compute_mv_similarity
from timeline_drift_utils import load_group_drift, add_group_drift_to_timeline, compute_group_drift
from utils.dim_keyword_mapper import compute_dim_keywords_from_abstracts
from semantic_explainer import generate_local_semantic_explanation


class DataLoader:
    @staticmethod
    def load_data():
        try:
            st.write("Attempting to connect to ChromaDB...")
            chroma_client, collection = connect_to_chromadb()

            st.write("Fetching data from collection...")
            results = collection.get(include=['embeddings', 'metadatas'])

            st.write(f"Number of documents found: {len(results['metadatas'])}")

            if len(results['embeddings']) > 0:
                st.write(f"Shape of first embedding: {np.array(results['embeddings'][0]).shape}")
            else:
                st.write("No embeddings found")

            if len(results["metadatas"]) == 0:
                st.warning("No documents found in the collection")
                return pd.DataFrame()

            metadata_dict = {
                "title": [],
                "authors": [],
                "year": [],
                "abstract": []
            }

            st.write("Processing metadata...")
            for metadata in results["metadatas"]:
                metadata_dict["title"].append(str(metadata.get("title", "Untitled")))
                metadata_dict["authors"].append(str(metadata.get("authors", "Unknown")))
                metadata_dict["year"].append(int(metadata.get("year", 2023)))
                metadata_dict["abstract"].append(str(metadata.get("abstract", "")))

            df = pd.DataFrame(metadata_dict)

            st.write("Processing embeddings...")
            processed_embeddings = []
            for i, embedding in enumerate(results["embeddings"]):
                try:
                    emb_array = np.array(embedding, dtype=np.float32).flatten()
                    processed_embeddings.append(emb_array)
                except Exception as e:
                    st.write(f"Error processing embedding {i}: {str(e)}")
                    st.write(f"Embedding type: {type(embedding)}")
                    st.write(f"Embedding value: {embedding}")
                    raise

            df["embedding"] = processed_embeddings

            st.write("Data processing completed successfully")
            return df

        except Exception as e:
            st.error(f"Error fetching data from ChromaDB: {str(e)}")
            st.write("Exception details:", e.__class__.__name__)
            import traceback
            st.write("Full traceback:", traceback.format_exc())
            return pd.DataFrame()


class DimensionalityReducer:
    @staticmethod
    def reduce_embeddings(embeddings, method, n_components):
        if method == "PCA":
            reducer = PCA(n_components=n_components)
        elif method == "t-SNE":
            reducer = TSNE(n_components=n_components, perplexity=3)
        elif method == "UMAP":
            reducer = UMAP(n_components=n_components)
        else:
            raise ValueError("Unsupported dimensionality reduction method")
        return reducer.fit_transform(embeddings)


class Clusterer:
    @staticmethod
    def cluster_documents(embeddings, algorithm, n_clusters=None):
        if algorithm == "K-Means":
            if n_clusters is None:
                n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
        elif algorithm == "DBSCAN":
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(embeddings)
        elif algorithm == "Hierarchical":
            if n_clusters is None:
                n_clusters = 5
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = hierarchical.fit_predict(embeddings)
        elif algorithm == "GMM":
            if n_clusters is None:
                n_clusters = 5
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = gmm.fit_predict(embeddings)
        else:
            raise ValueError("Unsupported clustering algorithm")

        return cluster_labels

    @staticmethod
    def assign_categories_to_clusters(cluster_labels, embeddings, predefined_categories):
        unique_clusters = np.unique(cluster_labels)
        cluster_to_category = {}

        if -1 in unique_clusters:
            cluster_to_category[-1] = "Noise"
            unique_clusters = unique_clusters[unique_clusters != -1]

        category_embeddings = {category: vectorize_text_specter(category)
                               for category in predefined_categories}

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]

            if len(cluster_embeddings) == 0:
                cluster_to_category[cluster_id] = f"Cluster {cluster_id}"
                continue

            avg_cluster_embedding = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            best_category = None
            max_similarity = -1

            for category, category_embedding in category_embeddings.items():
                category_embedding = np.array(category_embedding).reshape(1, -1)
                similarity = cosine_similarity(avg_cluster_embedding, category_embedding)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_category = category

            cluster_to_category[cluster_id] = best_category if max_similarity > 0.3 else f"Cluster {cluster_id}"

        return cluster_to_category


class Visualizer:
    @staticmethod
    def get_distinct_colors(palette_name, n_colors):
        if palette_name == "Rainbow":
            colors = list(mcolors.TABLEAU_COLORS.values())
        elif palette_name == "Nature":
            colors = list(mcolors.BASE_COLORS.values())
        elif palette_name == "Contrast":
            colors = list(mcolors.CSS4_COLORS.values())
        elif palette_name == "Bright":
            colors = list(mcolors.XKCD_COLORS.values())
        else:
            colors = list(mcolors.TABLEAU_COLORS.values())

        if n_colors > len(colors):
            colors = colors * (n_colors // len(colors)) + colors[:n_colors % len(colors)]

        return colors[:n_colors]

    @staticmethod
    def get_background_colors():
        return {
            "White": "white",
            "Light Gray": "#f5f5f5",
            "Dark Gray": "#2d2d2d",
            "Black": "black",
            "Navy": "#001f3f",
            "Forest": "#1a472a"
        }


    @staticmethod
    def create_network_graph(df, similarity_threshold=0.7, custom_colors=None, similarity_mode="cosine",
                             emb_weight=0.60, author_weight=0.15, cluster_weight=0.10, year_weight=0.10,
                             keyword_weight=0.05):
        G = nx.Graph()

        for idx, row in df.iterrows():
            if custom_colors:
                cluster_index = list(df["cluster"].unique()).index(row["cluster"])
                color = custom_colors[cluster_index % len(custom_colors)]
            else:
                color = "blue"

            G.add_node(
                idx,
                label=row["title"],
                title=f"Authors: {row['authors']}<br>Year: {row['year']}<br>Abstract: {row['abstract']}",
                group=row["cluster"],
                color=color
            )

        embeddings = np.array(df["embedding"].tolist())
        titles = df["title"].tolist()
        authors = df["authors"].tolist()
        clusters = df["cluster"].tolist()
        years = df["year"].tolist()
        abstracts = df["abstract"].tolist()

        def hybrid_similarity(i, j, emb_weight=0.60, author_weight=0.15, cluster_weight=0.10, year_weight=0.10,
                              keyword_weight=0.05):
            cosine_sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
            shared_authors = int(authors[i].strip().lower() == authors[j].strip().lower())
            cluster_match = int(clusters[i] == clusters[j])
            year_proximity = max(0, 1 - abs(years[i] - years[j]) / 10)
            keywords_i = set(abstracts[i].lower().split())
            keywords_j = set(abstracts[j].lower().split())
            keyword_overlap = len(keywords_i & keywords_j) / max(len(keywords_i | keywords_j), 1)

            return (
                    emb_weight * cosine_sim +
                    author_weight * shared_authors +
                    cluster_weight * cluster_match +
                    year_weight * year_proximity +
                    keyword_weight * keyword_overlap
            )

        def create_hybrid_similarity_graph(df, emb_weight, author_weight, cluster_weight, year_weight, keyword_weight,
                                           threshold=0.75):
            from networkx import Graph
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            G = Graph()
            embeddings = list(df['embedding'])
            abstracts = list(df['abstract'])
            authors = list(df['authors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x)))
            clusters = list(df['cluster'] if 'cluster' in df.columns else [None] * len(df))
            years = list(df['year'])

            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    score = hybrid_similarity(
                        i, j,
                        emb_weight, author_weight,
                        cluster_weight, year_weight,
                        keyword_weight
                    )
                    if score >= threshold:
                        G.add_edge(
                            df.iloc[i]["title"],
                            df.iloc[j]["title"],
                            weight=float(score),
                            title=f"Similarity: {score:.2f}"
                        )
            return G

        def compute_cs2_similarity(doc_idx, embeddings, num_negatives=10):
            emb1 = np.array(embeddings[doc_idx]).reshape(1, -1)
            all_indices = list(range(len(embeddings)))
            all_indices.remove(doc_idx)
            negative_indices = random.sample(all_indices, min(num_negatives, len(all_indices)))
            negative_sims = [cosine_similarity(emb1, np.array(embeddings[i]).reshape(1, -1))[0][0] for i in
                             negative_indices]
            mean_neg_sim = np.mean(negative_sims)

            cs2_scores = []
            for j in range(len(embeddings)):
                if j == doc_idx:
                    cs2_scores.append(0)
                else:
                    sim = cosine_similarity(emb1, np.array(embeddings[j]).reshape(1, -1))[0][0]
                    cs2_scores.append(float(sim - mean_neg_sim))
            return cs2_scores

        cs2_matrix = None
        if similarity_mode == "cs2sim":
            cs2_matrix = [compute_cs2_similarity(i, embeddings) for i in range(len(embeddings))]

            flat_vals = [score for row in cs2_matrix for score in row]
            min_val, max_val = min(flat_vals), max(flat_vals)
            if max_val != min_val:
                cs2_matrix = [[(s - min_val) / (max_val - min_val) for s in row] for row in cs2_matrix]

        edge_count = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if similarity_mode == "hybrid":
                    similarity = hybrid_similarity(i, j)
                elif similarity_mode == "cs2sim":
                    similarity = cs2_matrix[i][j] if cs2_matrix else 0
                elif similarity_mode == "mvfusion":
                    sim = compute_mv_similarity(df.iloc[i], df.iloc[j])
                    similarity = max(0.0, min(1.0, sim))
                else:
                    similarity = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]

                if similarity > similarity_threshold:
                    G.add_edge(i, j, weight=float(similarity))
                    edge_count += 1

        net = Network(height="600px", width="100%", notebook=False, directed=False)

        net.from_nx(G)

        net.set_options(json.dumps({
            "physics": {"barnesHut": {"avoidOverlap": 0.02}, "minVelocity": 0.75}
        }))
        net.note = f"Total edges: {edge_count} | Similarity method: {similarity_mode}"
        return net, G

    @staticmethod
    def create_heatmap(df, similarity_threshold=0.7):
        embeddings = np.array(df["embedding"].tolist())
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix[similarity_matrix < similarity_threshold] = 0

        truncated_titles = [title[:40] + "..." if len(title) > 40 else title for title in df["title"].tolist()]

        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=truncated_titles,
            y=truncated_titles,
            colorscale="Viridis",
            colorbar=dict(title="Similarity"),
        ))

        fig.update_layout(
            title={
                "text": "Document Similarity Heatmap",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 20},
            },
            xaxis_title="Documents",
            yaxis_title="Documents",
            margin=dict(l=50, r=50, t=80, b=50),
        )
        return fig

    @staticmethod
    def create_parallel_coordinates(df, color_scheme="Viridis"):
        if "x" not in df.columns or "y" not in df.columns:
            st.warning("Please perform dimensionality reduction first.")
            return None

        df = df.copy()

        unique_clusters = df["cluster"].unique()
        cluster_name_to_id = {name: i for i, name in enumerate(unique_clusters)}
        cluster_id_to_name = {i: name for name, i in cluster_name_to_id.items()}
        df["cluster_numeric"] = df["cluster"].map(cluster_name_to_id)

        df["title_index"] = range(len(df))
        title_index_to_title = dict(zip(df["title_index"], df["title"]))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df["cluster_numeric"],
                colorscale=color_scheme,
                showscale=True,
                cmin=df["cluster_numeric"].min(),
                cmax=df["cluster_numeric"].max()
            ),
            dimensions=[
                dict(
                    label="Title",
                    values=df["title_index"],
                    tickvals=df["title_index"].tolist(),
                    ticktext=[t[:50] for t in df["title"].tolist()],
                    range=[df["title_index"].min(), df["title_index"].max()]
                ),
                dict(label="X", values=df["x"]),
                dict(label="Y", values=df["y"]),
                dict(
                    label="Cluster",
                    values=df["cluster_numeric"],
                    tickvals=list(cluster_id_to_name.keys()),
                    ticktext=list(cluster_id_to_name.values())
                )
            ]
        ))

        fig.update_layout(
            title="Parallel Coordinates: Document Titles → Cluster Path",
            margin=dict(l=300, r=50, t=50, b=50)
        )

        return fig

    @staticmethod
    def create_sankey_diagram(df):
        cluster_counts = df["cluster"].value_counts().reset_index()
        cluster_counts.columns = ["cluster", "count"]

        cluster_mapping = {cluster: i for i, cluster in enumerate(cluster_counts["cluster"])}

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=cluster_counts["cluster"].tolist(),
                color="blue",
            ),
            link=dict(
                source=[0] * len(cluster_counts),
                target=[cluster_mapping[cluster] for cluster in cluster_counts["cluster"]],
                value=cluster_counts["count"].tolist(),
            ),
        )])

        fig.update_layout(
            title="Document Flow Between Clusters",
            font=dict(size=12),
        )
        return fig


class ExternalAPIs:
    @staticmethod
    def fetch_pubmed_articles(query, max_results=10):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            article_ids = data.get("esearchresult", {}).get("idlist", [])
            return article_ids
        else:
            st.error(f"Failed to fetch data from PubMed: {response.status_code}")
            return []

    @staticmethod
    def fetch_pubmed_article_details(article_id):
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": article_id,
            "retmode": "json"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to fetch article details: {response.status_code}")
            return None

    @staticmethod
    def fetch_arxiv_articles(query, max_results=50, year_range=None):
        base_url = "http://export.arxiv.org/api/query"

        terms = query.split()

        title_query = " OR ".join([f'ti:"{term}"' for term in terms])
        abstract_query = " OR ".join([f'abs:"{term}"' for term in terms])
        combined_query = f"({title_query}) OR ({abstract_query})"

        params = {
            "search_query": combined_query,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            feed = feedparser.parse(response.content)
            articles = []
            for entry in feed.entries:
                published_date = entry.published
                published_year = int(published_date[:4])

                if year_range and (published_year < year_range[0] or published_year > year_range[1]):
                    continue

                title_matches = any(term.lower() in entry.title.lower() for term in terms)
                abstract_matches = any(term.lower() in entry.summary.lower() for term in terms)

                if title_matches or abstract_matches:
                    article = {
                        "title": entry.title,
                        "authors": [author.name for author in entry.authors],
                        "summary": entry.summary,
                        "published": published_date,
                        "link": entry.link,
                        "year": published_year
                    }
                    articles.append(article)

            return articles
        else:
            st.error(f"Failed to fetch data from arXiv: {response.status_code}")
            return []

    @staticmethod
    def fetch_citation_count(title):
        try:
            search_query = scholarly.search_pubs(title)
            publication = next(search_query)
            return publication.get("num_citations", 0)
        except Exception as e:
            st.error(f"Failed to fetch citation count: {str(e)}")
            return 0


class SemanticSearch:
    @staticmethod
    def semantic_search(df, query, top_k=5, similarity_threshold=0.5):
        filtered_df = df.dropna(subset=["abstract", "authors", "title"])

        if filtered_df.empty:
            st.warning("No valid documents found (missing abstract, authors, or title).")
            return pd.DataFrame()

        query_embedding = vectorize_text_specter(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        document_embeddings = np.array(filtered_df["embedding"].tolist())

        similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

        filtered_df["similarity"] = similarities

        thresholded_df = filtered_df[filtered_df["similarity"] >= similarity_threshold]

        if thresholded_df.empty:
            st.warning(f"No documents found with similarity >= {similarity_threshold}.")
            return pd.DataFrame()

        df_sorted = thresholded_df.sort_values(by="similarity", ascending=False)

        return df_sorted.head(top_k)


@st.cache_resource
def load_link_model(model_path="lstm/link_predictor_lstm.pt"):
    model = LinkPredictorLSTM()
    abs_path = os.path.join(os.path.dirname(__file__), model_path)
    model.load_state_dict(torch.load(abs_path, map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_sdaf_data():
    with open("../group_embeddings/centrality_by_year.json", "r", encoding="utf-8") as f:
        historical = json.load(f)
    with open("../sdaf_forecasts/multi_year_forecasts.json", "r", encoding="utf-8") as f:
        forecasts = json.load(f)
    return historical, forecasts

def average_historical(historical):
    grouped = defaultdict(dict)
    for year, topics in historical.items():
        for topic, vals in topics.items():
            grouped[topic][int(year)] = float(np.mean(vals))
    return grouped

def plot_sdaf_curve(topic, hist_data, forecast_data):
    hist_years = sorted(hist_data[topic].keys())
    hist_vals = [hist_data[topic][y] for y in hist_years]

    forecast_years = sorted(map(int, forecast_data[topic].keys()))
    forecast_vals = [forecast_data[topic][str(y)] for y in forecast_years]

    plt.figure(figsize=(8, 4))
    plt.plot(hist_years, hist_vals, marker='o', label='Historical', color='C0')
    plt.plot(forecast_years, forecast_vals, marker='x', linestyle='--', label='Forecast', color='C1')
    plt.title(f"SDAF: {topic}")
    plt.xlabel("Year")
    plt.ylabel("Centrality")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()

def show_sdaf_section():
    st.markdown("## Semantic Drift-Aware Forecasting (SDAF)")

    historical_raw, forecast_raw = load_sdaf_data()
    hist_data = average_historical(historical_raw)

    common_topics = sorted(set(hist_data) & set(forecast_raw))
    selected = st.selectbox("Select a topic:", common_topics)

    if selected:
        plot_sdaf_curve(selected, hist_data, forecast_raw)

def predict_link_score(emb1, emb2, model):
    diff = np.abs(emb1 - emb2)
    x = np.concatenate([emb1, emb2, diff])
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = model(x_tensor).item()
    return score


def build_dynamic_graph(df, model, threshold=0.8):
    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(row["title"], metadata=row)

    embeddings = df["embedding"].tolist()
    titles = df["title"].tolist()

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = predict_link_score(np.array(embeddings[i]), np.array(embeddings[j]), model)
            if score >= threshold:
                G.add_edge(titles[i], titles[j], weight=score)

    return G


def extract_keywords(text, top_n=5):
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['the', 'and', 'for', 'with', 'that', 'from', 'this', 'using', 'are', 'such', 'their', 'have'])
    words = [w for w in words if w not in stopwords and len(w) > 3]
    freq = defaultdict(int)
    for word in words:
        freq[word] += 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]


def build_knowledge_graph(df, use_lstm_links=True, threshold=0.8, model=None):
    G = nx.MultiDiGraph()

    for _, row in df.iterrows():
        doc_id = row["title"]
        G.add_node(doc_id, type="document", metadata=row)

        authors = [a.strip() for a in row["authors"].split(",")]
        for author in authors:
            G.add_node(author, type="author")
            G.add_edge(doc_id, author, relation="authored_by")

        cluster = row["cluster"]
        G.add_node(cluster, type="cluster")
        G.add_edge(doc_id, cluster, relation="in_cluster")

        keywords = extract_keywords(row["abstract"])
        for kw in keywords:
            G.add_node(kw, type="keyword")
            G.add_edge(doc_id, kw, relation="mentions")

    if use_lstm_links and model:
        embeddings = df["embedding"].tolist()
        titles = df["title"].tolist()
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                score = predict_link_score(np.array(embeddings[i]), np.array(embeddings[j]), model)
                if score >= threshold:
                    G.add_edge(titles[i], titles[j], relation="similar_to", weight=score)

    return G


def preprocess_query(query):
    spell = SpellChecker()
    corrected_query = " ".join([spell.correction(word) for word in query.split()])

    corrected_query = re.sub(r"[^a-zA-Z0-9\s]", "", corrected_query)

    return corrected_query

class GRUForecast(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def compute_yearly_centrality(docs, cluster_label):
    yearly_embeddings = defaultdict(list)
    for doc in docs:
        if doc.get("cluster") == cluster_label and "specter_embedding" in doc and "year" in doc:
            yearly_embeddings[int(doc["year"])].append(np.array(doc["specter_embedding"]))

    centrality_by_year = {}
    for year, embeds in yearly_embeddings.items():
        if len(embeds) < 2:
            continue
        embeds = np.stack(embeds)
        centroid = np.mean(embeds, axis=0, keepdims=True)
        sims = cosine_similarity(embeds, centroid)
        centrality_by_year[year] = float(np.mean(sims))

    return dict(sorted(centrality_by_year.items()))

def forecast_centrality(time_series, forecast_horizon=3):
    values = np.array(list(time_series.values()), dtype=np.float32)
    if len(values) < 4:
        return None

    min_val, max_val = np.min(values), np.max(values)
    norm = (values - min_val) / (max_val - min_val + 1e-8)
    X = torch.tensor([norm[-3:]], dtype=torch.float32).unsqueeze(-1)

    model = GRUForecast()
    model.eval()

    preds = []
    current = X.clone()
    for _ in range(forecast_horizon):
        with torch.no_grad():
            next_val = model(current)
        preds.append(next_val.item())
        current = torch.cat((current[:, 1:, :], next_val.view(1, 1, 1)), dim=1)

    preds = np.array(preds) * (max_val - min_val + 1e-8) + min_val
    return preds.tolist()

def show_dynamic_sdaf_section(docs):
    st.markdown("## Semantic Drift-Aware Forecasting (SDAF)")

    clusters = sorted(set(doc.get("cluster") for doc in docs if "cluster" in doc))
    if not clusters:
        st.warning("No clusters found in current selection.")
        return

    selected_cluster = st.selectbox("Select a cluster:", clusters)
    if not selected_cluster:
        return

    yearly_centrality = compute_yearly_centrality(docs, selected_cluster)
    forecast = forecast_centrality(yearly_centrality)

    if not forecast:
        st.warning("Not enough historical data to forecast.")
        return

    years = list(yearly_centrality.keys())
    values = list(yearly_centrality.values())
    forecast_years = list(range(years[-1] + 1, years[-1] + 1 + len(forecast)))

    plt.figure(figsize=(8, 4))
    plt.plot(years, values, marker='o', label='Historical', color='blue')
    plt.plot(forecast_years, forecast, marker='x', linestyle='--', label='Forecast', color='orange')
    plt.xlabel("Year")
    plt.ylabel("Centrality")
    plt.title(f"SDAF Forecast for Cluster: {selected_cluster}")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()


def get_most_likely_field(query):
    field_descriptions = {
        "q-bio.NC": "Neuroscience and neural systems.",
        "cs.CL": "Computational linguistics and natural language processing.",
        "physics.bio-ph": "Biological physics and biophysics.",
        "stat.ML": "Machine learning and statistical methods.",
        "q-bio.BM": "Biomolecules and molecular biology."
    }

    st.write(f"Debug: Query = {query}")

    query_embedding = vectorize_text_specter(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    st.write(f"Debug: Query embedding shape = {query_embedding.shape}")

    field_embeddings = {}
    for field, description in field_descriptions.items():
        field_embedding = vectorize_text_specter(description)
        field_embeddings[field] = np.array(field_embedding).reshape(1, -1)

        st.write(f"Debug: Field = {field}, Description = {description}")
        st.write(f"Debug: Field embedding shape = {field_embeddings[field].shape}")

    similarities = {}
    for field, embedding in field_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)[0][0]
        similarities[field] = similarity

        st.write(f"Debug: Similarity for {field} = {similarity}")

    most_likely_field = max(similarities, key=similarities.get)

    st.write(f"Debug: Most likely field = {most_likely_field}")

    return most_likely_field

def load_tlg_scores(path="tlg_scores.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tlg_map = {}
        for entry in data:
            key = tuple(sorted([entry["doc1"], entry["doc2"]]))
            tlg_map[key] = entry["tlg"]
        return tlg_map
    except Exception as e:
        print("⚠Could not load TLG scores:", e)
        return {}

def calculate_inertia(embeddings, max_clusters=10):
    inertia_values = []
    n_samples = len(embeddings)

    for n_clusters in range(1, max_clusters + 1):
        if n_clusters > n_samples:
            break
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)

    return inertia_values


def compare_documents(embeddings_dict, abstracts_dict, cluster_labels):
    st.markdown("### Compare Two Documents")

    with st.expander("Document Comparison Tool (Side-by-Side)", expanded=False):
        doc_ids = list(abstracts_dict.keys())
        if len(doc_ids) < 2:
            st.warning("Need at least 2 documents to compare.")
            return

        col1, col2 = st.columns(2)

        with col1:
            doc_a = st.selectbox("Select Document A", doc_ids, key="doc_a")
        with col2:
            doc_b = st.selectbox("Select Document B", [d for d in doc_ids if d != doc_a], key="doc_b")

        if doc_a and doc_b:
            # Abstracts
            st.markdown("#### Abstracts")
            col1, col2 = st.columns(2)
            col1.markdown(f"**{doc_a}**")
            col1.info(abstracts_dict.get(doc_a, "No abstract available."))
            col2.markdown(f"**{doc_b}**")
            col2.info(abstracts_dict.get(doc_b, "No abstract available."))

            # Cosine Similarity
            vec_a = embeddings_dict[doc_a].reshape(1, -1)
            vec_b = embeddings_dict[doc_b].reshape(1, -1)
            sim = cosine_similarity(vec_a, vec_b)[0][0]
            st.markdown(f"**Cosine Similarity:** `{sim:.4f}`")

            cluster_a = cluster_labels.get(doc_a, "N/A")
            cluster_b = cluster_labels.get(doc_b, "N/A")
            same_cluster = cluster_a == cluster_b
            st.markdown(f"**Same Cluster:** {'✅ Yes' if same_cluster else '❌ No'} (A: {cluster_a}, B: {cluster_b})")

            vect = TfidfVectorizer(stop_words='english', max_features=100)
            vect.fit([abstracts_dict[doc_a], abstracts_dict[doc_b]])
            keywords_a = set(vect.build_analyzer()(abstracts_dict[doc_a]))
            keywords_b = set(vect.build_analyzer()(abstracts_dict[doc_b]))
            shared_keywords = keywords_a & keywords_b
            st.markdown("**Shared Keywords:**")
            st.write(", ".join(shared_keywords) if shared_keywords else "None found")


def calculate_silhouette_scores(embeddings, max_clusters=10):
    silhouette_scores = []
    n_samples = len(embeddings)

    for n_clusters in range(2, max_clusters + 1):
        if n_clusters >= n_samples:
            break
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    return silhouette_scores


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


GENERIC_TERMS = {"distributed", "data", "approach", "problem", "text", "paper", "model"}

def generate_natural_explanation(feature_importances, feature_names, doc1, doc2, top_k=5):
    """
    Create a natural language explanation for influence prediction between two documents.
    Filters generic tokens and skips blank authors.
    """
    importance_pairs = list(zip(feature_names, feature_importances))
    sorted_pairs = sorted(importance_pairs, key=lambda x: abs(x[1]), reverse=True)

    doc1_words, doc2_words, diff_words = [], [], []

    for name, score in sorted_pairs:
        if len(doc1_words + doc2_words + diff_words) >= top_k:
            break
        keyword = name.split(": ")[1].strip().lower()
        if keyword in GENERIC_TERMS:
            continue
        if "doc1" in name:
            doc1_words.append(keyword)
        elif "doc2" in name:
            doc2_words.append(keyword)
        elif "diff" in name:
            diff_words.append(keyword)

    explanation = "This document is predicted to influence the other because they both relate to "
    shared_terms = set(doc1_words) & set(doc2_words)

    if shared_terms:
        explanation += ", ".join(shared_terms)
    elif diff_words:
        explanation += ", ".join(diff_words)
    elif doc1_words or doc2_words:
        explanation += ", ".join(doc1_words + doc2_words)
    else:
        explanation += "several overlapping concepts"

    authors_1 = {a.strip().lower() for a in doc1.get("authors", []) if a.strip()}
    authors_2 = {a.strip().lower() for a in doc2.get("authors", []) if a.strip()}
    common_authors = authors_1 & authors_2

    if common_authors:
        readable_authors = [a.title() for a in sorted(common_authors)]
        explanation += f" and share common authors such as {', '.join(readable_authors[:2])}"

    explanation += "."

    return explanation



def add_pdf_summarization():
    st.sidebar.subheader("PDF Summarization")

    pdf_dir = "app/documents/pdfs"
    if not os.path.exists(pdf_dir):
        st.sidebar.warning(f"Directory '{pdf_dir}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        st.sidebar.warning(f"No PDF files found in '{pdf_dir}'.")
        return

    selected_pdf = st.sidebar.selectbox("Select a PDF file", pdf_files)

    max_length = st.sidebar.slider("Maximum Summary Length", 50, 200, 130)
    min_length = st.sidebar.slider("Minimum Summary Length", 10, 100, 30)

    st.sidebar.markdown("### Timeline Overlay")
    enable_drift = st.sidebar.checkbox("Show Group Drift Overlay", value=True)

    if st.sidebar.button("Summarize"):
        with st.spinner("Summarizing PDF... This may take a moment."):
            try:
                pdf_path = os.path.join(pdf_dir, selected_pdf)
                st.sidebar.info(f"Extracting text from '{selected_pdf}'...")

                text = extract_text_from_pdf(pdf_path)

                if not text or len(text.strip()) == 0:
                    st.error("Could not extract any text from the PDF. The file might be scanned or protected.")
                    return

                st.sidebar.info("Summarizing text...")
                summary = chunk_and_summarize(text, max_length=max_length, min_length=min_length)

                if summary.startswith("Error"):
                    st.error(summary)
                else:
                    st.subheader(f"Summary of '{selected_pdf}'")
                    st.write(summary)

            except Exception as e:
                st.error(f"Error during summarization: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


def export_knowledge_graph_svg(G):
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(12, 8))

    node_colors = []
    for _, data in G.nodes(data=True):
        t = data.get("type")
        node_colors.append({
                               "document": "#8ecae6",
                               "author": "#ffafcc",
                               "keyword": "#ffd6a5",
                               "cluster": "#caffbf"
                           }.get(t, "#cccccc"))

    nx.draw(
        G, pos, ax=ax,
        with_labels=False,
        node_size=100,
        node_color=node_colors,
        edge_color="#999999",
        width=0.5,
        alpha=0.9
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    buf.seek(0)
    return buf


def export_all_formats(G, zip_path="knowledge_graph_bundle.zip", namespace="http://example.org/"):
    ttl, rdfxml, ntriples, json_data, cypher = [], [], [], {"nodes": [], "edges": []}, []

    for node_id, data in G.nodes(data=True):
        label = data.get("type", "Entity").capitalize()

        json_data["nodes"].append({
            "id": node_id,
            "type": data.get("type", "Entity"),
            "attributes": {k: str(v) for k, v in data.items() if k != "type"}
        })

        props = ", ".join([f'{k}: "{str(v).replace(chr(34), "")}"'
                           for k, v in data.items() if isinstance(v, (str, int, float))])
        cypher.append(f'CREATE (:`{label}` {{id: "{node_id}", {props}}});')

    for u, v, data in G.edges(data=True):
        pred = data.get("relation", "relatedTo")

        ttl.append(f'<{namespace}{u}> <{namespace}{pred}> <{namespace}{v}> .\n')
        ntriples.append(f'<{namespace}{u}> <{namespace}{pred}> <{namespace}{v}> .\n')
        rdfxml.append(f'  <rdf:Description rdf:about="{namespace}{u}">\n'
                      f'    <{pred} rdf:resource="{namespace}{v}" />\n'
                      f'  </rdf:Description>\n')

        json_data["edges"].append({
            "source": u,
            "target": v,
            "relation": pred
        })

        cypher.append(f'MATCH (a {{id: "{u}"}}), (b {{id: "{v}"}})\nCREATE (a)-[:`{pred.upper()}`]->(b);')

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("knowledge_graph.ttl", "".join(ttl))
        zf.writestr("knowledge_graph.nt", "".join(ntriples))
        zf.writestr("knowledge_graph.rdf",
                    '<?xml version="1.0"?>\n<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n' +
                    "".join(rdfxml) + '\n</rdf:RDF>'
                    )
        zf.writestr("knowledge_graph.json", json.dumps(json_data, indent=2))
        zf.writestr("knowledge_graph.cypher", "\n".join(cypher))

    return zip_path


def export_document_clusters(df, zip_path="document_clusters.zip"):
    cluster_json = df[["title", "authors", "year", "abstract", "cluster", "x", "y"]].to_dict(orient="records")
    cluster_csv = df[["title", "authors", "year", "abstract", "cluster", "x", "y"]]

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("clusters.csv", cluster_csv.to_csv(index=False))
        zf.writestr("clusters.json", json.dumps(cluster_json, indent=2))

    return zip_path


def export_network_graph(df, similarity_threshold, zip_path="network_graph.zip"):
    G = nx.Graph()
    for idx, row in df.iterrows():
        G.add_node(row["title"], cluster=row["cluster"], authors=row["authors"], year=row["year"])

    embeddings = np.array(df["embedding"].tolist())
    titles = df["title"].tolist()

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
            if sim > similarity_threshold:
                G.add_edge(titles[i], titles[j], similarity=sim)

    with zipfile.ZipFile(zip_path, "w") as zf:
        nx.write_graphml(G, "temp.graphml")
        nx.write_gexf(G, "temp.gexf")

        graph_json = nx.node_link_data(G)
        zf.writestr("network.json", json.dumps(graph_json, indent=2))

        cypher_lines = []
        for n, d in G.nodes(data=True):
            props = ", ".join([f'{k}: "{str(v)}"' for k, v in d.items()])
            cypher_lines.append(f'CREATE (:Document {{id: "{n}", {props}}});')
        for u, v, d in G.edges(data=True):
            cypher_lines.append(
                f'MATCH (a {{id: "{u}"}}), (b {{id: "{v}"}}) CREATE (a)-[:SIMILAR_TO {{similarity: {d["similarity"]:.4f}}}]->(b);')
        zf.writestr("network.cypher", "\n".join(cypher_lines))

        zf.write("temp.graphml", "network.graphml")
        zf.write("temp.gexf", "network.gexf")

    os.remove("temp.graphml")
    os.remove("temp.gexf")
    return zip_path


def get_feature_labels_from_dim_words(expanded_top_words: dict) -> list:
    """
    Constructs human-readable feature labels from dimension-to-keyword map.
    Format: ['doc1: word', 'doc2: word', 'diff: word']

    Args:
        expanded_top_words (dict): Mapping from dim index to top word

    Returns:
        List[str]: Full list of 2304 labels
    """
    labels = []
    for i in range(768):
        word = expanded_top_words.get(i, f"dim{i}")
        labels.append(f"doc1: {word}")
    for i in range(768):
        word = expanded_top_words.get(i, f"dim{i}")
        labels.append(f"doc2: {word}")
    for i in range(768):
        word = expanded_top_words.get(i, f"dim{i}")
        labels.append(f"diff: {word}")
    return labels


def compute_hybrid_similarity(doc1, doc2, embedding1, embedding2):
    cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]

    authors1 = set(a.strip().lower() for a in doc1["authors"].split(","))
    authors2 = set(a.strip().lower() for a in doc2["authors"].split(","))
    shared_authors = len(authors1 & authors2) / max(len(authors1 | authors2), 1)

    cluster_match = 1 if doc1["cluster"] == doc2["cluster"] else 0

    year1 = int(doc1.get("year", 2020))
    year2 = int(doc2.get("year", 2020))
    year_proximity = 1 - min(abs(year1 - year2), 10) / 10

    keywords1 = set(extract_keywords(doc1["abstract"]))
    keywords2 = set(extract_keywords(doc2["abstract"]))
    keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)

    hybrid_score = (
            0.60 * cosine_sim +
            0.15 * shared_authors +
            0.10 * cluster_match +
            0.10 * year_proximity +
            0.05 * keyword_overlap
    )

    return hybrid_score


def prepare_lstm_features(doc1, doc2):
    import numpy as np
    emb1 = doc1["embedding"]
    emb2 = doc2["embedding"]
    diff = np.abs(emb1 - emb2)
    return np.concatenate([emb1, emb2, diff])


@st.cache_resource
def load_custom_shap_explainer(_model, df, sample_size=20):
    import numpy as np
    import torch

    pairs = []
    embeddings = list(df['embedding'])[:sample_size]
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            emb1, emb2 = embeddings[i], embeddings[j]
            diff = np.abs(emb1 - emb2)
            vec = np.concatenate([emb1, emb2, diff])
            pairs.append(vec)
            if len(pairs) >= sample_size:
                break
        if len(pairs) >= sample_size:
            break

    background = np.stack(pairs)

    baseline_preds = []
    for sample in background:
        x_tensor = torch.tensor(sample.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            pred = _model(x_tensor).item()
        baseline_preds.append(pred)

    expected_value = np.mean(baseline_preds)

    def explain_prediction(features):
        n_features = len(features)
        feature_importances = np.zeros(n_features)

        x_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            baseline_pred = _model(x_tensor).item()

        for i in range(n_features):
            perturbed = features.copy()
            perturbed[i] = 0

            x_tensor = torch.tensor(perturbed.reshape(1, -1), dtype=torch.float32)
            with torch.no_grad():
                new_pred = _model(x_tensor).item()

            feature_importances[i] = baseline_pred - new_pred

        return feature_importances, expected_value

    return explain_prediction, expected_value

def export_cosine_graph(df, similarity_threshold=0.7, path="graph_cosine.gexf"):
    G = nx.Graph()
    embeddings = list(df["embedding"])
    titles = list(df["title"])

    for i in range(len(embeddings)):
        G.add_node(titles[i], title=titles[i])

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= similarity_threshold:
                G.add_edge(titles[i], titles[j], weight=sim)

    nx.write_gexf(G, path)
    print(f"✅ Cosine similarity graph saved to {path}")

def export_lstm_graph(df, model, threshold=0.8, path="graph_lstm.gexf"):
    G = nx.Graph()
    embeddings = list(df["embedding"])
    titles = list(df["title"])

    for i in range(len(embeddings)):
        G.add_node(titles[i], title=titles[i])

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            score = predict_link_score(np.array(embeddings[i]), np.array(embeddings[j]), model)
            if score >= threshold:
                G.add_edge(titles[i], titles[j], weight=score)

    nx.write_gexf(G, path)
    print(f"✅ LSTM prediction graph saved to {path}")



def main():
    st.title("Interactive Document Dashboard")
    st.sidebar.header("Filters and Settings")

    df = DataLoader.load_data()
    if df.empty:
        st.warning("No data found in ChromaDB. Please add documents first.")
        return

    st.sidebar.subheader("Filter by Year")
    min_year = int(df["year"].min())
    current_year = datetime.now().year

    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=current_year,
        value=(min_year, current_year),
        help="Filter documents by publication year."
    )

    filtered_df = df[
        (df["year"] >= year_range[0]) &
        (df["year"] <= year_range[1])
        ]

    embeddings = np.array(filtered_df["embedding"].tolist())

    clustering_algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical", "GMM"],
        help="Choose a clustering algorithm to group similar documents."
    )

    if clustering_algorithm in ["K-Means", "Hierarchical", "GMM"]:
        max_clusters = 30

        inertia_values = calculate_inertia(embeddings, max_clusters)
        silhouette_scores = calculate_silhouette_scores(embeddings, max_clusters)

        st.sidebar.subheader("Cluster Optimization Methods")

        with st.sidebar.expander("What are the Elbow Method and Silhouette Score?"):
            st.markdown("""
                **Elbow Method**:  
                The Elbow Method helps determine the optimal number of clusters by plotting the inertia (sum of squared distances to the nearest cluster center) against the number of clusters.  
                Look for the 'elbow' point where the inertia starts to decrease linearly. This point suggests the optimal number of clusters.

                **Silhouette Score**:  
                The Silhouette Score measures how similar an object is to its own cluster compared to other clusters.  
                Scores range from -1 to 1, where higher values indicate better-defined clusters.  
                The optimal number of clusters is where the Silhouette Score is highest.
            """)

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(range(1, max_clusters + 1)),
            y=inertia_values,
            mode='lines+markers',
            name='Inertia'
        ))
        fig_elbow.update_layout(
            title="Elbow Method",
            xaxis_title="Number of Clusters",
            yaxis_title="Inertia",
            showlegend=True
        )
        st.sidebar.plotly_chart(fig_elbow, use_container_width=True)

        fig_silhouette = go.Figure()
        fig_silhouette.add_trace(go.Scatter(
            x=list(range(2, max_clusters + 1)),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score'
        ))
        fig_silhouette.update_layout(
            title="Silhouette Scores",
            xaxis_title="Number of Clusters",
            yaxis_title="Silhouette Score",
            showlegend=True
        )
        st.sidebar.plotly_chart(fig_silhouette, use_container_width=True)

        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=2,
            max_value=max_clusters,
            value=3,
            help="Select the number of clusters to group documents into."
        )
    else:
        n_clusters = None

    cluster_labels = Clusterer.cluster_documents(
        embeddings,
        algorithm=clustering_algorithm,
        n_clusters=n_clusters
    )

    cluster_to_category = Clusterer.assign_categories_to_clusters(
        cluster_labels,
        embeddings,
        PREDEFINED_CATEGORIES
    )

    filtered_df["cluster"] = [cluster_to_category[label] for label in cluster_labels]

    docs_by_year = defaultdict(list)
    for idx, row in filtered_df.iterrows():
        docs_by_year[row["year"]].append({
            "title": row["title"],
            "year": row["year"],
            "cluster": row["cluster"],
            "embedding": row["embedding"]
        })

    centroids_by_year = defaultdict(dict)
    for year, docs in docs_by_year.items():
        clusters = defaultdict(list)
        for doc in docs:
            clusters[doc["cluster"]].append(doc["embedding"])
        for cluster, emb_list in clusters.items():
            centroids_by_year[year][cluster] = np.mean(np.array(emb_list), axis=0)

    years_sorted = sorted(centroids_by_year.keys())
    influence_scores = []
    for idx, year in enumerate(years_sorted[:-1]):
        current_year_docs = docs_by_year[year]
        for doc in current_year_docs:
            total_influence = 0
            doc_emb = np.array(doc["embedding"]).reshape(1, -1)
            for future_year in years_sorted[idx + 1:]:
                future_centroids = centroids_by_year[future_year].values()
                sims = cosine_similarity(doc_emb, np.array(list(future_centroids)))
                max_sim = np.max(sims)
                total_influence += max_sim

            influence_scores.append({
                "title": doc["title"],
                "year": year,
                "cluster": doc["cluster"],
                "influence_score": float(total_influence)
            })

    influence_scores_sorted = sorted(influence_scores, key=lambda x: x["influence_score"], reverse=True)

    os.makedirs("app", exist_ok=True)
    with open("app/semantic_influence_scores.json", "w", encoding="utf-8") as f:
        json.dump(influence_scores_sorted, f, indent=2, ensure_ascii=False)

    print("✅ Semantic influence scores saved to app/semantic_influence_scores.json")

    grouped_embeddings_by_year = defaultdict(lambda: defaultdict(list))

    for _, row in filtered_df.iterrows():
        year = int(row["year"])
        group = row["cluster"]
        embedding = row["embedding"]
        grouped_embeddings_by_year[year][group].append(embedding.tolist())

    os.makedirs("app/group_embeddings", exist_ok=True)
    path = "app/group_embeddings/group_embeddings_by_year.json"

    save_friendly = {str(y): {str(g): embs for g, embs in group_map.items()}
                     for y, group_map in grouped_embeddings_by_year.items()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(save_friendly, f, indent=2, ensure_ascii=False)

    from tlg.tlg_group import compute_group_drift

    drift_data = compute_group_drift(save_friendly)
    with open("app/group_embeddings/group_tlg_scores.json", "w", encoding="utf-8") as f:
        json.dump(drift_data, f, indent=2)


    drift_entries = []
    sorted_years = sorted(save_friendly.keys())

    for i in range(len(sorted_years) - 1):
        y1, y2 = sorted_years[i], sorted_years[i + 1]
        groups1 = save_friendly[y1]
        groups2 = save_friendly[y2]

        common_groups = set(groups1.keys()).intersection(groups2.keys())

        for group in common_groups:
            vecs1 = np.array(groups1[group])
            vecs2 = np.array(groups2[group])

            if len(vecs1) == 0 or len(vecs2) == 0:
                continue

            avg1 = np.mean(vecs1, axis=0).reshape(1, -1)
            avg2 = np.mean(vecs2, axis=0).reshape(1, -1)

            sim = float(cosine_similarity(avg1, avg2)[0][0])
            drift_entries.append({
                "group": group,
                "from": y1,
                "to": y2,
                "similarity": sim,
                "drift": sim - 1.0
            })

    with open("app/group_embeddings/group_tlg_scores.json", "w", encoding="utf-8") as f:
        json.dump(drift_entries, f, indent=2)

    reduction_method = st.sidebar.selectbox(
        "Select Method",
        ["PCA", "t-SNE", "UMAP"],
        help="Choose a method to reduce high-dimensional embeddings to 2D or 3D for visualization."
    )
    n_components = st.sidebar.radio(
        "Select Dimensions",
        [2, 3],
        help="Choose whether to visualize the data in 2D or 3D."
    )

    reduced_embeddings = DimensionalityReducer.reduce_embeddings(
        embeddings,
        method=reduction_method,
        n_components=n_components
    )

    if n_components == 2:
        filtered_df["x"] = reduced_embeddings[:, 0]
        filtered_df["y"] = reduced_embeddings[:, 1]
    elif n_components == 3:
        filtered_df["x"] = reduced_embeddings[:, 0]
        filtered_df["y"] = reduced_embeddings[:, 1]
        filtered_df["z"] = reduced_embeddings[:, 2]

    st.sidebar.subheader("Visualization Settings")
    tlg_enabled = st.sidebar.checkbox("Enable TLG Forecasting", value=False)
    tlg_threshold = st.sidebar.slider("TLG Threshold", min_value=-1.0, max_value=1.0, value=0.2, step=0.05)
    use_custom_colors = st.sidebar.checkbox(
        "Use Custom Node Colors",
        help="Enable to manually select colors for each cluster."
    )

    if use_custom_colors:
        custom_colors = []
        for i in range(n_clusters if n_clusters else 10):
            color = st.sidebar.color_picker(f"Cluster {i + 1} Color", "#0000FF")
            custom_colors.append(color)
    else:
        color_palette = st.sidebar.selectbox(
            "Node Color Scheme",
            ["Rainbow", "Nature", "Contrast", "Bright"],
            help="Choose a color scheme for the clusters."
        )
        custom_colors = Visualizer.get_distinct_colors(color_palette, n_clusters if n_clusters else 10)

    bg_colors = Visualizer.get_background_colors()
    bg_color_name = st.sidebar.selectbox(
        "Background Color",
        list(bg_colors.keys()),
        help="Choose the background color for the visualizations."
    )
    bg_color = bg_colors[bg_color_name]

    is_dark_bg = bg_color in ["black", "#2d2d2d", "#001f3f", "#1a472a"]
    text_color = "white" if is_dark_bg else "black"
    grid_color = "gray" if is_dark_bg else "LightGray"

    st.subheader("Document Clusters")
    with st.expander("What are Document Clusters?"):
        st.markdown("""
            **Document Clusters** group similar documents together based on their embeddings.  
            Each cluster represents a group of documents that share similar topics or themes.  
            Use the scatter plot to visualize the clusters in 2D or 3D space.
        """)

    marker_size = st.sidebar.slider(
        "Node Size",
        min_value=5,
        max_value=20,
        value=10,
        help="Adjust the size of nodes in the scatter plot."
    )
    marker_opacity = st.sidebar.slider(
        "Node Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Adjust the opacity of nodes in the scatter plot."
    )

    plot_settings = {
        "color": "cluster",
        "color_discrete_sequence": custom_colors,
        "hover_data": ["title", "authors", "year"],
        "opacity": marker_opacity,
    }

    if n_components == 2:
        fig = px.scatter(
            filtered_df,
            x="x",
            y="y",
            title="2D Document Clusters",
            **plot_settings
        )
        fig.update_traces(marker=dict(size=marker_size))

    elif n_components == 3:
        fig = px.scatter_3d(
            filtered_df,
            x="x",
            y="y",
            z="z",
            title="3D Document Clusters",
            **plot_settings
        )
        fig.update_traces(marker=dict(size=marker_size))

    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color),
        title_x=0.5,
        showlegend=True,
        legend_title_text='Cluster',
        legend=dict(
            font=dict(color=text_color),
            bgcolor='rgba(0,0,0,0)'
        )
    )

    axis_settings = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor=grid_color,
        color=text_color
    )
    fig.update_xaxes(**axis_settings)
    fig.update_yaxes(**axis_settings)
    if n_components == 3:
        fig.update_scenes(
            xaxis=axis_settings,
            yaxis=axis_settings,
            zaxis=axis_settings,
            bgcolor=bg_color
        )

    st.plotly_chart(fig, use_container_width=True)

    if st.button("📦 Export Document Clusters (.csv & .json)"):
        zip_path = export_document_clusters(filtered_df)
        with open(zip_path, "rb") as f:
            st.download_button("⬇️ Download Cluster Export", data=f, file_name="document_clusters.zip",
                               mime="application/zip")

    st.subheader("Interactive Network Graph")

    with st.expander("What is a Network Graph?"):
        st.markdown("""
            **Network Graph** visualizes relationships between documents.  
            Nodes represent documents, and edges represent similarities between them.  
            Use the slider to adjust the similarity threshold for connecting documents.
        """)

    similarity_threshold = st.slider(
        "Similarity Threshold for Edges",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1
    )

    similarity_mode = st.radio("Similarity Method", ["cosine", "hybrid", "cs2sim", "mvfusion"]) #MVFUSION TODO

    if similarity_mode == "hybrid":
        st.markdown("### Adjust Hybrid Similarity Weights (Sum ≤ 1.0)")

        with st.form(key="hybrid_similarity_form"):
            col1, col2 = st.columns(2)
            with col1:
                emb_weight = st.slider("Embedding weight", 0.0, 1.0, 0.60, 0.01)
                author_weight = st.slider("Author match weight", 0.0, 1.0, 0.15, 0.01)
                cluster_weight = st.slider("Cluster match weight", 0.0, 1.0, 0.10, 0.01)
            with col2:
                year_weight = st.slider("Year proximity weight", 0.0, 1.0, 0.10, 0.01)
                keyword_weight = st.slider("Abstract keyword overlap", 0.0, 1.0, 0.05, 0.01)

            weight_sum = emb_weight + author_weight + cluster_weight + year_weight + keyword_weight

            if weight_sum > 1.0:
                st.error(f"❌ Total weight = {weight_sum:.2f} (must be ≤ 1.0)")
                st.form_submit_button("✅ Recalculate and Render", disabled=True)
            else:
                st.success(f"✅ Total weight = {weight_sum:.2f}")
                recalc = st.form_submit_button("✅ Recalculate and Render")

        if 'recalc' in locals() and recalc:
            with st.spinner("Generating hybrid similarity graph..."):
                net, G = Visualizer.create_network_graph(
                    filtered_df,
                    similarity_threshold,
                    custom_colors,
                    similarity_mode="hybrid",
                    emb_weight=emb_weight,
                    author_weight=author_weight,
                    cluster_weight=cluster_weight,
                    year_weight=year_weight,
                    keyword_weight=keyword_weight
                )
                html_content = net.generate_html()
                st.components.v1.html(html_content, height=600, scrolling=True)
                st.success(f"{net.note}")

        st.markdown("### Adjust Hybrid Similarity Weights")
        with st.form(key="hybrid_similarity_form"):
            emb_weight = st.slider("Embedding weight", 0.0, 1.0, 0.60, 0.05)
            author_weight = st.slider("Author match weight", 0.0, 1.0, 0.15, 0.05)
            cluster_weight = st.slider("Cluster match weight", 0.0, 1.0, 0.10, 0.05)
            year_weight = st.slider("Year proximity weight", 0.0, 1.0, 0.10, 0.05)
            keyword_weight = st.slider("Abstract keyword overlap", 0.0, 1.0, 0.05, 0.01)

            recalc = st.form_submit_button("Recalculate and Render")

        if recalc:
            with st.spinner("Generating hybrid similarity graph..."):
                net, G = Visualizer.create_network_graph(
                    filtered_df,
                    similarity_threshold,
                    custom_colors,
                    similarity_mode="hybrid",
                    emb_weight=emb_weight,
                    author_weight=author_weight,
                    cluster_weight=cluster_weight,
                    year_weight=year_weight,
                    keyword_weight=keyword_weight
                )
                html_content = net.generate_html()
                st.components.v1.html(html_content, height=600, scrolling=True)
                st.success(f"🔗 {net.note}")

    if st.button("🚀 Build Network Graph (Intensive in 200+ nodes)"):
        with st.spinner("Generating interactive network..."):
            net, G = Visualizer.create_network_graph(
                filtered_df,
                similarity_threshold,
                custom_colors,
                similarity_mode=similarity_mode
            )

            html_content = net.generate_html()
            st.components.v1.html(html_content, height=600, scrolling=True)

            st.success(f"🔗 **{net.note}**")

    # Network graph export, doesn't work for now TODO
    # if st.button("📦 Export Interactive Network Graph"):
    #     zip_path = export_network_graph(filtered_df, similarity_threshold=similarity_threshold)
    #     with open(zip_path, "rb") as f:
    #         st.download_button("⬇️ Download Network Graph Export", data=f, file_name="network_graph.zip",
    #                            mime="application/zip")

    st.subheader("Advanced Visualizations")

    st.subheader("Document Similarity Heatmap")
    with st.expander("What is a Heatmap?"):
        st.markdown("""
            **Heatmap** shows the similarity between documents in a matrix format.  
            Darker colors indicate higher similarity, while lighter colors indicate lower similarity.  
            Use the slider to adjust the similarity threshold for the heatmap.
        """)

    similarity_threshold = st.slider(
        "Similarity Threshold for Heatmap",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust the similarity threshold to control which connections are shown in the heatmap."
    )
    heatmap_fig = Visualizer.create_heatmap(filtered_df, similarity_threshold)
    st.plotly_chart(heatmap_fig, use_container_width=True)

    st.subheader("Parallel Coordinates Plot")
    with st.expander("What is a Parallel Coordinates Plot?"):
        st.markdown("""
            **Parallel Coordinates Plot** visualizes multi-dimensional data.  
            Each line represents a document, and each axis represents a dimension (e.g., x, y, cluster).  
            Use this plot to explore relationships between dimensions.
        """)

    color_scheme = st.sidebar.selectbox(
        "Select Color Scheme",
        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
        help="Choose a color scheme for the parallel coordinates plot."
    )
    parallel_coords_fig = Visualizer.create_parallel_coordinates(filtered_df, color_scheme=color_scheme)
    if parallel_coords_fig:
        st.plotly_chart(parallel_coords_fig, use_container_width=True)

    st.subheader("Sankey Diagram")
    with st.expander("What is a Sankey Diagram?"):
        st.markdown("""
            **Sankey Diagram** visualizes the flow of documents between clusters.  
            The width of the links represents the number of documents flowing between clusters.  
            Use this diagram to understand how documents are distributed across clusters.
        """)

    sankey_fig = Visualizer.create_sankey_diagram(filtered_df)
    st.plotly_chart(sankey_fig, use_container_width=True)

    st.sidebar.subheader("Search Documents")
    search_query = st.sidebar.text_input(
        "Search by Title, Author, or Abstract",
        key="search_documents_input",
        help="Enter a query to search for documents by title, author, or abstract."
    )

    st.subheader("Knowledge Graph Viewer")

    if st.button("Build Knowledge Graph (Intensive in 200+ nodes)"):
        with st.spinner("Constructing Knowledge Graph..."):
            kg = build_knowledge_graph(filtered_df, model=load_link_model())
            st.session_state.kg = kg
            st.success(f"Graph built: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

            try:
                nx.write_gexf(kg, "knowledge_graph.gexf")
                st.info("Graph saved as knowledge_graph.gexf")
            except Exception as e:
                st.warning(f"Export failed: {e}")

            st.markdown("### Sample Relations")
            sample_edges = list(kg.edges(data=True))[:10]
            for u, v, data in sample_edges:
                st.write(f"{u} -[{data['relation']}]-> {v}")

            try:
                from pyvis.network import Network
                import streamlit.components.v1 as components

                net = Network(height="600px", width="100%")

                for node, data in kg.nodes(data=True):
                    label = node[:40]
                    color = {"document": "#8ecae6", "author": "#ffafcc", "keyword": "#ffd6a5",
                             "cluster": "#caffbf"}.get(data.get("type"), "#ccc")
                    net.add_node(node, label=label, color=color)

                for u, v, data in kg.edges(data=True):
                    net.add_edge(u, v, title=data.get("relation", ""))

                net.save_graph("kg_vis.html")
                with open("kg_vis.html", "r", encoding="utf-8") as f:
                    components.html(f.read(), height=650, scrolling=True)

            except Exception as e:
                st.warning(f"Graph visualization failed: {e}")

    if "kg" in st.session_state:
        if st.button("📦 Export Neo4j + RDF + JSON Bundle"):
            zip_path = export_all_formats(st.session_state.kg)
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download All Graph Formats (.zip)",
                    data=f,
                    file_name="knowledge_graph_bundle.zip",
                    mime="application/zip"
                )

    if st.checkbox("Show all relations"):
        for u, v, data in kg.edges(data=True):
            st.write(f"{u} -[{data['relation']}]-> {v}")

    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Set the minimum similarity score for documents to be included in the results."
    )

    if search_query:
        corrected_query = preprocess_query(search_query)

        search_results = SemanticSearch.semantic_search(
            filtered_df,
            corrected_query,
            top_k=5,
            similarity_threshold=similarity_threshold
        )

        if not search_results.empty:
            st.subheader("Search Results")
            st.write(search_results[["title", "authors", "year", "abstract", "similarity"]])
        else:
            st.warning("No documents found for the search query above the similarity threshold.")

        st.subheader("Related Articles from arXiv")
        arxiv_articles = ExternalAPIs.fetch_arxiv_articles(
            corrected_query,
            max_results=5
        )
        if arxiv_articles:
            for article in arxiv_articles:
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Authors:** {', '.join(article['authors'])}")
                st.write(f"**Published:** {article['published']}")
                st.write(f"**Summary:** {article['summary']}")
                st.write(f"**Link:** [Read Paper]({article['link']})")
                st.write("---")
        else:
            st.write("No related articles found in arXiv.")

    st.sidebar.subheader("Fetch Citation Count")
    citation_query = st.sidebar.text_input(
        "Enter a paper title to fetch citation count",
        key="citation_count_input",
        help="Enter the title of a paper to fetch its citation count from Google Scholar."
    )
    if citation_query:
        citation_count = ExternalAPIs.fetch_citation_count(citation_query)
        st.write(f"**Citation Count:** {citation_count}")

    st.subheader("Contextual Graph Evolution Explorer")

    if "context_history" not in st.session_state:
        st.session_state.context_history = []

    def select_context(name):
        cluster = st.selectbox(f"{name}: Cluster", ["All"] + sorted(filtered_df["cluster"].unique()),
                               key=f"{name}_cluster")
        year_range = st.slider(f"{name}: Year Range", 2010, 2025, (2015, 2023), key=f"{name}_year")
        threshold = st.slider(f"{name}: Link Threshold", 0.5, 0.95, 0.8, 0.01, key=f"{name}_thresh")
        return {"cluster": cluster, "year_range": year_range, "threshold": threshold}

    ctx1 = select_context("Context 1")
    ctx2 = select_context("Context 2")

    if st.button("Save Contexts to History"):
        st.session_state.context_history.append((ctx1, ctx2))
        st.success("Contexts saved.")

    def apply_context(df, ctx):
        result = df[df["year"].between(ctx["year_range"][0], ctx["year_range"][1])]
        if ctx["cluster"] != "All":
            result = result[result["cluster"] == ctx["cluster"]]
        return result

    df1 = apply_context(filtered_df, ctx1)
    df2 = apply_context(filtered_df, ctx2)

    model = load_link_model()
    G1 = build_dynamic_graph(df1, model, threshold=ctx1["threshold"])
    G2 = build_dynamic_graph(df2, model, threshold=ctx2["threshold"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔹 Context 1")
        st.write(f"{len(G1.nodes)} nodes, {len(G1.edges)} edges")
        from pyvis.network import Network
        import streamlit.components.v1 as components

        net1 = Network(height="500px", width="100%")
        for node in G1.nodes:
            data = G1.nodes[node]
            cluster = data.get("metadata", {}).get("cluster", "Unknown")
            net1.add_node(node, label=node, title=cluster, group=cluster)
        for u, v, data in G1.edges(data=True): net1.add_edge(u, v, value=data["weight"])
        net1.save_graph("graph1.html")
        components.html(open("graph1.html", "r").read(), height=550)

    with col2:
        st.markdown("### 🔸 Context 2")
        st.write(f"{len(G2.nodes)} nodes, {len(G2.edges)} edges")
        net2 = Network(height="500px", width="100%")
        for node in G2.nodes:
            data = G2.nodes[node]
            cluster = data.get("metadata", {}).get("cluster", "Unknown")
            net2.add_node(node, label=node, title=cluster, group=cluster)
        for u, v, data in G2.edges(data=True): net2.add_edge(u, v, value=data["weight"])
        net2.save_graph("graph2.html")
        components.html(open("graph2.html", "r").read(), height=550)

    def compare_graphs(G1, G2):
        edges1 = set(G1.edges())
        edges2 = set(G2.edges())
        intersection = edges1 & edges2
        union = edges1 | edges2
        jaccard = len(intersection) / len(union) if union else 0
        gained = edges2 - edges1
        lost = edges1 - edges2
        return jaccard, list(gained), list(lost)

    st.markdown("### 📊 Graph Comparison Metrics")
    jaccard, gained, lost = compare_graphs(G1, G2)
    st.write(f"**Jaccard Similarity (edges):** `{jaccard:.4f}`")
    st.write(f"🔼 **Links Gained in Context 2:** {len(gained)}")
    st.write(f"🔽 **Links Lost in Context 2:** {len(lost)}")

    if st.checkbox("🔍 Show Edge Changes"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gained Links**")
            for u, v in gained:
                st.write(f"{u} ↔ {v}")
        with col2:
            st.markdown("**Lost Links**")
            for u, v in lost:
                st.write(f"{u} ↔ {v}")

    os.makedirs("app/group_embeddings", exist_ok=True)
    with open("app/group_embeddings/group_embeddings_by_year.json", "w", encoding="utf-8") as f:
        json.dump(grouped_embeddings_by_year, f, indent=2, ensure_ascii=False)

    st.markdown("### Group-Level Drift Between Contexts")

    drift_data = load_group_drift("app/group_embeddings/group_tlg_scores.json")

    drift_net = Network(height="500px", width="100%", notebook=False, directed=True)

    ctx1_years = range(ctx1["year_range"][0], ctx1["year_range"][1] + 1)
    ctx2_years = range(ctx2["year_range"][0], ctx2["year_range"][1] + 1)

    for entry in drift_data:
        group = entry["group"]
        y1 = int(entry["from"])
        y2 = int(entry["to"])
        drift = entry["drift"]
        sim = entry["similarity"]

        if y1 in ctx1_years and y2 in ctx2_years:
            n1 = f"{group}_{y1}"
            n2 = f"{group}_{y2}"
            color = "green" if drift >= 0 else "red"
            width = min(5, max(1.5, abs(drift * 5)))

            drift_net.add_node(n1, label=f"{group} ({y1})", group=group, shape="ellipse")
            drift_net.add_node(n2, label=f"{group} ({y2})", group=group, shape="ellipse")

            drift_net.add_edge(
                n1, n2,
                title=f"Δ: {drift:.2f}",
                value=abs(drift),
                color=color,
                arrows="to",
                width=width
            )

    st.components.v1.html(drift_net.generate_html(), height=520, scrolling=True)

    st.subheader("🔗 Dynamic Document Graph (LSTM-Based)")
    with st.expander("View and Explore Dynamic Graph", expanded=False):

        threshold = st.slider("Link prediction threshold", min_value=0.5, max_value=0.95, step=0.01, value=0.8)
        model = load_link_model()
        G = build_dynamic_graph(filtered_df, model, threshold=threshold)

        st.markdown(f"Graph contains `{len(G.nodes)}` documents and `{len(G.edges)}` predicted links")

        if st.checkbox("Show edge list"):
            edges = list(G.edges(data=True))
            for u, v, data in edges:
                st.write(f"{u} ↔ {v} (score: {data['weight']:.3f})")

        try:
            from pyvis.network import Network
            import streamlit.components.v1 as components

            net = Network(height="600px", width="100%", notebook=False)

            for node in G.nodes:
                net.add_node(node, label=node, title=node)

            for u, v, data in G.edges(data=True):
                net.add_edge(u, v, value=data["weight"])

            net.save_graph("graph.html")
            with open("graph.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            components.html(html_content, height=650, scrolling=True)

        except Exception as e:
            st.warning("Graph visualization skipped (install `pyvis` to enable).")
            st.text(str(e))

    # doesn't work for now TODO
    # st.subheader("🔍 Explain LSTM Link Prediction (SHAP)")
    #
    # model = load_link_model()
    #
    # def predict_fn(X):
    #     with torch.no_grad():
    #         X_tensor = torch.tensor(X, dtype=torch.float32)
    #         out = model(X_tensor)
    #         return out.detach().numpy().reshape(-1, 1)
    #
    # explainer = load_shap_explainer(model, filtered_df)
    #
    # titles = list(filtered_df["title"])
    # doc1_title = st.selectbox("Document A", titles)
    # doc2_title = st.selectbox("Document B", titles, index=1)
    #
    # doc1 = filtered_df[filtered_df["title"] == doc1_title].iloc[0]
    # doc2 = filtered_df[filtered_df["title"] == doc2_title].iloc[0]
    #
    # features = prepare_lstm_features(doc1, doc2)
    # shap_values = explainer.shap_values(np.array([features]))
    #
    # st.write(f"Predicted score: {predict_fn([features])[0]:.3f}")
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # shap.plots.bar(shap_values[0], show=False)
    # st.pyplot(bbox_inches="tight")

    st.subheader("📄 Document Comparison Tool (Side-by-Side)")
    with st.expander("Compare Two Documents", expanded=False):

        doc_titles = filtered_df["title"].tolist()

        with st.form("compare_form"):
            col1, col2 = st.columns(2)
            with col1:
                doc1_title = st.selectbox("Select Document 1", doc_titles, key="doc1")
            with col2:
                doc2_title = st.selectbox("Select Document 2", doc_titles, key="doc2")

            submitted = st.form_submit_button("Compare")

        if submitted:
            doc1 = filtered_df[filtered_df["title"] == doc1_title].iloc[0]
            doc2 = filtered_df[filtered_df["title"] == doc2_title].iloc[0]

            emb1 = np.array(doc1["embedding"])
            emb2 = np.array(doc2["embedding"])

            sim_cosine = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
            sim_euclidean = euclidean(emb1, emb2)
            sim_manhattan = cityblock(emb1, emb2)
            sim_dot = np.dot(emb1, emb2)

            st.markdown("### 🔎 Similarity Metrics")
            sim_df = pd.DataFrame({
                "Metric": ["Cosine Similarity", "Euclidean Distance", "Manhattan Distance", "Dot Product"],
                "Value": [f"{sim_cosine:.4f}", f"{sim_euclidean:.4f}", f"{sim_manhattan:.4f}", f"{sim_dot:.4f}"]
            })
            st.table(sim_df)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"###  {doc1['title']}")
                st.markdown(f"**Authors:** {doc1['authors']}")
                st.markdown(f"**Year:** {doc1['year']}  \n**Cluster:** `{doc1['cluster']}`")
                st.markdown("**Abstract:**")
                st.write(doc1["abstract"])

            with col2:
                st.markdown(f"###  {doc2['title']}")
                st.markdown(f"**Authors:** {doc2['authors']}")
                st.markdown(f"**Year:** {doc2['year']}  \n**Cluster:** `{doc2['cluster']}`")
                st.markdown("**Abstract:**")
                st.write(doc2["abstract"])

            def extract_keywords(text, top_n=10):
                stop_words = set(stopwords.words('english'))
                words = word_tokenize(text.lower())
                words = [re.sub(r'\W+', '', w) for w in words if w not in stop_words and len(w) > 2]
                freq = Counter(words)
                return [word for word, _ in freq.most_common(top_n)]

            kw1 = set(extract_keywords(doc1["abstract"]))
            kw2 = set(extract_keywords(doc2["abstract"]))

            shared_kw = sorted(kw1 & kw2)
            unique_kw1 = sorted(kw1 - kw2)
            unique_kw2 = sorted(kw2 - kw1)

            st.markdown("### Explanation: Why Are They Similar/Different?")
            st.markdown("**Shared Keywords:**")
            st.write(", ".join(shared_kw) if shared_kw else "*None*")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Unique to Document 1:**")
                st.write(", ".join(unique_kw1) if unique_kw1 else "*None*")
            with col2:
                st.markdown("**Unique to Document 2:**")
                st.write(", ".join(unique_kw2) if unique_kw2 else "*None*")

    st.subheader("Influence Network")

    num_vis_docs = st.slider("Number of Top Influential Documents in Network", 5, 30, 10)

    net = Network(height="650px", width="100%", notebook=False)

    net.barnes_hut(
        gravity=-8000,
        central_gravity=0.3,
        spring_length=300,
        spring_strength=0.005,
        damping=0.1
    )

    for doc in influence_scores[:num_vis_docs]:
        net.add_node(
            doc["title"],
            label=f'{doc["title"][:50]} ({doc["year"]})',
            title=f'Influence: {doc["influence_score"]:.2f}<br>Cluster: {doc["cluster"]}',
            size=10 + doc["influence_score"] * 5,
            color="orange"
        )

    years_sorted = sorted(filtered_df["year"].unique())
    for doc in influence_scores[:num_vis_docs]:
        source = doc["title"]
        embedding_row = filtered_df[filtered_df["title"] == doc["title"]]
        if embedding_row.empty:
            continue
        doc_emb = np.array(embedding_row.iloc[0]["embedding"]).reshape(1, -1)

        top_links = []
        doc_year_index = years_sorted.index(doc["year"])

        for future_year in years_sorted[doc_year_index + 1:]:
            for cluster, centroid in centroids_by_year[future_year].items():
                target = f"{cluster} ({future_year})"
                sim = cosine_similarity(doc_emb, centroid.reshape(1, -1))[0][0]
                top_links.append((sim, target, future_year, cluster))

        top_links = sorted(top_links, reverse=True)[:2]

        for sim, target, future_year, cluster in top_links:
            net.add_node(target, label=target, color="lightblue", shape="ellipse")

            min_width = 0.1
            max_width = 20
            thickness = min_width + (sim ** 2) * (max_width - min_width)

            label = f"📘 Future cluster: {cluster} ({future_year})\n Similarity: {sim:.3f}"

            net.add_edge(
                source,
                target,
                color="red",
                width=thickness,
                title=label,
                arrows="to"
            )

    net.save_graph("app/influence_network.html")
    HtmlFile = open("app/influence_network.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=650, width=None)

    st.subheader("Explain Link Prediction")

    doc_titles = list(filtered_df["title"])
    doc1_title = st.selectbox("Document A", doc_titles, key="shap_doc1")
    doc2_title = st.selectbox("Document B", doc_titles, key="shap_doc2", index=min(1, len(doc_titles) - 1))

    if st.button("Explain Influence Prediction"):
        with st.spinner("Calculating feature importance..."):
            doc1 = filtered_df[filtered_df["title"] == doc1_title].iloc[0]
            doc2 = filtered_df[filtered_df["title"] == doc2_title].iloc[0]

            features = prepare_lstm_features(doc1, doc2)

            pred_score = predict_link_score(doc1["embedding"], doc2["embedding"], model)
            st.markdown(f"### Predicted Influence Score: `{pred_score:.4f}`")

            explain_fn, expected_value = load_custom_shap_explainer(_model=model, df=filtered_df)

            try:
                start_time = time.time()
                feature_importances, baseline = explain_fn(features)
                end_time = time.time()

                st.markdown(f"### Feature Importance Analysis (Computed in {end_time - start_time:.2f}s)")

                emb_dim = len(features) // 3

                expanded_top_words = compute_dim_keywords_from_abstracts("app/fused_documents.json")

                feature_names = get_feature_labels_from_dim_words(expanded_top_words)

                shap.plots.bar(
                    shap.Explanation(
                        values=feature_importances,
                        base_values=baseline,
                        feature_names=feature_names
                    ),
                    show=False
                )

                top_n = 20
                sorted_idx = np.argsort(np.abs(feature_importances))[-top_n:]

                fig, ax = plt.subplots(figsize=(10, 8))

                y_pos = np.arange(len(sorted_idx))
                ax.barh(y_pos,
                        feature_importances[sorted_idx],
                        color=['#1E88E5' if x > 0 else '#FF0D57' for x in feature_importances[sorted_idx]])

                ax.set_yticks(y_pos)
                ax.set_yticklabels([feature_names[i] for i in sorted_idx])
                ax.invert_yaxis()
                ax.set_xlabel('Impact on prediction')
                ax.set_title('Top Feature Importances')

                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

                import matplotlib.patches as mpatches
                positive_patch = mpatches.Patch(color='#1E88E5', label='Increases score')
                negative_patch = mpatches.Patch(color='#FF0D57', label='Decreases score')
                ax.legend(handles=[positive_patch, negative_patch])

                st.pyplot(fig)

                st.markdown("### 💡 Interpretation")
                st.markdown("""
                    - **Positive values (blue)**: These features increase the predicted influence score 
                    - **Negative values (red)**: These features decrease the predicted influence score
                    - **Larger bars**: Higher impact on the model's prediction
                    """)

                st.markdown("### 📑 Document Contribution Analysis")

                doc1_imp = np.sum(np.abs(feature_importances[:emb_dim]))
                doc2_imp = np.sum(np.abs(feature_importances[emb_dim:2 * emb_dim]))
                diff_imp = np.sum(np.abs(feature_importances[2 * emb_dim:]))
                total_imp = doc1_imp + doc2_imp + diff_imp

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie([doc1_imp, doc2_imp, diff_imp],
                       labels=[f"Document A\n({doc1_imp / total_imp:.1%})",
                               f"Document B\n({doc2_imp / total_imp:.1%})",
                               f"Difference\n({diff_imp / total_imp:.1%})"],
                       autopct='%1.1f%%',
                       colors=['#1E88E5', '#FF0D57', '#33BB33'])
                ax.set_title('Contribution by Feature Group')
                st.pyplot(fig)

                st.markdown("Semantic Explanation (Offline LLM)")
                explanation = generate_local_semantic_explanation(doc1, doc2)
                st.markdown("### Offline LLM Explanation")
                st.markdown(f"> {explanation}")

            except Exception as e:
                st.error(f"Error calculating feature importance: {str(e)}")
                st.code(str(e), language="python")

    with st.expander("Semantic Drift-Aware Forecasting (SDAF)", expanded=False):
        filtered_df.to_csv("filtered_df_example.csv", index=False)

        def df_to_docs(df):
            docs = []
            for _, row in df.iterrows():
                try:
                    embed = row["embedding"]
                    if isinstance(embed, str):
                        embed = literal_eval(embed)
                    docs.append({
                        "cluster": row.get("cluster"),
                        "specter_embedding": embed,
                        "year": int(row.get("year", 0))
                    })
                except Exception as e:
                    continue
            return docs

        filtered_docs = df_to_docs(filtered_df)
        show_dynamic_sdaf_section(filtered_docs)

    export_cosine_graph(filtered_df, similarity_threshold=0.8)
    export_lstm_graph(filtered_df, model, threshold=0.3)

    st.subheader("Cluster Details")
    with st.expander("View Documents in Each Cluster"):
        for cluster_name, group in filtered_df.groupby("cluster"):
            st.write(f"### Cluster: {cluster_name}")
            st.write(group[["title", "authors", "year", "abstract"]])

    add_pdf_summarization()


if __name__ == "__main__":
    main()