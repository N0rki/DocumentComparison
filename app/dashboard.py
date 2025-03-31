import os
import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import feedparser
import torch
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
    def create_network_graph(df, similarity_threshold=0.7, custom_colors=None):
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
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[j].reshape(1, -1)
                )[0][0]
                if similarity > similarity_threshold:
                    G.add_edge(i, j, weight=float(similarity))

        net = Network(height="600px", width="100%", notebook=False, directed=False)
        net.from_nx(G)

        options = {
            "physics": {
                "barnesHut": {
                    "avoidOverlap": 0.02
                },
                "minVelocity": 0.75
            }
        }

        options_str = json.dumps(options)
        net.set_options(options_str)
        net.save_graph("network.html")
        return net

    @staticmethod
    def create_heatmap(df, similarity_threshold=0.7):
        embeddings = np.array(df["embedding"].tolist())
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix[similarity_matrix < similarity_threshold] = 0

        truncated_titles = [title[:50] + "..." if len(title) > 50 else title for title in df["title"].tolist()]

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

        cluster_mapping = {cluster: i for i, cluster in enumerate(df["cluster"].unique())}
        df["cluster_numeric"] = df["cluster"].map(cluster_mapping)

        fig = px.parallel_coordinates(
            df,
            dimensions=["x", "y", "cluster_numeric"],
            color="cluster_numeric",
            labels={"x": "X", "y": "Y", "cluster_numeric": "Cluster"},
            color_continuous_scale=color_scheme,
        )
        fig.update_layout(title="Parallel Coordinates Plot")
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

def predict_link_score(emb1, emb2, model):
    diff = np.abs(emb1 - emb2)
    x = np.concatenate([emb1, emb2, diff])
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = model(x_tensor).item()
    return score

def build_dynamic_graph(filtered_df, model, threshold=0.8):
    G = nx.Graph()
    for i, row in filtered_df.iterrows():
        G.add_node(row["title"], metadata=row)

    embeddings = filtered_df["embedding"].tolist()
    titles = filtered_df["title"].tolist()

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            score = predict_link_score(np.array(embeddings[i]), np.array(embeddings[j]), model)
            if score >= threshold:
                G.add_edge(titles[i], titles[j], weight=score)

    return G

def preprocess_query(query):
    spell = SpellChecker()
    corrected_query = " ".join([spell.correction(word) for word in query.split()])

    corrected_query = re.sub(r"[^a-zA-Z0-9\s]", "", corrected_query)

    return corrected_query

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

def calculate_inertia(embeddings, max_clusters=10):
    inertia_values = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)
    return inertia_values

def compare_documents(embeddings_dict, abstracts_dict, cluster_labels):
    st.markdown("### 🔍 Compare Two Documents")

    with st.expander("📄 Document Comparison Tool (Side-by-Side)", expanded=False):
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
            st.markdown("#### 📘 Abstracts")
            col1, col2 = st.columns(2)
            col1.markdown(f"**{doc_a}**")
            col1.info(abstracts_dict.get(doc_a, "No abstract available."))
            col2.markdown(f"**{doc_b}**")
            col2.info(abstracts_dict.get(doc_b, "No abstract available."))

            # Cosine Similarity
            vec_a = embeddings_dict[doc_a].reshape(1, -1)
            vec_b = embeddings_dict[doc_b].reshape(1, -1)
            sim = cosine_similarity(vec_a, vec_b)[0][0]
            st.markdown(f"**🧠 Cosine Similarity:** `{sim:.4f}`")

            # Cluster Match
            cluster_a = cluster_labels.get(doc_a, "N/A")
            cluster_b = cluster_labels.get(doc_b, "N/A")
            same_cluster = cluster_a == cluster_b
            st.markdown(f"**🧩 Same Cluster:** {'✅ Yes' if same_cluster else '❌ No'} (A: {cluster_a}, B: {cluster_b})")

            # Shared Keywords (basic TF-IDF)
            vect = TfidfVectorizer(stop_words='english', max_features=100)
            vect.fit([abstracts_dict[doc_a], abstracts_dict[doc_b]])
            keywords_a = set(vect.build_analyzer()(abstracts_dict[doc_a]))
            keywords_b = set(vect.build_analyzer()(abstracts_dict[doc_b]))
            shared_keywords = keywords_a & keywords_b
            st.markdown("**🔁 Shared Keywords:**")
            st.write(", ".join(shared_keywords) if shared_keywords else "None found")

def calculate_silhouette_scores(embeddings, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
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
        step=0.1,
        help="Adjust the similarity threshold to control how documents are connected in the graph."
    )

    net = Visualizer.create_network_graph(filtered_df, similarity_threshold, custom_colors)

    with open("network.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)

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

    st.subheader("🔗 Dynamic Document Graph (LSTM-Based)")
    with st.expander("View and Explore Dynamic Graph", expanded=False):

        threshold = st.slider("Link prediction threshold", min_value=0.5, max_value=0.95, step=0.01, value=0.8)
        model = load_link_model()
        G = build_dynamic_graph(filtered_df, model, threshold=threshold)

        st.markdown(f"📌 Graph contains `{len(G.nodes)}` documents and `{len(G.edges)}` predicted links")

        # Optionally show network as edge list
        if st.checkbox("Show edge list"):
            edges = list(G.edges(data=True))
            for u, v, data in edges:
                st.write(f"{u} ↔ {v} (score: {data['weight']:.3f})")

        # Optional: Visualize using pyvis or plotly
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
                st.markdown(f"### 📘 {doc1['title']}")
                st.markdown(f"**Authors:** {doc1['authors']}")
                st.markdown(f"**Year:** {doc1['year']}  \n**Cluster:** `{doc1['cluster']}`")
                st.markdown("**Abstract:**")
                st.write(doc1["abstract"])

            with col2:
                st.markdown(f"### 📗 {doc2['title']}")
                st.markdown(f"**Authors:** {doc2['authors']}")
                st.markdown(f"**Year:** {doc2['year']}  \n**Cluster:** `{doc2['cluster']}`")
                st.markdown("**Abstract:**")
                st.write(doc2["abstract"])

            # Keyword Explanation Section
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

            st.markdown("### 🧠 Explanation: Why Are They Similar/Different?")
            st.markdown("**Shared Keywords:**")
            st.write(", ".join(shared_kw) if shared_kw else "*None*")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Unique to Document 1:**")
                st.write(", ".join(unique_kw1) if unique_kw1 else "*None*")
            with col2:
                st.markdown("**Unique to Document 2:**")
                st.write(", ".join(unique_kw2) if unique_kw2 else "*None*")

    st.subheader("Cluster Details")
    with st.expander("View Documents in Each Cluster"):
        for cluster_name, group in filtered_df.groupby("cluster"):
            st.write(f"### Cluster: {cluster_name}")
            st.write(group[["title", "authors", "year", "abstract"]])

    add_pdf_summarization()

if __name__ == "__main__":
    main()