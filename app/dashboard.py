import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import feedparser
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


# Fetch data from ChromaDB
class DataLoader:
    @staticmethod
    def load_data_from_chromadb():
        """
        Fetch embeddings and metadata from ChromaDB with debug statements.
        """
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


# Dimensionality reduction
class DimensionalityReducer:
    @staticmethod
    def reduce_embeddings(embeddings, n_components=2, method="PCA"):
        """
        Reduce high-dimensional embeddings to 2D or 3D using PCA, t-SNE, or UMAP.
        """
        if method == "PCA":
            reducer = PCA(n_components=n_components)
        elif method == "t-SNE":
            reducer = TSNE(n_components=n_components, perplexity=3)
        elif method == "UMAP":
            reducer = UMAP(n_components=n_components)
        else:
            raise ValueError("Unsupported dimensionality reduction method")
        return reducer.fit_transform(embeddings)


# Clustering
class Clusterer:
    @staticmethod
    def cluster_documents(embeddings, algorithm, n_clusters=None):
        """
        Cluster documents using the selected algorithm.
        """
        if algorithm == "K-Means":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
        elif algorithm == "DBSCAN":
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(embeddings)
        elif algorithm == "Hierarchical":
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = hierarchical.fit_predict(embeddings)
        elif algorithm == "GMM":
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = gmm.fit_predict(embeddings)
        else:
            raise ValueError("Unsupported clustering algorithm")
        return cluster_labels

    @staticmethod
    def assign_categories_to_clusters(cluster_labels, embeddings, predefined_categories):
        """
        Assign predefined categories to clusters based on cosine similarity.
        """
        category_embeddings = {category: vectorize_text_specter(category) for category in predefined_categories}
        cluster_to_category = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_id_int = int(cluster_id)
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            avg_cluster_embedding = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

            max_similarity = -1
            best_category = None
            for category, category_embedding in category_embeddings.items():
                category_embedding = np.array(category_embedding).reshape(1, -1)
                similarity = cosine_similarity(avg_cluster_embedding, category_embedding)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_category = category

            cluster_to_category[cluster_id_int] = best_category

        if len(unique_clusters) > len(predefined_categories):
            for cluster_id in unique_clusters:
                cluster_id_int = int(cluster_id)
                if cluster_id_int >= len(predefined_categories):
                    cluster_to_category[cluster_id_int] = f"Cluster {cluster_id_int + 1}"

        return cluster_to_category


# Visualization
class Visualizer:
    @staticmethod
    def get_distinct_colors(palette_name, n_colors):
        """
        Dynamically generate distinct colors for a given number of clusters.
        """
        if palette_name == "Rainbow":
            colors = list(mcolors.TABLEAU_COLORS.values())
        elif palette_name == "Nature":
            colors = list(mcolors.BASE_COLORS.values())
        elif palette_name == "Contrast":
            colors = list(mcolors.CSS4_COLORS.values())
        elif palette_name == "Bright":
            colors = list(mcolors.XKCD_COLORS.values())
        else:
            colors = list(mcolors.TABLEAU_COLORS.values())  # Default to Rainbow

        if n_colors > len(colors):
            colors = colors * (n_colors // len(colors)) + colors[:n_colors % len(colors)]

        return colors[:n_colors]

    @staticmethod
    def get_background_colors():
        """
        Return background color options.
        """
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
        """
        Create an interactive network graph using pyvis with custom options.
        """
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
        """
        Create a heatmap of document similarities.
        """
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
        """
        Create a parallel coordinates plot for reduced embeddings.
        """
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
        """
        Create a Sankey diagram to visualize document flow between clusters.
        """
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


# External API Integration
class ExternalAPIs:
    @staticmethod
    def fetch_pubmed_articles(query, max_results=10):
        """
        Fetch articles from PubMed based on a search query.
        """
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
        """
        Fetch details for a specific PubMed article by ID.
        """
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
    def fetch_arxiv_articles(query, max_results=10, category=None):
        """
        Fetch articles from arXiv based on a search query.
        """
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"{query}",
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        if category:
            params["search_query"] += f" AND cat:{category}"
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            feed = feedparser.parse(response.content)
            articles = []
            for entry in feed.entries:
                article = {
                    "title": entry.title,
                    "authors": [author.name for author in entry.authors],
                    "summary": entry.summary,
                    "published": entry.published,
                    "link": entry.link
                }
                articles.append(article)
            return articles
        else:
            st.error(f"Failed to fetch data from arXiv: {response.status_code}")
            return []

    @staticmethod
    def fetch_citation_count(title):
        """
        Fetch citation count for a paper using Google Scholar.
        """
        try:
            search_query = scholarly.search_pubs(title)
            publication = next(search_query)
            return publication.get("num_citations", 0)
        except Exception as e:
            st.error(f"Failed to fetch citation count: {str(e)}")
            return 0


# Semantic Search
class SemanticSearch:
    @staticmethod
    def semantic_search(df, query, top_k=5, similarity_threshold=0.5):
        """
        Perform semantic search on the DataFrame using SPECTER embeddings.

        Args:
            df (pd.DataFrame): DataFrame containing documents and their embeddings.
            query (str): The search query.
            top_k (int): Number of top results to return.
            similarity_threshold (float): Minimum similarity score for a document to be included in the results.

        Returns:
            pd.DataFrame: DataFrame containing the top_k most similar documents above the threshold.
        """
        # Filter out documents missing abstract, authors, or title
        filtered_df = df.dropna(subset=["abstract", "authors", "title"])

        # Check if there are any valid documents left
        if filtered_df.empty:
            st.warning("No valid documents found (missing abstract, authors, or title).")
            return pd.DataFrame()

        # Vectorize the query using SPECTER
        query_embedding = vectorize_text_specter(query)

        # Convert query_embedding to a NumPy array and reshape it
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Get document embeddings from the filtered DataFrame
        document_embeddings = np.array(filtered_df["embedding"].tolist())

        # Compute cosine similarity between the query and all documents
        similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

        # Add similarity scores to the filtered DataFrame
        filtered_df["similarity"] = similarities

        # Filter documents based on the similarity threshold
        thresholded_df = filtered_df[filtered_df["similarity"] >= similarity_threshold]

        # Check if any documents meet the threshold
        if thresholded_df.empty:
            st.warning(f"No documents found with similarity >= {similarity_threshold}.")
            return pd.DataFrame()

        # Sort the DataFrame by similarity in descending order
        df_sorted = thresholded_df.sort_values(by="similarity", ascending=False)

        # Return the top_k results
        return df_sorted.head(top_k)


# Query Preprocessing
def preprocess_query(query):
    """
    Preprocess the query by correcting misspellings and removing non-alphanumeric characters.

    Args:
        query (str): The search query.

    Returns:
        str: The preprocessed query.
    """
    # Correct misspellings
    spell = SpellChecker()
    corrected_query = " ".join([spell.correction(word) for word in query.split()])

    # Remove non-alphanumeric characters
    corrected_query = re.sub(r"[^a-zA-Z0-9\s]", "", corrected_query)

    return corrected_query


# Field Prediction
def get_most_likely_field(query):
    """
    Determine the most likely arXiv field for a given query using semantic similarity.

    Args:
        query (str): The search query.

    Returns:
        str: The most likely arXiv field.
    """
    # Define arXiv categories and their descriptions
    field_descriptions = {
        "q-bio.NC": "Neuroscience and neural systems.",
        "cs.CL": "Computational linguistics and natural language processing.",
        "physics.bio-ph": "Biological physics and biophysics.",
        "stat.ML": "Machine learning and statistical methods.",
        "q-bio.BM": "Biomolecules and molecular biology."
    }

    # Debug: Print the query
    st.write(f"Debug: Query = {query}")

    # Compute the query embedding
    query_embedding = vectorize_text_specter(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    # Debug: Print the query embedding shape
    st.write(f"Debug: Query embedding shape = {query_embedding.shape}")

    # Compute embeddings for each field description
    field_embeddings = {}
    for field, description in field_descriptions.items():
        field_embedding = vectorize_text_specter(description)
        field_embeddings[field] = np.array(field_embedding).reshape(1, -1)

        # Debug: Print the field description and embedding shape
        st.write(f"Debug: Field = {field}, Description = {description}")
        st.write(f"Debug: Field embedding shape = {field_embeddings[field].shape}")

    # Calculate cosine similarity between the query and each field
    similarities = {}
    for field, embedding in field_embeddings.items():
        similarity = cosine_similarity(query_embedding, embedding)[0][0]
        similarities[field] = similarity

        # Debug: Print the similarity score for each field
        st.write(f"Debug: Similarity for {field} = {similarity}")

    # Find the field with the highest similarity
    most_likely_field = max(similarities, key=similarities.get)

    # Debug: Print the most likely field
    st.write(f"Debug: Most likely field = {most_likely_field}")

    return most_likely_field

def calculate_inertia(embeddings, max_clusters=10):
    """
    Calculate inertia for different numbers of clusters using K-Means.
    """
    inertia_values = []
    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)
    return inertia_values

def calculate_silhouette_scores(embeddings, max_clusters=10):
    """
    Calculate silhouette scores for different numbers of clusters using K-Means.
    """
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):  # Silhouette score requires at least 2 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

# Main function
def main():
    st.title("Interactive Document Dashboard")
    st.sidebar.header("Filters and Settings")

    # Load data from ChromaDB
    df = DataLoader.load_data_from_chromadb()
    if df.empty:
        st.warning("No data found in ChromaDB. Please add documents first.")
        return

    # Year range filters
    st.sidebar.subheader("Filter by Year")
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    if min_year == max_year:
        st.sidebar.info(f"All documents are from year {min_year}")
        year_range = (min_year, min_year)
    else:
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            help="Filter documents by publication year."
        )

    # Filter data by year
    filtered_df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    # Define embeddings after filtering the data
    embeddings = np.array(filtered_df["embedding"].tolist())

    # Clustering algorithm selection
    clustering_algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical", "GMM"],
        help="Choose a clustering algorithm to group similar documents."
    )

    # Define n_clusters with actual slider value (if applicable)
    if clustering_algorithm in ["K-Means", "Hierarchical", "GMM"]:
        # Set a default max_clusters value for all algorithms that require it
        max_clusters = 10  # Default value for max_clusters

        # Calculate inertia and silhouette scores for all algorithms that require n_clusters
        inertia_values = calculate_inertia(embeddings, max_clusters)
        silhouette_scores = calculate_silhouette_scores(embeddings, max_clusters)

        # Display both Elbow Method and Silhouette Score plots
        st.sidebar.subheader("Cluster Optimization Methods")

        # Drop-down for informational tips
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

        # Plot the Elbow Method graph
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

        # Plot the Silhouette Scores graph
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

        # Let the user choose the number of clusters based on the selected method
        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=2,
            max_value=max_clusters,
            value=3,
            help="Select the number of clusters to group documents into."
        )
    else:
        n_clusters = None  # DBSCAN doesn't require n_clusters

    # Visualization settings in sidebar
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

    # Background color selection
    bg_colors = Visualizer.get_background_colors()
    bg_color_name = st.sidebar.selectbox(
        "Background Color",
        list(bg_colors.keys()),
        help="Choose the background color for the visualizations."
    )
    bg_color = bg_colors[bg_color_name]

    # Text color based on background
    is_dark_bg = bg_color in ["black", "#2d2d2d", "#001f3f", "#1a472a"]
    text_color = "white" if is_dark_bg else "black"
    grid_color = "gray" if is_dark_bg else "LightGray"

    # Dimensionality reduction selection
    st.sidebar.subheader("Dimensionality Reduction")
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

    # Perform dimensionality reduction
    reduced_embeddings = DimensionalityReducer.reduce_embeddings(embeddings, n_components=n_components,
                                                                 method=reduction_method)

    # Cluster documents using the selected algorithm
    cluster_labels = Clusterer.cluster_documents(embeddings, clustering_algorithm, n_clusters)

    # Debug: Print the number of unique clusters
    st.write(f"Number of unique clusters: {len(np.unique(cluster_labels))}")
    st.write("Unique Cluster Labels:", np.unique(cluster_labels))

    # Assign predefined categories to clusters based on cosine similarity
    cluster_to_category = Clusterer.assign_categories_to_clusters(cluster_labels, embeddings, PREDEFINED_CATEGORIES)
    st.write("Cluster to Category Mapping:", cluster_to_category)

    # Assign cluster names based on cluster labels
    filtered_df["cluster"] = [cluster_to_category[label] for label in cluster_labels]

    if n_components == 2:
        filtered_df["x"] = reduced_embeddings[:, 0]
        filtered_df["y"] = reduced_embeddings[:, 1]
    elif n_components == 3:
        filtered_df["x"] = reduced_embeddings[:, 0]
        filtered_df["y"] = reduced_embeddings[:, 1]
        filtered_df["z"] = reduced_embeddings[:, 2]

    # Enhanced visualization settings
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

    # Plot settings
    plot_settings = {
        "color": "cluster",
        "color_discrete_sequence": custom_colors,  # Use either custom or palette colors
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

    # Update plot layout with selected colors
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

    # Update axes
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

    # Interactive Network Graph
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

    # Create the network graph with custom colors
    net = Visualizer.create_network_graph(filtered_df, similarity_threshold, custom_colors)

    # Display the network graph in Streamlit
    with open("network.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)

    # Advanced Visualizations
    st.subheader("Advanced Visualizations")

    # Heatmap
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

    # Parallel Coordinates
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

    # Sankey Diagram
    st.subheader("Sankey Diagram")
    with st.expander("What is a Sankey Diagram?"):
        st.markdown("""
            **Sankey Diagram** visualizes the flow of documents between clusters.  
            The width of the links represents the number of documents flowing between clusters.  
            Use this diagram to understand how documents are distributed across clusters.
        """)

    sankey_fig = Visualizer.create_sankey_diagram(filtered_df)
    st.plotly_chart(sankey_fig, use_container_width=True)

    # Search Documents
    st.sidebar.subheader("Search Documents")
    search_query = st.sidebar.text_input(
        "Search by Title, Author, or Abstract",
        key="search_documents_input",  # Unique key
        help="Enter a query to search for documents by title, author, or abstract."
    )

    # Add a slider for similarity threshold
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,  # Default threshold
        step=0.1,
        help="Set the minimum similarity score for documents to be included in the results."
    )

    if search_query:
        # Preprocess the query
        corrected_query = preprocess_query(search_query)

        # Get the most likely field for the query
        most_likely_field = get_most_likely_field(corrected_query)

        # Perform semantic search using SPECTER with the similarity threshold
        search_results = SemanticSearch.semantic_search(
            filtered_df,
            corrected_query,
            top_k=5,
            similarity_threshold=similarity_threshold
        )

        # Display search results
        if not search_results.empty:
            st.subheader("Search Results")
            st.write(search_results[["title", "authors", "year", "abstract", "similarity"]])
        else:
            st.warning("No documents found for the search query above the similarity threshold.")

        # Fetch and display relevant articles from PubMed
        st.subheader("Related Articles from PubMed")
        pubmed_article_ids = ExternalAPIs.fetch_pubmed_articles(corrected_query, max_results=5)
        if pubmed_article_ids:
            for article_id in pubmed_article_ids:
                details = ExternalAPIs.fetch_pubmed_article_details(article_id)
                if details:
                    st.write(f"**Title:** {details.get('title', 'N/A')}")
                    st.write(f"**Authors:** {', '.join(details.get('authors', []))}")
                    st.write(f"**Abstract:** {details.get('abstract', 'N/A')}")
                    st.write("---")
        else:
            st.write("No related articles found in PubMed.")

        # Fetch and display relevant articles from arXiv
        st.subheader(f"Related Articles from arXiv ({most_likely_field})")
        arxiv_articles = ExternalAPIs.fetch_arxiv_articles(corrected_query, max_results=5, category=most_likely_field)
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

    # Citation Counts
    st.sidebar.subheader("Fetch Citation Count")
    citation_query = st.sidebar.text_input(
        "Enter a paper title to fetch citation count",
        key="citation_count_input",  # Unique key
        help="Enter the title of a paper to fetch its citation count from Google Scholar."
    )
    if citation_query:
        citation_count = ExternalAPIs.fetch_citation_count(citation_query)
        st.write(f"**Citation Count:** {citation_count}")

    # Scroll-down section to show clusters and their documents
    st.subheader("Cluster Details")
    with st.expander("View Documents in Each Cluster"):
        for cluster_name, group in filtered_df.groupby("cluster"):
            st.write(f"### Cluster: {cluster_name}")
            st.write(group[["title", "authors", "year", "abstract"]])


if __name__ == "__main__":
    main()