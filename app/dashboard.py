import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import feedparser
from scholarly import scholarly
import time
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
from pyvis.network import Network

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
        Reduce high-dimensional embeddings to 2D or 3D using PCA or t-SNE.
        """
        if method == "PCA":
            reducer = PCA(n_components=n_components)
        elif method == "t-SNE":
            reducer = TSNE(n_components=n_components, perplexity=3)
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
    def fetch_arxiv_articles(query, max_results=10):
        """
        Fetch articles from arXiv based on a search query.
        """
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
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

# Main function
def main():
    st.title("Interactive Document Dashboard")
    st.sidebar.header("Filters and Settings")

    # Load data from ChromaDB
    df = DataLoader.load_data_from_chromadb()
    if df.empty:
        st.warning("No data found in ChromaDB. Please add documents first.")
        return

    # Clustering algorithm selection
    clustering_algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical", "GMM"],
        help="Choose a clustering algorithm"
    )

    # Define n_clusters with actual slider value (if applicable)
    if clustering_algorithm in ["K-Means", "Hierarchical", "GMM"]:
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=50, value=3)
    else:
        n_clusters = None  # DBSCAN doesn't require n_clusters

    # Visualization settings in sidebar
    st.sidebar.subheader("Visualization Settings")
    use_custom_colors = st.sidebar.checkbox("Use Custom Node Colors")

    if use_custom_colors:
        custom_colors = []
        for i in range(n_clusters if n_clusters else 10):
            color = st.sidebar.color_picker(f"Cluster {i + 1} Color", "#0000FF")
            custom_colors.append(color)
    else:
        color_palette = st.sidebar.selectbox(
            "Node Color Scheme",
            ["Rainbow", "Nature", "Contrast", "Bright"],
            help="Choose a color scheme for the clusters"
        )
        custom_colors = Visualizer.get_distinct_colors(color_palette, n_clusters if n_clusters else 10)

    # Background color selection
    bg_colors = Visualizer.get_background_colors()
    bg_color_name = st.sidebar.selectbox(
        "Background Color",
        list(bg_colors.keys()),
        help="Choose the plot background color"
    )
    bg_color = bg_colors[bg_color_name]

    # Text color based on background
    is_dark_bg = bg_color in ["black", "#2d2d2d", "#001f3f", "#1a472a"]
    text_color = "white" if is_dark_bg else "black"
    grid_color = "gray" if is_dark_bg else "LightGray"

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
        )

    # Filter data by year
    filtered_df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    # Dimensionality reduction selection
    st.sidebar.subheader("Dimensionality Reduction")
    reduction_method = st.sidebar.selectbox("Select Method", ["PCA", "t-SNE"])
    n_components = st.sidebar.radio("Select Dimensions", [2, 3])

    embeddings = np.array(filtered_df["embedding"].tolist())
    reduced_embeddings = DimensionalityReducer.reduce_embeddings(embeddings, n_components=n_components, method=reduction_method)

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

    # Node customization (appears below "Number of Clusters")
    marker_size = st.sidebar.slider("Node Size", min_value=5, max_value=20, value=10)
    marker_opacity = st.sidebar.slider("Node Opacity", min_value=0.1, max_value=1.0, value=0.8, step=0.1)

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
    color_scheme = st.sidebar.selectbox(
        "Select Color Scheme",
        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
        help="Choose a color scheme for the visualizations."
    )
    parallel_coords_fig = Visualizer.create_parallel_coordinates(filtered_df, color_scheme=color_scheme)
    if parallel_coords_fig:
        st.plotly_chart(parallel_coords_fig, use_container_width=True)

    # Sankey Diagram
    st.subheader("Sankey Diagram")
    sankey_fig = Visualizer.create_sankey_diagram(filtered_df)
    st.plotly_chart(sankey_fig, use_container_width=True)

    # Search Documents
    st.sidebar.subheader("Search Documents")
    search_query = st.sidebar.text_input("Search by Title, Author, or Abstract")

    if search_query:
        # Perform semantic search using SPECTER
        search_results = SemanticSearch.semantic_search(filtered_df, search_query)

        # Display search results
        st.subheader("Search Results")
        st.write(search_results[["title", "authors", "year", "abstract", "similarity"]])

    # External API Integration
    st.sidebar.subheader("External API Integration")

    # PubMed Search
    st.sidebar.subheader("Search PubMed")
    pubmed_query = st.sidebar.text_input("Enter a PubMed search query")
    if pubmed_query:
        article_ids = ExternalAPIs.fetch_pubmed_articles(pubmed_query, max_results=5)
        if article_ids:
            st.write("### PubMed Results")
            for article_id in article_ids:
                details = ExternalAPIs.fetch_pubmed_article_details(article_id)
                if details:
                    st.write(f"**Title:** {details.get('title', 'N/A')}")
                    st.write(f"**Authors:** {', '.join(details.get('authors', []))}")
                    st.write(f"**Abstract:** {details.get('abstract', 'N/A')}")
                    st.write("---")

    # arXiv Search
    st.sidebar.subheader("Search arXiv")
    arxiv_query = st.sidebar.text_input("Enter an arXiv search query")
    if arxiv_query:
        articles = ExternalAPIs.fetch_arxiv_articles(arxiv_query, max_results=5)
        if articles:
            st.write("### arXiv Results")
            for article in articles:
                st.write(f"**Title:** {article['title']}")
                st.write(f"**Authors:** {', '.join(article['authors'])}")
                st.write(f"**Published:** {article['published']}")
                st.write(f"**Summary:** {article['summary']}")
                st.write(f"**Link:** [Read Paper]({article['link']})")
                st.write("---")

    # Citation Counts
    st.sidebar.subheader("Fetch Citation Count")
    citation_query = st.sidebar.text_input("Enter a paper title to fetch citation count")
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