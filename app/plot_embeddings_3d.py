import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objs as go
from config.science_categories import PREDEFINED_CATEGORIES
from database_connection import connect_to_chromadb
from vectorization import vectorize_text_specter


def fetch_embeddings_and_metadata_from_chromadb(collection_name="research_documents"):
    """
    Fetch embeddings and metadata (including titles) from a ChromaDB collection.
    """
    try:
        chroma_client, collection = connect_to_chromadb()

        print("Fetching embeddings and metadata...")
        results = collection.get(include=['embeddings', 'metadatas'])
        embeddings = np.array(results['embeddings'])
        titles = [metadata.get('title', 'Untitled') for metadata in results['metadatas']]
        ids = results['ids']
        print(f"Fetched {len(embeddings)} embeddings and {len(titles)} titles")

        return embeddings, titles, ids

    except Exception as e:
        print(f"Error fetching embeddings and metadata: {str(e)}")
        raise


def reduce_to_3d(embeddings):
    print("Reducing embeddings to 3D using PCA...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    print("Reduction complete")
    return embeddings_3d


def cluster_embeddings(embeddings, n_clusters=5):
    print("Clustering embeddings using K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    print(f"Clustering complete. Created {n_clusters} clusters.")
    return cluster_labels


def assign_categories_to_clusters(cluster_labels, embeddings, predefined_categories):

    category_embeddings = {category: vectorize_text_specter(category) for category in predefined_categories}

    cluster_to_category = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
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

        cluster_to_category[cluster_id] = best_category

    if len(unique_clusters) > len(predefined_categories):
        for cluster_id in unique_clusters:
            if cluster_id >= len(predefined_categories):
                cluster_to_category[cluster_id] = f"Cluster {cluster_id}"

    return cluster_to_category


def plot_interactive_3d_with_categories(embeddings_3d, titles, cluster_labels, cluster_to_category):

    print("Creating interactive 3D plot with categories...")

    fig = go.Figure()

    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings_3d[cluster_mask]
        cluster_titles = [titles[i] for i in np.where(cluster_mask)[0]]

        category_name = cluster_to_category.get(cluster_id, f"Cluster {cluster_id}")

        fig.add_trace(go.Scatter3d(
            x=cluster_embeddings[:, 0],
            y=cluster_embeddings[:, 1],
            z=cluster_embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.8
            ),
            text=cluster_titles,
            hoverinfo='text',
            name=category_name
        ))

    fig.update_layout(
        title="Interactive 3D Visualization of Document Embeddings with Categories",
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    fig.show()


def main():
    try:
        embeddings, titles, ids = fetch_embeddings_and_metadata_from_chromadb()

        embeddings_3d = reduce_to_3d(embeddings)

        n_clusters = 10
        cluster_labels = cluster_embeddings(embeddings, n_clusters=n_clusters)

        cluster_to_category = assign_categories_to_clusters(cluster_labels, embeddings, PREDEFINED_CATEGORIES)

        plot_interactive_3d_with_categories(embeddings_3d, titles, cluster_labels, cluster_to_category)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
