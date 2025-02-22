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
        # Connect to ChromaDB
        chroma_client, collection = connect_to_chromadb()

        # Fetch all embeddings and metadata from the collection
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
    """
    Reduce high-dimensional embeddings to 3D using PCA.
    """
    print("Reducing embeddings to 3D using PCA...")
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    print("Reduction complete")
    return embeddings_3d

def cluster_embeddings(embeddings, n_clusters=5):
    """
    Cluster embeddings using K-Means.
    """
    print("Clustering embeddings using K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    print(f"Clustering complete. Created {n_clusters} clusters.")
    return cluster_labels

def assign_categories_to_clusters(cluster_labels, embeddings, predefined_categories):
    """
    Assign predefined categories to clusters based on cosine similarity.
    If there are more clusters than predefined categories, assign generic names.
    """
    # Compute embeddings for predefined categories
    category_embeddings = {category: vectorize_text_specter(category) for category in predefined_categories}

    # Map cluster IDs to their assigned categories
    cluster_to_category = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        # Get embeddings for documents in the current cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]

        # Compute the average embedding for the cluster
        avg_cluster_embedding = np.mean(cluster_embeddings, axis=0).reshape(1, -1)

        # Compare the average embedding to each category embedding
        max_similarity = -1
        best_category = None
        for category, category_embedding in category_embeddings.items():
            category_embedding = np.array(category_embedding).reshape(1, -1)
            similarity = cosine_similarity(avg_cluster_embedding, category_embedding)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_category = category

        # Assign the best category to the cluster
        cluster_to_category[cluster_id] = best_category

    # If there are more clusters than predefined categories, assign generic names
    if len(unique_clusters) > len(predefined_categories):
        for cluster_id in unique_clusters:
            if cluster_id >= len(predefined_categories):
                cluster_to_category[cluster_id] = f"Cluster {cluster_id}"

    return cluster_to_category

def plot_interactive_3d_with_categories(embeddings_3d, titles, cluster_labels, cluster_to_category):
    """
    Create an interactive 3D plot using Plotly with clusters labeled by assigned categories.
    """
    print("Creating interactive 3D plot with categories...")

    # Create a scatter plot for each cluster
    fig = go.Figure()

    for cluster_id in np.unique(cluster_labels):
        # Filter points belonging to the current cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings_3d[cluster_mask]
        cluster_titles = [titles[i] for i in np.where(cluster_mask)[0]]

        # Get the category name for the cluster
        category_name = cluster_to_category.get(cluster_id, f"Cluster {cluster_id}")

        # Add a scatter trace for the cluster
        fig.add_trace(go.Scatter3d(
            x=cluster_embeddings[:, 0],
            y=cluster_embeddings[:, 1],
            z=cluster_embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=10,  # Larger size for clusters
                opacity=0.8
            ),
            text=cluster_titles,  # Add titles as hover text
            hoverinfo='text',  # Show only the title on hover
            name=category_name  # Use assigned category as label
        ))

    # Update layout for better visualization
    fig.update_layout(
        title="Interactive 3D Visualization of Document Embeddings with Categories",
        scene=dict(
            xaxis_title='PCA Component 1',
            yaxis_title='PCA Component 2',
            zaxis_title='PCA Component 3'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    # Show the plot
    fig.show()

def main():
    try:
        # Fetch embeddings, metadata, and IDs from ChromaDB
        embeddings, titles, ids = fetch_embeddings_and_metadata_from_chromadb()

        # Reduce embeddings to 3D
        embeddings_3d = reduce_to_3d(embeddings)

        # Cluster embeddings into categories (e.g., "biology," "physics")
        n_clusters = 10  # Adjust n_clusters as needed
        cluster_labels = cluster_embeddings(embeddings, n_clusters=n_clusters)

        # Assign predefined categories to clusters based on cosine similarity
        cluster_to_category = assign_categories_to_clusters(cluster_labels, embeddings, PREDEFINED_CATEGORIES)

        # Plot the 3D embeddings interactively with assigned categories
        plot_interactive_3d_with_categories(embeddings_3d, titles, cluster_labels, cluster_to_category)

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()