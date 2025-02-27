import numpy as np
import chromadb
from chromadb.config import Settings

def verify_collection(collection_name="research_documents", host="localhost", port=8000):
    """
    Verify that the collection was created and updated successfully in ChromaDB.

    Args:
        collection_name (str): Name of the collection to verify.
        host (str): ChromaDB server host.
        port (int): ChromaDB server port.
    """
    try:
        chroma_client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_api_impl="rest",
                chroma_server_ssl_enabled=False
            )
        )
        print("Successfully connected to ChromaDB API")

        print(f"Loading collection '{collection_name}'...")
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' loaded successfully")

        print(f"\nTotal documents in collection: {collection.count()}")

        print("\nQuerying a random document from the collection...")
        random_embedding = np.random.rand(768).tolist()  # Random embedding for testing
        results = collection.query(
            query_embeddings=[random_embedding],
            n_results=1,
            include=['documents', 'metadatas', 'distances']
        )
        print("Sample document from collection:")
        print(f"Document: {results['documents'][0][0]}")
        print(f"Metadata: {results['metadatas'][0][0]}")
        print(f"Distance: {results['distances'][0][0]:.4f}")

        print("\nRetrieving metadata for the first 5 documents...")
        first_five_ids = [f"doc_{i}" for i in range(5)]
        metadata_results = collection.get(ids=first_five_ids, include=['metadatas'])
        print("Metadata for first 5 documents:")
        for metadata in metadata_results['metadatas']:
            print(metadata)

        print("\nRetrieving embeddings for the first 5 documents...")
        embedding_results = collection.get(ids=first_five_ids, include=['embeddings'])
        print("Embeddings for first 5 documents:")
        for embedding in embedding_results['embeddings']:
            print(f"Embedding length: {len(embedding)}")

        print("\n=== Collection verification completed successfully ===")

    except Exception as e:
        print(f"\nERROR during collection verification: {str(e)}")
        print("=== Collection verification failed ===")
        raise


if __name__ == "__main__":
    verify_collection()