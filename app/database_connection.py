import chromadb
from chromadb.config import Settings

def connect_to_chromadb(host="localhost", port=8000):
    """
    Connect to ChromaDB and return the collection.

    Args:
        host (str): ChromaDB server host.
        port (int): ChromaDB server port.

    Returns:
        tuple: A tuple containing:
            - chroma_client: The ChromaDB client.
            - collection: The ChromaDB collection.
    """
    try:
        print("Connecting to ChromaDB API...")
        chroma_client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_api_impl="rest",
                chroma_server_ssl_enabled=False
            )
        )
        print("Successfully connected to ChromaDB API")

        print("Creating/loading collection 'research_documents'...")
        collection = chroma_client.get_or_create_collection(name="research_documents")
        print(f"Collection ready. Current count: {collection.count()}")

        return chroma_client, collection

    except Exception as e:
        print(f"Error during ChromaDB connection: {str(e)}")
        raise