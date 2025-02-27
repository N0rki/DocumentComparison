import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from vectorization import vectorize_text_specter
import nltk
from nltk.corpus import wordnet
from database_connection import connect_to_chromadb

# Download WordNet data
nltk.download('wordnet')

def expand_query(query):
    """
    Expand a query using synonyms from WordNet.

    Args:
        query (str): The input query.

    Returns:
        str: The expanded query.
    """
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return query + " " + " ".join(synonyms)


def query_similar_documents(query_text, top_k=5):
    """
    Query similar documents from the ChromaDB collection using precomputed embeddings.

    Args:
        query_text (str): The input query.
        top_k (int): Number of similar documents to retrieve.

    Returns:
        dict: Query results containing documents, metadata, and distances.
    """
    try:
        chroma_client, collection = connect_to_chromadb()

        expanded_query = expand_query(query_text)
        print(f"Expanded query: {expanded_query}")

        print("Generating query embedding...")
        query_embedding = vectorize_text_specter(expanded_query)

        print(f"Searching for top {top_k} similar documents...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        print("Search completed successfully")

        return results

    except Exception as e:
        print(f"Error in query_similar_documents: {str(e)}")
        raise


def main():
    print("\n=== Starting query execution ===")
    try:
        query_text = "computational biology"
        print(f"\nPerforming query: '{query_text}'")

        results = query_similar_documents(query_text)

        print("\n=== Query Results ===")
        for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        )):
            print(f"\nResult {i + 1} (Distance: {distance:.4f}):")
            print(f"Title: {metadata['title']}")
            print(f"Authors: {metadata['authors']}")
            print(f"Abstract: {metadata['abstract'][:200]}...")

        print("\n=== Query execution completed successfully ===")

    except Exception as e:
        print(f"\nERROR in query execution: {str(e)}")
        print("=== Query execution failed ===")
        raise

if __name__ == "__main__":
    main()