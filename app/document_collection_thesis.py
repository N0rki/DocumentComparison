import json
from database_connection import connect_to_chromadb
import numpy as np

client, collection = connect_to_chromadb()

results = collection.get(include=["embeddings", "metadatas", "documents"], limit=1)

if results["ids"]:
    output = {
        "id": results["ids"][0],
        "embedding": results["embeddings"][0].tolist(),
        "metadata": results["metadatas"][0]
    }

    with open("output.txt", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("First entry written to output.txt.")
else:
    print("No documents found in the collection.")
