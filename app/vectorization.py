import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from database_connection import connect_to_chromadb
from extract_data import extract_details
from config.constants import PDF_PATH

print("Starting application...")

# Cache for storing extracted details
extracted_details_cache = {}

# Connect to ChromaDB
chroma_client, collection = connect_to_chromadb()

# Load SPECTER model and tokenizer
print("Loading SPECTER model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')
print("SPECTER model loaded successfully")


def vectorize_text_specter(text):
    print(f"Vectorizing text (length: {len(text)} characters)...")
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        print(f"Text tokenized. Input shape: {inputs['input_ids'].shape}")

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the [CLS] token embedding as the document embedding
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Convert to list for JSON serialization
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        print(f"Vectorization successful. Embedding length: {len(embeddings)}")
        return embeddings

    except Exception as e:
        print(f"Error in vectorization: {str(e)}")
        raise


def add_documents_to_collection(directory_path):
    print(f"\nProcessing directory: {directory_path}")
    try:
        # Extract details from all PDFs in the directory
        if directory_path in extracted_details_cache:
            print("Using cached details for directory.")
            details = extracted_details_cache[directory_path]
        else:
            print("Extracting details from PDFs...")
            details = extract_details(directory_path)
            extracted_details_cache[directory_path] = details  # Cache the details
            print(f"Extracted details from {len(details)} PDFs")

        # Process documents in batches
        batch_size = 100
        documents = []
        embeddings = []
        metadatas = []
        ids = []

        for filename, info in details.items():
            print(f"\nProcessing document: {filename}")
            combined_text = f"{info['title']} {info['abstract']} {info['authors']}"
            print(f"Combined text length: {len(combined_text)} characters")

            print("Generating embedding...")
            embedding = vectorize_text_specter(combined_text)

            documents.append(combined_text)
            embeddings.append(embedding)
            metadatas.append({
                "filename": filename,
                "title": info['title'],
                "authors": info['authors'],
                "abstract": info['abstract']
            })
            ids.append(filename)

            print(f"Current batch size: {len(documents)}/{batch_size}")

            if len(documents) >= batch_size:
                print(f"Adding batch of {len(documents)} documents to collection...")
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                print("Batch added successfully")
                documents = []
                embeddings = []
                metadatas = []
                ids = []

        if documents:
            print(f"Adding final batch of {len(documents)} documents to collection...")
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print("Final batch added successfully")

    except Exception as e:
        print(f"Error in add_documents_to_collection: {str(e)}")
        raise


def main():
    print("\n=== Starting main execution ===")
    try:
        # Example usage
        directory_path = PDF_PATH
        print(f"Processing directory: {directory_path}")

        add_documents_to_collection(directory_path)

        print(f"\nTotal documents in collection: {collection.count()}")

    except Exception as e:
        print(f"\nERROR in main execution: {str(e)}")
        print("=== Execution failed ===")
        raise


if __name__ == "__main__":
    main()