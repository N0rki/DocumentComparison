import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from database_connection import connect_to_chromadb
from extract_data import extract_details
from config.constants import PDF_PATH

print("Starting application...")

extracted_details_cache = {}

chroma_client, collection = connect_to_chromadb()

print("Loading SPECTER model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')
print("SPECTER model loaded successfully")


def vectorize_text_specter(text):
    print(f"Vectorizing text (length: {len(text)} characters)...")
    try:
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        print(f"Text tokenized. Input shape: {inputs['input_ids'].shape}")

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

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
        if directory_path in extracted_details_cache:
            print("Using cached details for directory.")
            details = extracted_details_cache[directory_path]
        else:
            print("Extracting details from PDFs...")
            details = extract_details(directory_path)
            extracted_details_cache[directory_path] = details
            print(f"Extracted details from {len(details)} PDFs")

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

            pdf_path = os.path.join(directory_path, filename)

            documents.append(combined_text)
            embeddings.append(embedding)
            metadatas.append({
                "filename": filename,
                "filepath": pdf_path,
                "title": info['title'],
                "authors": info['authors'],
                "abstract": info['abstract'],
                "year": info.get('year', 2023)
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