import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from database_connection import connect_to_chromadb
from extract_data import extract_details
from config.constants import PDF_PATH
import logging
from tqdm import tqdm
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vectorization.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting application...")

extracted_details_cache = {}
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    global model, tokenizer

    if model is None or tokenizer is None:
        logger.info(f"Loading SPECTER model and tokenizer on {device}...")
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = AutoModel.from_pretrained('allenai/specter').to(device)
        logger.info("SPECTER model loaded successfully")

    return model, tokenizer


@lru_cache(maxsize=128)
def vectorize_text_specter(text):
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for vectorization")
        return [0.0] * 768
    try:
        model, tokenizer = load_model()
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')
        if len(text) > 10000:
            logger.warning(f"Very long text ({len(text)} chars) truncated to 10000 chars")
            text = text[:10000]
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        with torch.no_grad():
            model.eval()
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()

        if len(embeddings.shape) == 0:
            embeddings = np.array([embeddings])
        elif len(embeddings.shape) > 1:
            embeddings = embeddings[0]

        result = embeddings.tolist()

        return result

    except Exception as e:
        logger.error(f"Error in vectorization: {str(e)}", exc_info=True)
        return [0.0] * 768


def process_documents_in_batches(details, directory_path, collection, batch_size=100):
    total_docs = len(details)
    total_batches = (total_docs + batch_size - 1) // batch_size

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    with tqdm(total=total_docs, desc="Processing documents") as pbar:
        for filename, info in details.items():
            try:
                if not info['title'] and not info['abstract'] and not info['authors']:
                    logger.warning(f"Skipping document with no content: {filename}")
                    pbar.update(1)
                    continue

                combined_text = f"{info['title']} {info['abstract']} {info['authors']}".strip()
                embedding = vectorize_text_specter(combined_text)
                pdf_path = os.path.join(directory_path, filename)

                existing_document = collection.get(ids=[filename])
                if existing_document and len(existing_document["ids"]) > 0:
                    collection.update(
                        ids=[filename],
                        embeddings=[embedding],
                        metadatas=[{
                            "filename": filename,
                            "filepath": pdf_path,
                            "title": info['title'],
                            "authors": info['authors'],
                            "abstract": info['abstract'],
                            "year": info.get('year', 2023)
                        }]# Výchozí hodnota roku 2023 je použita v případě,
                    )     # že v textu není detekován konkrétní rok publikace
                else:
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

                if len(documents) >= batch_size:
                    collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added batch of {len(documents)} documents to collection")
                    documents, embeddings, metadatas, ids = [], [], [], []

                pbar.update(1)

            except Exception as e:
                logger.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
                pbar.update(1)

    if documents:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added final batch of {len(documents)} documents to collection")

    return collection.count()

def add_documents_to_collection(directory_path, collection=None):
    logger.info(f"Processing directory: {directory_path}")
    try:
        if directory_path in extracted_details_cache:
            logger.info("Using cached details for directory")
            details = extracted_details_cache[directory_path]
        else:
            logger.info("Extracting details from PDFs...")
            details = extract_details(directory_path)
            extracted_details_cache[directory_path] = details
            logger.info(f"Extracted details from {len(details)} PDFs")

        if collection is None:
            _, collection = connect_to_chromadb()

        total_count = process_documents_in_batches(details, directory_path, collection)
        logger.info(f"Total documents in collection after processing: {total_count}")

        return total_count

    except Exception as e:
        logger.error(f"Error in add_documents_to_collection: {str(e)}", exc_info=True)
        raise

    except Exception as e:
        logger.error(f"Error in add_documents_to_collection: {str(e)}", exc_info=True)
        raise


def main():
    logger.info("=== Starting main execution ===")
    try:
        load_model()

        directory_path = PDF_PATH
        logger.info(f"Processing directory: {directory_path}")

        if not os.path.exists(directory_path):
            logger.error(f"Directory not found: {directory_path}")
            return

        total_count = add_documents_to_collection(directory_path)

        logger.info(f"Total documents in collection: {total_count}")
        logger.info("=== Execution completed successfully ===")

    except Exception as e:
        logger.error(f"ERROR in main execution: {str(e)}", exc_info=True)
        logger.error("=== Execution failed ===")

    finally:
        if model is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")


if __name__ == "__main__":
    main()