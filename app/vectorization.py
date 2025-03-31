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

# Configure logging
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

# Global variables
extracted_details_cache = {}
model = None
tokenizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model():
    """Load SPECTER model and tokenizer with lazy initialization."""
    global model, tokenizer

    if model is None or tokenizer is None:
        logger.info(f"Loading SPECTER model and tokenizer on {device}...")
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        model = AutoModel.from_pretrained('allenai/specter').to(device)
        logger.info("SPECTER model loaded successfully")

    return model, tokenizer


@lru_cache(maxsize=128)
def vectorize_text_specter(text):
    """Vectorize text using SPECTER model with caching for repeated texts."""
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for vectorization")
        # Return zero vector of correct dimension (768 for SPECTER)
        return [0.0] * 768

    try:
        model, tokenizer = load_model()

        # Ensure text is properly encoded as string
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='ignore')

        # Truncate extremely long texts before tokenization to avoid memory issues
        if len(text) > 100000:
            logger.warning(f"Very long text ({len(text)} chars) truncated to 100000 chars")
            text = text[:100000]

        # Tokenize with proper handling
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        # Process in evaluation mode for efficiency
        with torch.no_grad():
            model.eval()
            outputs = model(**inputs)

        # Get embeddings from the [CLS] token
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().squeeze().numpy()

        # Handle different dimensionalities
        if len(embeddings.shape) == 0:  # scalar
            embeddings = np.array([embeddings])
        elif len(embeddings.shape) > 1:  # batch
            embeddings = embeddings[0]  # take first embedding

        # Convert to list for storage
        result = embeddings.tolist()

        return result

    except Exception as e:
        logger.error(f"Error in vectorization: {str(e)}", exc_info=True)
        # Return zero vector as fallback
        return [0.0] * 768


def process_documents_in_batches(details, directory_path, batch_size=100):
    """Process documents in batches to improve memory efficiency."""
    chroma_client, collection = connect_to_chromadb()

    total_docs = len(details)
    total_batches = (total_docs + batch_size - 1) // batch_size  # Ceiling division

    documents = []
    embeddings = []
    metadatas = []
    ids = []

    with tqdm(total=total_docs, desc="Processing documents") as pbar:
        for filename, info in details.items():
            try:
                # Skip empty or severely corrupted documents
                if not info['title'] and not info['abstract'] and not info['authors']:
                    logger.warning(f"Skipping document with no content: {filename}")
                    pbar.update(1)
                    continue

                # Prepare combined text for embedding
                combined_text = f"{info['title']} {info['abstract']} {info['authors']}".strip()

                # Generate embedding
                embedding = vectorize_text_specter(combined_text)

                pdf_path = os.path.join(directory_path, filename)

                # Check if document already exists in collection
                existing_document = collection.get(ids=[filename])
                if existing_document and len(existing_document["ids"]) > 0:
                    # Update existing document
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
                        }]
                    )
                else:
                    # Add to batch for new document
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

                # Process batch if it reaches the desired size
                if len(documents) >= batch_size:
                    collection.add(
                        documents=documents,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                    logger.info(f"Added batch of {len(documents)} documents to collection")

                    # Clear batch data
                    documents = []
                    embeddings = []
                    metadatas = []
                    ids = []

                pbar.update(1)

            except Exception as e:
                logger.error(f"Error processing document {filename}: {str(e)}", exc_info=True)
                pbar.update(1)

    # Add any remaining documents
    if documents:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        logger.info(f"Added final batch of {len(documents)} documents to collection")

    return collection.count()


def add_documents_to_collection(directory_path):
    """Extract details and add documents to the collection."""
    logger.info(f"Processing directory: {directory_path}")
    try:
        # Check cache for extracted details
        if directory_path in extracted_details_cache:
            logger.info("Using cached details for directory")
            details = extracted_details_cache[directory_path]
        else:
            logger.info("Extracting details from PDFs...")
            details = extract_details(directory_path)
            extracted_details_cache[directory_path] = details
            logger.info(f"Extracted details from {len(details)} PDFs")

        # Process documents in optimized batches
        total_count = process_documents_in_batches(details, directory_path)
        logger.info(f"Total documents in collection after processing: {total_count}")

        return total_count

    except Exception as e:
        logger.error(f"Error in add_documents_to_collection: {str(e)}", exc_info=True)
        raise


def main():
    """Main execution function with better error handling."""
    logger.info("=== Starting main execution ===")
    try:
        # Load the model at startup
        load_model()

        directory_path = PDF_PATH
        logger.info(f"Processing directory: {directory_path}")

        # Verify directory exists
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
        # Free up GPU memory if using CUDA
        if model is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache cleared")


if __name__ == "__main__":
    main()