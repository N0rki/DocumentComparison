import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from extract_data import extract_details

sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
specter_tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
specter_model = AutoModel.from_pretrained('allenai/specter').to('cuda' if torch.cuda.is_available() else 'cpu')

def sbert_embed(text):
    return sbert_model.encode(text, show_progress_bar=False)

def specter_embed(text):
    inputs = specter_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(specter_model.device)
    with torch.no_grad():
        output = specter_model(**inputs)
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def build_metadata_vector(doc, all_categories, year_range=(2000, 2025)):
    vector = []
    year = int(doc.get("year", 2020)) if str(doc.get("year", "")).isdigit() else 2020
    norm_year = (year - year_range[0]) / (year_range[1] - year_range[0])
    vector.append(norm_year)

    authors = doc.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    author_count = len(authors)
    vector.append(min(author_count / 10.0, 1.0))

    category_vector = [1.0 if doc.get("category") == c else 0.0 for c in all_categories]
    vector.extend(category_vector)
    return np.array(vector)

def extract_and_fuse_embeddings(pdf_folder, output_path):
    print(f"📂 Extracting from folder: {pdf_folder}")
    document_dict = extract_details(pdf_folder)
    documents = [v for v in document_dict.values()]
    all_categories = sorted(list({doc.get("category", "") for doc in documents}))

    fused = []
    for doc in tqdm(documents):
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        text = title + " " + abstract

        doc["sbert_embedding"] = sbert_embed(text).tolist()
        doc["specter_embedding"] = specter_embed(text).tolist()
        doc["metadata_vector"] = build_metadata_vector(doc, all_categories).tolist()

        fused.append(doc)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fused, f, indent=2)
    print(f"✅ Saved fused embeddings to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python embedding_fusion_from_pdfs.py path/to/pdf_folder output.json")
    else:
        extract_and_fuse_embeddings(sys.argv[1], sys.argv[2])