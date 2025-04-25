import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from extract_data import extract_details

specter_tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
specter_model = AutoModel.from_pretrained("allenai/specter")
specter_model = specter_model.to("cuda" if torch.cuda.is_available() else "cpu")
specter_model.eval()

def specter_embed(text):
    inputs = specter_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(specter_model.device)
    with torch.no_grad():
        output = specter_model(**inputs)
        return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

def process_pdfs_to_specter_json(pdf_folder, output_path="fused_documents.json"):
    doc_dict = extract_details(pdf_folder)
    fused = []

    for filename, meta in tqdm(doc_dict.items()):
        title = meta.get("title", "").strip()
        abstract = meta.get("abstract", "").strip()
        text = title + " " + abstract
        year = meta.get("year", "unknown")
        authors = meta.get("authors", "")

        try:
            embedding = specter_embed(text)
        except Exception as e:
            print(f"❌ Failed to embed {filename}: {e}")
            continue

        fused.append({
            "id": title,
            "title": title,
            "year": year,
            "authors": authors,
            "abstract": abstract,
            "specter_embedding": embedding.tolist()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fused, f, indent=2)
    print(f"✅ Saved {len(fused)} documents to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_specter_only.py path/to/pdf_folder")
    else:
        process_pdfs_to_specter_json(sys.argv[1])