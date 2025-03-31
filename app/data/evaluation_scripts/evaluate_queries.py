
import sys
import os
import fitz  # PyMuPDF

# Path fix to allow local imports if needed
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

PDF_DIR = os.path.join(BASE_DIR, 'documents', 'three_categories')

def load_pdf_texts(pdf_dir=PDF_DIR):
    texts, filenames = [], []
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            try:
                path = os.path.join(pdf_dir, filename)
                doc = fitz.open(path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                doc.close()
                texts.append(full_text.strip())
                filenames.append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return texts, filenames

from query import query_similar_documents

ground_truth = {
    "graph neural networks": ["paper1.pdf", "gnn_study_2021.pdf", "deep_gnn_theory.pdf"],
    "protein folding": ["protein_structures_2022.pdf", "bio_folding_ai.pdf"]
}

def precision_at_k(retrieved, relevant, k=5):
    hits = [doc for doc in retrieved[:k] if doc in relevant]
    return len(hits) / k

def recall_at_k(retrieved, relevant, k=5):
    hits = [doc for doc in retrieved[:k] if doc in relevant]
    return len(hits) / len(relevant)

def evaluate_queries():
    for query_text, relevant_ids in ground_truth.items():
        print(f"\n--- Evaluating query: '{query_text}' ---")
        results = query_similar_documents(query_text, top_k=5)
        retrieved_ids = results['ids'][0]
        p = precision_at_k(retrieved_ids, relevant_ids)
        r = recall_at_k(retrieved_ids, relevant_ids)
        print(f"Precision@5: {p:.2f}")
        print(f"Recall@5:    {r:.2f}")

if __name__ == "__main__":
    evaluate_queries()
