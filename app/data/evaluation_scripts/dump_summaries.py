
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

from summarizer import summarize_text
import json

def main():
    texts, filenames = load_pdf_texts()
    summaries = {}
    for i, text in enumerate(texts[:10]):
        summary = summarize_text(text)
        summaries[filenames[i]] = summary
    with open("summaries_for_review.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("Saved 10 summaries for manual evaluation.")

if __name__ == "__main__":
    main()
