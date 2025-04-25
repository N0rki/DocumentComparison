import os
import re
import string
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import PyPDF4
import spacy
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

from config.constants import PDF_PATH


def sanitize(filename):
    """Remove invalid characters from filename."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join(c for c in filename if c in valid_chars)


def sanitize_authors(authors):
    """Clean up author text by removing digits and newlines."""
    if not authors:
        return ""
    authors = authors.replace('\n', '')
    return ''.join(c for c in authors if not c.isdigit())


def remove_email_addresses(text):
    """Remove email addresses from text."""
    if not text:
        return ""
    return re.sub(r'[\w\.-]+@[\w\.-]+', '', text)


def empty_str(s):
    """Check if a string is empty."""
    if s is None:
        return True
    return len(str(s).strip()) == 0


def copyright_line(line):
    """Check if line contains copyright-related text."""
    return bool(re.search(r'technical\s+report|proceedings|preprint|to\s+appear|submission', line.lower()))


def valid_title(title):
    """Check if title is valid."""
    if title is None:
        return False
    if isinstance(title, (str, bytes)):
        return len(str(title).strip()) > 0
    return False


@lru_cache(maxsize=32)
def metadata(filename):
    """Extract metadata from PDF file with caching."""
    try:
        with open(filename, 'rb') as file:
            reader = PyPDF4.PdfFileReader(file)
            docinfo = reader.getDocumentInfo()
            if docinfo is None:
                return {}
            return {k: str(v) for k, v in docinfo.items()}
    except Exception:
        return {}


@lru_cache(maxsize=32)
def pdf_text(filename, max_pages=2):
    """Extract text from PDF with caching and page limit."""
    try:
        pdf = fitz.open(filename)
        text = ""
        for page_num in range(min(max_pages, len(pdf))):
            page = pdf.load_page(page_num)
            text += page.get_text()
        pdf.close()
        return text
    except Exception:
        try:
            return extract_text(filename)
        except (PDFSyntaxError, Exception):
            return ""


def text_title(filename):
    lines = pdf_text(filename).strip().split('\n')
    ignore_keywords = {"abstract", "introduction", "keywords", "proceedings", "submission", "preprint"}
    lines = [line.strip() for line in lines if line.strip()]

    def is_valid(line):
        lowered = line.lower()
        return (
                len(line) >= 20
                and not any(c in line for c in "@0123456789")
                and not any(k in lowered for k in ignore_keywords)
        )

    valid_lines = [line for line in lines[:6] if is_valid(line)]

    if len(valid_lines) >= 2 and valid_lines[0][-1] != '.':
        title = f"{valid_lines[0]} {valid_lines[1]}"
    elif valid_lines:
        title = valid_lines[0]
    else:
        fallback_candidates = [line for line in lines[:10] if not any(k in line.lower() for k in ignore_keywords)]
        title = max(fallback_candidates, key=len, default="Title not found")

    try:
        title_index = lines.index(valid_lines[0] if valid_lines else title)
        author_lines = [
            lines[i].strip() for i in range(title_index + 1, title_index + 6)
            if i < len(lines) and not empty_str(lines[i])
        ]
        authors = ' '.join(author_lines)
    except Exception:
        authors = ""

    return title, sanitize_authors(authors)


def pdf_title(filename):
    """Get PDF title from metadata or text."""
    try:
        meta = metadata(filename)
        title = meta.get('/Title', "")
        if valid_title(title):
            return title

        title, _ = text_title(filename)
        if valid_title(title):
            return title

        return os.path.basename(os.path.splitext(filename)[0])
    except Exception as e:
        print(f"Error getting title for {filename}: {str(e)}")
        return os.path.basename(os.path.splitext(filename)[0])


_nlp = None


def get_nlp():
    """Singleton pattern for loading spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_trf")
    return _nlp


def extract_abstract(pdf_path):
    """Extract abstract from PDF."""
    try:
        pdf_document = fitz.open(pdf_path)
        if pdf_document.is_encrypted:
            pdf_document.close()
            return "Nothing was found (encrypted document)"

        text = ""
        for page_num in range(min(2, len(pdf_document))):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        abstract_patterns = [
            r'(?s)(?<=\bAbstract\b)\s*(.*?)(?=\b(?:\d*\s*)?(?:Introduction|1\.)\b)',
            r'(?s)(?<=\bA b s t r a c t\b)\s*(.*?)(?=\b(?:\d*\s*)?(?:Introduction|1\.)\b)',
            r'(?s)(?<=\bABSTRACT\b)\s*(.*?)(?=\b(?:\d*\s*)?(?:Introduction|1\.)\b)'
        ]
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return "Nothing was found"
    except Exception as e:
        print(f"Error extracting abstract from {pdf_path}: {str(e)}")
        return "Nothing was found (error)"


def find_persons_locations(filename, title):
    """Extract persons and locations from PDF."""
    try:
        nlp = get_nlp()
        text = pdf_text(filename, max_pages=2)

        lines = text.split('\n')
        abstract_index = next((i for i, line in enumerate(lines)
                               if "abstract" in line.lower() or "a b s t r a c t" in line.lower()), len(lines))

        first_lines = [line.strip() for line in lines[:abstract_index] if line.strip()]
        sanitized_text = '\n'.join(first_lines)

        doc = nlp(sanitized_text)

        persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and "ORG" not in ent.text]
        locations = [ent.text.strip().replace(".", "") for ent in doc.ents
                     if ent.label_ == "GPE" and ent.text.strip().replace
                     (".", "").replace(" ", "").isalpha()]

        persons_text = sanitize_authors(', '.join(persons))
        persons_text = remove_email_addresses(persons_text)

        person_list = [name.strip() for name in persons_text.split(',') if name.strip()]
        title_parts = title.lower().split()

        filtered_persons = []
        for person in person_list:
            person_words = person.lower().split()
            if not any(' '.join(title_parts[i:i + len(person_words)]) == ' '.join(person_words)
                       for i in range(len(title_parts) - len(person_words) + 1)):
                filtered_persons.append(person)

        return ', '.join(filtered_persons), ', '.join(set(locations))
    except Exception as e:
        print(f"Error finding persons/locations in {filename}: {str(e)}")
        return "Nothing was found (error)", "Nothing was found (error)"


def extract_references(pdf_path):
    """Extract references section from PDF."""
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = len(pdf_document)

        text = ""
        for page_num in range(num_pages - 1, max(num_pages - 4, -1), -1):
            page = pdf_document.load_page(page_num)
            text = page.get_text() + text

        pdf_document.close()

        if not re.search(r'\bReference(?:s)?\b', text, re.IGNORECASE):
            return "Nothing was found"

        appendix_match = re.search(r'\bAppendix\b', text, re.IGNORECASE)
        if appendix_match:
            text = text[:appendix_match.start()]

        references_match = re.search(r'\bReference(?:s)?\b\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if references_match:
            return references_match.group(1).strip()

        return "Nothing was found"
    except Exception as e:
        print(f"Error extracting references from {pdf_path}: {str(e)}")
        return "Nothing was found (error)"


def process_pdf(file_path, filename):
    """Process a single PDF file and extract all details."""
    try:
        print(f"Processing: {filename}")

        title = pdf_title(file_path) or "Nothing was found"
        abstract = extract_abstract(file_path) or "Nothing was found"
        references = extract_references(file_path) or "Nothing was found"

        is_encrypted = False
        try:
            doc = fitz.open(file_path)
            is_encrypted = doc.is_encrypted
            doc.close()
        except Exception:
            is_encrypted = True

        if is_encrypted:
            authors = "Nothing was found (encrypted document)"
            locations = "Nothing was found (encrypted document)"
        else:
            authors, locations = find_persons_locations(file_path, title) or ("Nothing was found", "Nothing was found")

        return {
            "filename": filename,
            "title": title,
            "authors": authors,
            "locations": locations,
            "abstract": abstract,
            "references": references
        }
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return {
            "filename": filename,
            "title": "Nothing was found (error)",
            "authors": "Nothing was found (error)",
            "locations": "Nothing was found (error)",
            "abstract": "Nothing was found (error)",
            "references": "Nothing was found (error)"
        }


def extract_details(directory):
    """Extract details from all PDFs in directory using parallel processing."""
    files = [file for file in os.listdir(directory) if file.endswith(".pdf")]
    total_files = len(files)
    start_time = time.time()

    pdf_details = {}

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
        future_to_file = {
            executor.submit(process_pdf, os.path.join(directory, filename), filename): filename
            for filename in files
        }

        for i, future in enumerate(as_completed(future_to_file)):
            filename = future_to_file[future]
            try:
                details = future.result()
                pdf_details[filename] = {
                    "abstract": details["abstract"],
                    "title": details["title"],
                    "authors": details["authors"],
                    "locations": details["locations"],
                    "references": details["references"]
                }
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                pdf_details[filename] = {
                    "abstract": "Nothing was found (error)",
                    "title": "Nothing was found (error)",
                    "authors": "Nothing was found (error)",
                    "locations": "Nothing was found (error)",
                    "references": "Nothing was found (error)"
                }

            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / (i + 1)
            remaining_files = total_files - (i + 1)
            estimated_time_remaining = avg_time_per_file * remaining_files

            print(f"Processed {i + 1}/{total_files} files. Est. time remaining: {estimated_time_remaining:.2f} seconds")

    return pdf_details


def get_pdf_details():
    """Main function to extract details from PDFs."""
    directory_path = PDF_PATH
    pdf_details = extract_details(directory_path)
    return pdf_details


if __name__ == "__main__":
    get_pdf_details()


def is_probably_title(line):
    words = line.split()
    if len(words) < 3:
        return False
    capitalized = sum(w.isupper() or w.istitle() for w in words)
    return capitalized / len(words) > 0.5 or line.isupper()


def extract_title(text_lines):
    """
    Extracts a robust title from the text lines of a PDF.
    - Uses the first substantial line (>= 20 characters).
    - If no such line is found, returns the longest early line.
    """
    lines = [line.strip() for line in text_lines if line.strip()]

    for line in lines[:5]:
        if len(line) >= 20 and not any(c in line for c in "@0123456789"):
            return line

    fallback = max(lines[:10], key=len, default="Title not found")
    return fallback
