import os
import random
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

def download_random_arxiv_papers(num_papers: int, save_directory: str, category: Optional[str] = None):

    """
    Download random scientific papers from arXiv.org using the official API

    Parameters:
    num_papers (int): Number of papers to download
    save_directory (str): Directory to save PDFs
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # List of some common arXiv categories with proper format
    categories = [
        "cs.AI",  # Artificial Intelligence
        "cs.CL",  # Computation and Language
        "cs.CV",  # Computer Vision
        "cs.LG",  # Machine Learning
        "math.ST",  # Statistics Theory
        "physics.comp-ph",  # Computational Physics
        "q-bio.BM",  # Biomolecules
        "q-fin.PM",  # Portfolio Management
        "stat.ML"  # Machine Learning (Statistics)
    ]

    # arXiv API namespace
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    papers_downloaded = 0

    while papers_downloaded < num_papers:
        try:
            # Use selected category or choose a random one
            chosen_category = category if category else random.choice(categories)

            start_index = random.randint(0, 50)

            query_params = {
                'search_query': f'cat:{chosen_category}',
                'start': start_index,
                'max_results': 50,
                'sortBy': 'lastUpdatedDate',
                'sortOrder': 'descending'
            }

            api_url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(query_params)}"

            print(f"Searching in category: {chosen_category}")

            # Fetch results from the API
            with urllib.request.urlopen(api_url) as response:
                response_data = response.read()

            # Parse the XML response
            root = ET.fromstring(response_data)

            # Find all entries (papers)
            entries = root.findall('.//atom:entry', namespace)

            if not entries:
                print(f"No papers found in category {category}, trying another...")
                time.sleep(1)
                continue

            print(f"Found {len(entries)} papers in category {category}")

            # Select a random paper from the results
            random_entry = random.choice(entries)

            # Extract paper details
            title_elem = random_entry.find('./atom:title', namespace)
            title = title_elem.text.strip() if title_elem is not None else "Untitled"

            # Get the paper ID
            id_elem = random_entry.find('./atom:id', namespace)
            if id_elem is not None:
                paper_url = id_elem.text.strip()
                # The ID is a full URL, extract just the arXiv ID
                paper_id = paper_url.split('/abs/')[-1]
            else:
                print("Couldn't find paper ID, skipping...")
                continue

            # Create a valid filename
            valid_filename = ''.join(c if c.isalnum() or c in [' ', '.', '-', '_'] else '_' for c in title)
            valid_filename = valid_filename[:100]  # Limit filename length

            # PDF download URL
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

            # Save location
            save_path = os.path.join(save_directory, f"{valid_filename}.pdf")

            print(f"Downloading: {title}")
            print(f"URL: {pdf_url}")

            # Download the PDF
            urllib.request.urlretrieve(pdf_url, save_path)

            papers_downloaded += 1
            if progress_callback:
                progress_callback()

            print(f"Successfully downloaded ({papers_downloaded}/{num_papers})")
            print(f"Saved to: {save_path}")
            print("-" * 50)

            # Be nice to arXiv servers with a delay between downloads
            time.sleep(3)

        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Current API URL: {api_url}")
            time.sleep(2)  # Wait a bit before trying again

    print(f"Download complete! {num_papers} papers have been saved to {save_directory}")
    print(f"These papers were downloaded using the official arXiv API in compliance with arXiv's terms of service.")


if __name__ == "__main__":
    # You can change the number of papers and the save directory here
    download_random_arxiv_papers(num_papers=10, save_directory="documents/pdfs")