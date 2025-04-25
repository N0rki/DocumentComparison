import os
import time
import random
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional


def sanitize_filename(title: str, paper_id: str) -> str:
    """Sanitize and shorten filename while appending paper ID to avoid overwrites."""
    valid = ''.join(c if c.isalnum() or c in [' ', '.', '-', '_'] else '_' for c in title)
    valid = valid.strip().replace(' ', '_')[:80]
    return f"{valid}_{paper_id}.pdf"

def check_arxiv_throttling():
    """Check if current IP is being throttled or blocked by arXiv API."""
    try:
        query_params = {
            'search_query': 'cat:cs.AI',
            'start': 0,
            'max_results': 1,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }

        api_url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(query_params)}"
        print(f"🔍 Testing arXiv access: {api_url}")

        with urllib.request.urlopen(api_url, timeout=10) as response:
            status_code = response.getcode()
            response_data = response.read()

        if status_code != 200:
            print(f"⚠️ Unexpected HTTP status: {status_code}")
            return False

        try:
            root = ET.fromstring(response_data)
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            if entries:
                print("✅ IP is NOT throttled — arXiv API is accessible.")
                return True
            else:
                print("⚠️ No entries returned — possible API throttle or empty index.")
                return False
        except ET.ParseError:
            print("❌ Failed to parse response XML — potential throttle or block.")
            return False

    except Exception as e:
        print(f"🚫 Could not connect to arXiv: {e}")
        return False

def download_random_arxiv_papers(num_papers: int, save_directory: str, category: Optional[str] = None):
    os.makedirs(save_directory, exist_ok=True)

    if not check_arxiv_throttling():
        print("🛑 Aborting: Your IP might be throttled or blocked by arXiv.")
        return

    categories = [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "math.ST",
        "physics.comp-ph", "q-bio.BM", "q-fin.PM", "stat.ML"
    ]

    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    papers_downloaded = 0
    used_filenames = set(os.listdir(save_directory))

    max_empty_attempts = 1000
    empty_attempts = 0

    category_progress = {cat: 0 for cat in categories}  # track start index per category

    while papers_downloaded < num_papers and empty_attempts < max_empty_attempts:
        try:
            chosen_category = category if category else random.choice(categories)
            start_index = category_progress[chosen_category]

            print(f"\n🔄 Searching in category: {chosen_category} (from index {start_index})")

            query_params = {
                'search_query': f'cat:{chosen_category}',
                'start': start_index,
                'max_results': 50,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }

            api_url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(query_params)}"
            print(f"\nQuerying {chosen_category} from index {start_index}...")

            with urllib.request.urlopen(api_url) as response:
                response_data = response.read()

            root = ET.fromstring(response_data)
            entries = root.findall('.//atom:entry', namespace)

            if not entries:
                print("❌ No entries found — skipping index.")
                category_progress[chosen_category] += 50
                empty_attempts += 1
                continue

            print(f"📄 Found {len(entries)} entries.")

            new_papers = 0
            for entry in entries:
                if papers_downloaded >= num_papers:
                    break

                id_elem = entry.find('./atom:id', namespace)
                title_elem = entry.find('./atom:title', namespace)

                if id_elem is None or title_elem is None:
                    continue

                paper_id = id_elem.text.strip().split('/abs/')[-1]
                title = title_elem.text.strip()
                filename = sanitize_filename(title, paper_id)
                save_path = os.path.join(save_directory, filename)

                if filename in used_filenames or os.path.exists(save_path):
                    continue

                pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

                try:
                    urllib.request.urlretrieve(pdf_url, save_path)
                    papers_downloaded += 1
                    new_papers += 1
                    used_filenames.add(filename)

                    print(f"✅ ({papers_downloaded}/{num_papers}) {filename}")
                    print(f"Saved to: {save_path}")
                    print(f"URL: {pdf_url}")
                except Exception as download_err:
                    print(f"❌ Failed to download {filename}: {download_err}")

                time.sleep(2)

            category_progress[chosen_category] += 50

            if new_papers == 0:
                empty_attempts += 1
                print(f"⚠️ No new papers from index {start_index}. Empty attempt {empty_attempts}/{max_empty_attempts}")
            else:
                empty_attempts = 0

            if len(entries) < 50:
                print(f"📭 End of available results for {chosen_category}. Moving on.")
                category_progress[chosen_category] = 0

        except Exception as e:
            print(f"⚠️ Error occurred: {e}")
            empty_attempts += 1
            category_progress[chosen_category] += 50
            time.sleep(2)

    print(f"\n🎉 Finished downloading {papers_downloaded} unique papers to '{save_directory}'.")


if __name__ == "__main__":
    download_random_arxiv_papers(num_papers=500, save_directory="documents/pdfs_500")
