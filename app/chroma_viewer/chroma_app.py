from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import chromadb
from typing import List, Dict, Any
import os
import random
import requests
import logging
import sys
import datetime
import subprocess
import threading

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from vectorization import (
    load_model,
    vectorize_text_specter,
    add_documents_to_collection,
    process_documents_in_batches
)
from database_connection import connect_to_chromadb

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['SESSION_TYPE'] = 'filesystem'

chroma_client = chromadb.HttpClient(host='localhost', port=8000)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chroma_app.log')
    ]
)
logger = logging.getLogger(__name__)

def get_collection_details(collection_name: str) -> Dict[str, Any]:
    collection = chroma_client.get_collection(collection_name)
    return {
        'name': collection.name,
        'count': collection.count(),
        'metadata': collection.metadata or {}
    }

def extract_pdf_info(filepath):
    from extract_data import process_pdf
    filename = os.path.basename(filepath)
    try:
        details = process_pdf(filepath, filename)
        return {
            'title': details.get('title', 'Unknown Title'),
            'authors': details.get('authors', 'Unknown Authors'),
            'abstract': details.get('abstract', 'No abstract found'),
            'year': random.randint(2010, 2023)
        }
    except Exception as e:
        logger.error(f"Error extracting info from {filename}: {str(e)}")
        return {
            'title': f"Unknown Title - {filename}",
            'authors': "Unknown Authors",
            'abstract': "Extraction failed",
            'year': random.randint(2010, 2023)
        }

def run_dashboard():
    # Resolve path: go one directory up from this file, then dashboard.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up to 'app/'
    dashboard_path = os.path.join(base_dir, "dashboard.py")
    subprocess.run(["streamlit", "run", dashboard_path])

@app.route('/')
def index():
    try:
        collection_names = chroma_client.list_collections()
        collections_info = [get_collection_details(c) for c in collection_names]

        documents_dir = os.path.join(APP_DIR, 'documents')
        directories = [d for d in os.listdir(documents_dir)
                       if os.path.isdir(os.path.join(documents_dir, d))]

        return render_template('index.html', collections=collections_info, directories=directories)
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return render_template('index.html', collections=[], directories=[])

@app.route('/delete/<collection_name>', methods=['POST'])
def delete_collection(collection_name):
    try:
        chroma_client.delete_collection(collection_name)
        flash(f"Deleted '{collection_name}'", "success")
    except Exception as e:
        flash(f"Error deleting: {str(e)}", "error")
    return redirect(url_for('index'))

@app.route('/create', methods=['POST'])
def create_collection():
    collection_name = request.form.get('collection_name')
    if not collection_name:
        flash("Collection name is required", "error")
        return redirect(url_for('index'))
    try:
        chroma_client.create_collection(collection_name)
        flash(f"Collection '{collection_name}' created", "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
    return redirect(url_for('index'))

@app.route('/view/<collection_name>')
def view_collection(collection_name):
    try:
        collection = chroma_client.get_collection(collection_name)
        items = collection.get()
        records = [
            {
                'id': items['ids'][i],
                'embedding': f"Vector of length {len(items['embeddings'][i])}" if items['embeddings'] else None,
                'document': items['documents'][i] if items['documents'] else None,
                'metadata': items['metadatas'][i] if items['metadatas'] else None
            }
            for i in range(len(items['ids']))
        ]
        return render_template('view_collection.html', collection_name=collection_name,
                               count=collection.count(), metadata=collection.metadata or {}, records=records)
    except Exception as e:
        flash(f"Error: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/download-pdfs', methods=['POST'])
def download_pdfs():
    try:
        num_pdfs = int(request.form.get('pdf_count', 5))
        selected = request.form.get('download_directory')
        category = request.form.get('category', '').strip()

        if selected == "__NEW__" or not selected:
            target_dir = f"pdfs_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        else:
            target_dir = selected

        save_path = os.path.join(APP_DIR, 'documents', target_dir)
        os.makedirs(save_path, exist_ok=True)

        from download_pdfs import download_random_arxiv_papers
        download_random_arxiv_papers(num_papers=num_pdfs, save_directory=save_path, category=category if category else None)


        flash(f"Downloaded {num_pdfs} PDFs to {target_dir}" + (f" (category: {category})" if category else ""), "success")
    except Exception as e:
        flash(f"Error: {str(e)}", "error")

    return redirect(url_for('index'))

@app.route('/vectorize-pdfs', methods=['POST'])
def vectorize_pdfs():
    try:
        collection_name = request.form.get('collection_name')
        directory_name = request.form.get('directory_name')
        if not collection_name or not directory_name:
            flash("Collection name and directory selection are required", "error")
            return redirect(url_for('index'))

        try:
            collection = chroma_client.get_collection(collection_name)
        except:
            collection = chroma_client.create_collection(collection_name)

        directory_path = os.path.join(APP_DIR, 'documents', directory_name)
        if not os.path.exists(directory_path):
            flash("Selected directory does not exist", "error")
            return redirect(url_for('index'))

        files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        if not files:
            flash("No PDFs found in the selected directory", "warning")
            return redirect(url_for('index'))

        details = {}
        for filename in files:
            filepath = os.path.join(directory_path, filename)
            info = extract_pdf_info(filepath)
            details[filename] = {
                'title': info['title'],
                'authors': info['authors'],
                'abstract': info['abstract'],
                'year': info.get('year', 2023)
            }

        count = process_documents_in_batches(details, directory_path)
        flash(f"Added {count} documents to '{collection_name}'", "success")
        return redirect(url_for('view_collection', collection_name=collection_name))

    except Exception as e:
        flash(f"Vectorization error: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/status')
def status():
    try:
        model, _ = load_model()
        return jsonify({
            'model_loaded': model is not None,
            'device': str(model.device) if model else 'unknown',
            'pdf_count': 0
        })
    except:
        return jsonify({'model_loaded': False, 'device': 'unknown', 'pdf_count': 0})

@app.route('/list-files/<directory>')
def list_files(directory):
    try:
        directory_path = os.path.join(APP_DIR, 'documents', directory)
        if not os.path.exists(directory_path):
            return jsonify({'error': 'Directory not found'}), 404
        pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
        return jsonify({'files': pdf_files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list-directories')
def list_directories():
    try:
        documents_dir = os.path.join(APP_DIR, 'documents')
        directories = [d for d in os.listdir(documents_dir)
                       if os.path.isdir(os.path.join(documents_dir, d))]
        return jsonify({'directories': directories})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload-model', methods=['POST'])
def reload_model():
    try:
        global model, tokenizer
        model, tokenizer = load_model()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/delete-document/<collection_name>', methods=['POST'])
def delete_document(collection_name):
    record_id = request.form.get('record_id')
    if not record_id:
        flash('Missing record ID.', 'error')
        return redirect(url_for('view_collection', collection_name=collection_name))

    try:
        collection = chroma_client.get_collection(name=collection_name)
        collection.delete(ids=[record_id])
        flash(f'Record {record_id} deleted successfully.', 'success')
    except Exception as e:
        flash(f'Error deleting record: {str(e)}', 'error')

    return redirect(url_for('view_collection', collection_name=collection_name))

@app.route('/launch-dashboard', methods=['POST'])
def launch_dashboard():
    try:
        threading.Thread(target=run_dashboard, daemon=True).start()
        flash("Dashboard is launching in a new tab (or visit http://localhost:8501)", "success")
    except Exception as e:
        flash(f"Failed to launch dashboard: {str(e)}", "error")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
