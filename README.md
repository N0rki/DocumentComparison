
# Dynamic Document Link Analysis System

A context-aware platform for analyzing, comparing, and predicting semantic relationships between scientific research papers using neural networks, vector embeddings, and knowledge graphs.

## Overview

This system supports dynamic scientific document analysis by combining state-of-the-art NLP, document vectorization, clustering, and neural link prediction.

It enables interactive exploration of document relationships, evolving link structures, and graph-based insights through a browser-based manager and a Streamlit dashboard.

## Key Features

- Transformer-Based Embedding: SPECTER, SBERT, SPECTER2
- Hybrid Similarity: Cosine + Metadata + Clusters
- Neural Link Prediction: LSTM-based model
- Clustering & Dimensionality Reduction: KMeans, UMAP, t-SNE
- Interactive Visualizations: Clusters, Networks, Comparisons
- Knowledge Graph Export: RDF/XML, GraphML, Cypher, JSON
- SPARQL Query Interface: Query RDF document graph
- Explainability (WIP): LSTM via SHAP
- Web UI: Launch, view, manage collections and dashboard

## Interfaces

### Flask Web App

- Manage document collections
- Download and vectorize PDFs
- Launch the Streamlit dashboard
- View and delete documents

Starts at: http://localhost:5000  
Launch dashboard: via "Launch Streamlit Dashboard" button

### Streamlit Dashboard

- Interactive graph exploration
- Similarity visualizations and comparison tools
- Clustering, filtering, explainability

Runs at: http://localhost:8501  
Launch via Flask → "Launch Dashboard" or `streamlit run app/dashboard.py`

## Project Structure

app/
├── chroma_app.py           # Flask web server + interface
├── dashboard.py            # Streamlit dashboard UI
├── vectorization.py        # Embedding & model loading
├── model_lstm.py           # LSTM architecture
├── train_lstm.py           # Training script
├── extract_data.py         # PDF text & metadata extractor
├── knowledge_graph.py      # Knowledge graph export
├── evaluation.py           # Evaluation tools
├── documents/              # Stored PDF folders
├── templates/              # HTML UI for Flask
│   ├── index.html
│   ├── view_collection.html
│   ├── download_results.html

## Quickstart

### Install Dependencies
```
pip install -r requirements.txt
```

### Run Web Manager (Flask)
```
python app/chroma_app.py
```

Then go to: http://localhost:5000

### Launch Streamlit Dashboard
Click “Launch Streamlit Dashboard” or run manually:
```
streamlit run app/dashboard.py
```

## Output Files

- knowledge_graph.rdf – RDF/XML export
- graph.cypher – Neo4j import
- graph.graphml, graph.json – Network formats
- model_lstm.pt – Trained link predictor
- .html – Interactive network visualizations

## Visual Features

- 2D/3D graph projections (UMAP, PCA, t-SNE)
- Document clusters & similarity thresholds
- Side-by-side document comparison
- Link prediction (Cosine, Hybrid, LSTM)
- Link evolution & churn visualizations
- SPARQL query interface

## Author

Your Name  
Master’s Thesis, Software Engineering  
GitHub • LinkedIn • Email
