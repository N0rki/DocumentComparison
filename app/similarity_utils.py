from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "similarity_models", "mv_fusion_model.pkl")

_fusion_model = None
def load_mv_fusion_model(path=MODEL_PATH):
    global _fusion_model
    if _fusion_model is None:
        _fusion_model = joblib.load(path)
    return _fusion_model

def compute_mv_similarity(doc1, doc2, model_path=MODEL_PATH):
    model = load_mv_fusion_model(model_path)

    v1_specter = np.array(doc1['specter_embedding'])
    v2_specter = np.array(doc2['specter_embedding'])

    v1_sbert = np.array(doc1['sbert_embedding'])
    v2_sbert = np.array(doc2['sbert_embedding'])

    v1_meta = np.array(doc1['metadata_vector'])
    v2_meta = np.array(doc2['metadata_vector'])

    sim_specter = float(cosine_similarity([v1_specter], [v2_specter])[0][0])
    sim_sbert = float(cosine_similarity([v1_sbert], [v2_sbert])[0][0])
    sim_meta = float(cosine_similarity([v1_meta], [v2_meta])[0][0])

    X = np.array([[sim_specter, sim_sbert, sim_meta]])
    score = model.predict_proba(X)[0][1]  # Probability for class 1 (link)

    return score