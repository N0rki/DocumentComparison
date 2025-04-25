import shap
import torch
import numpy as np
from ..model_lstm import LinkPredictorLSTM

def prepare_input_features(emb1, emb2):
    emb1, emb2 = np.array(emb1), np.array(emb2)
    return np.concatenate([emb1, emb2, np.abs(emb1 - emb2)])

def predict_fn(model, X):
    with torch.no_grad():
        x_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        preds = model(x_tensor).numpy()
    return preds

def explain_pair_shap(model, emb1, emb2, background_pairs):
    f_vector = prepare_input_features(emb1, emb2)
    background = np.stack([prepare_input_features(a, b) for a, b in background_pairs])

    explainer = shap.KernelExplainer(lambda x: predict_fn(model, x), background)
    shap_values = explainer.shap_values(np.array([f_vector]))[0]
    return shap_values
