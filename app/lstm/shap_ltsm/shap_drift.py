from scipy.spatial.distance import cosine
import numpy as np

def calculate_shap_drift(shap_t1, shap_t2):
    return cosine(shap_t1, shap_t2)

def compare_drift_across_time(shap_dict_by_year, doc_pair):
    years = sorted(shap_dict_by_year.keys())
    drift_series = []

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        shap1 = shap_dict_by_year[y1].get(doc_pair)
        shap2 = shap_dict_by_year[y2].get(doc_pair)
        if shap1 is not None and shap2 is not None:
            drift = calculate_shap_drift(shap1, shap2)
            drift_series.append({
                "pair": doc_pair,
                "from": y1,
                "to": y2,
                "shap_drift": drift
            })
    return drift_series
