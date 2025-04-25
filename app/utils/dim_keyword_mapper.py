import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def compute_dim_keywords_from_abstracts(json_path: str, n_dims: int = 768) -> dict:
    """
    Generate interpretable keywords for each SPECTER embedding dimension.

    Args:
        json_path (str): Path to fused_documents.json file
        n_dims (int): Total number of embedding dimensions (default 768)

    Returns:
        dict: Mapping of embedding dimension index to top keyword
    """
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    df = pd.DataFrame(docs)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    # TF-IDF + SVD
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(df["abstract"])

    svd = TruncatedSVD(n_components=300, random_state=42)
    svd.fit(tfidf_matrix)

    vocab = np.array(vectorizer.get_feature_names_out())

    reduced_top_words = {
        dim: vocab[np.argmax(np.abs(svd.components_[dim]))]
        for dim in range(svd.components_.shape[0])
    }

    expanded_top_words = {
        i: reduced_top_words[i % svd.components_.shape[0]]
        for i in range(n_dims)
    }

    return expanded_top_words
