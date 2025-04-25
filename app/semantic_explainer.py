import torch
from transformers import pipeline

_explainer_pipe = None

def load_local_explainer():
    global _explainer_pipe
    if _explainer_pipe is None:
        _explainer_pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            max_length=256,
            device=0 if torch.cuda.is_available() else -1
        )
    return _explainer_pipe

def generate_local_semantic_explanation(doc1, doc2):
    explainer = load_local_explainer()
    prompt = f"""You are a helpful assistant that compares two research paper abstracts and explains how they are semantically similar.

    Your task is to write a clear explanation of their **common focus, methods, or problems they address**.

    Only describe what they share — do NOT summarize them separately or copy their content directly.

    Abstract A:
    {doc1.get("abstract", "")}

    Abstract B:
    {doc2.get("abstract", "")}

    Explain in 2-3 full sentences what these abstracts have in common, in plain English."""

    result = explainer(prompt)[0]["generated_text"]
    return result.strip()
