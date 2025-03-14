# Import necessary libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch


def summarize_text(text, max_length=130, min_length=30, model_name="facebook/bart-large-cnn"):
    """
    Summarize the given text using a pre-trained model.

    Parameters:
    text (str): The text to summarize
    max_length (int): Maximum length of the summary
    min_length (int): Minimum length of the summary
    model_name (str): The pre-trained model to use

    Returns:
    str: The generated summary
    """
    try:
        if not text or len(text.strip()) == 0:
            return "Error: Empty text provided for summarization."

        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )

        max_input_length = 1024
        if len(text.split()) > max_input_length:
            truncated_text = " ".join(text.split()[:max_input_length])
        else:
            truncated_text = text

        summary = summarizer(
            truncated_text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return summary[0]['summary_text']

    except Exception as e:
        return f"Error during summarization: {str(e)}"


def chunk_and_summarize(text, max_chunk_size=1000, max_length=130, min_length=30):
    """
    Chunk the text and summarize each chunk, then combine the results.

    Parameters:
    text (str): The text to summarize
    max_chunk_size (int): Maximum size of each chunk in words
    max_length (int): Maximum length of each chunk summary
    min_length (int): Minimum length of each chunk summary

    Returns:
    str: The combined summary
    """
    if not text:
        return "Error: Empty text provided for summarization."

    words = text.split()

    if len(words) <= max_chunk_size:
        return summarize_text(text, max_length, min_length)

    chunks = []
    for i in range(0, len(words), max_chunk_size):
        chunk = " ".join(words[i:i + max_chunk_size])
        chunks.append(chunk)

    chunk_summaries = []
    for chunk in chunks:
        summary = summarize_text(chunk, max_length=max_length // 2, min_length=min_length // 2)
        if not summary.startswith("Error"):
            chunk_summaries.append(summary)

    combined_text = " ".join(chunk_summaries)

    if len(combined_text.split()) > max_chunk_size:
        return summarize_text(combined_text, max_length, min_length)

    return combined_text