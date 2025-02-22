from transformers import pipeline

ner_pipeline = pipeline("ner", grouped_entities=True ,tokenizer="dslim/bert-base-NER")

def extract_entities(text):
    #PDF text
    return ner_pipeline(text)
