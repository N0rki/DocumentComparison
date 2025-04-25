import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Function to download nltk resources if not already downloaded
# def download_nltk_resources():
#     resources = ["stopwords", "punkt"]
#     for resource_name in resources:
#         try:
#             nltk.data.find(f"tokenizers/{resource_name}")
#         except LookupError:
#             nltk.download(resource_name)


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def text_normalization(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.lower()
    words = word_tokenize(text)

    filtered_words = [
        re.sub(r'\W+', '', word) for word in words
        if word not in stop_words
           and len(word) > 1
           and len(re.findall(r'\d', word)) <= 5
    ]

    return ' '.join(filtered_words)
