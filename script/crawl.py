from datasets import load_dataset
from unidecode import unidecode
from tqdm import tqdm
import re

print("Downloading the entire Vietnamese Wikipedia...")
# This pulls the official Vietnamese Wikipedia dataset
dataset = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)

print("Processing text...")
# Grab the first 50,000 articles (more than enough for a quadgram dictionary)
articles = []
for idx, article in enumerate(tqdm(dataset, total=50000, desc="Fetching articles")):
    if idx >= 50000:
        break
    articles.append(article['text'])
    
raw_text = " ".join(articles)

print("Saving raw text to data/corpus_vn.txt...")
with open("data/corpus_vn.txt", "w", encoding="utf-8") as f:
    f.write(raw_text)