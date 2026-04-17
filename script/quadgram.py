from unidecode import unidecode
import re
from collections import Counter
import json
from tqdm import tqdm

with open('data/corpus_vn.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read().lower()

flat_text = unidecode(raw_text)
clean_text = re.sub(r'[^a-z]', '', flat_text)

print("Extracting quadgrams...")
quadgrams = []
for i in tqdm(range(len(clean_text) - 3), desc="Extracting quadgrams"):
    chunk = clean_text[i:i+4]
    quadgrams.append(chunk)

# 5. Count how many times each quadgram appears
counts = Counter(quadgrams)
total_quadgrams = sum(counts.values())

# 6. Convert raw counts into percentages (probabilities)
quadgram_probabilities = {}
for chunk, count in tqdm(counts.items(), desc="Calculating percentages"):
    quadgram_probabilities[chunk] = count / total_quadgrams

# 7. Save it to a tiny JSON file for your solver/website to use
with open('quadgram.json', 'w') as f:
    json.dump(quadgram_probabilities, f)

print(f"Done! Saved {len(quadgram_probabilities)} unique quadgrams to quadgram.json")