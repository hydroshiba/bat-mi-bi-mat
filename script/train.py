import torch
import json
import random
import re
from tqdm import tqdm
from unidecode import unidecode
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

model_name = 'NlpHUST/gpt2-vietnamese'

def caesar_cipher_solve(text):
	ciphers = []
	for shift in range(0, 26):
		candidate = ""
		for char in text:
			if char.isalpha():
				shifted_char = chr((ord(char.lower()) - ord('a') + shift) % 26 + ord('a'))
				candidate += shifted_char
			else:
				candidate += char
		ciphers.append(candidate)
	return ciphers

def atbash_solve(text):
	cipher = ""
	for char in text:
		if char.isalpha():
			shifted_char = chr(ord('z') - (ord(char.lower()) - ord('a')))
			cipher += shifted_char
		else:
			cipher += char
	return [cipher]

def semantic_loss(model, tokenizer, text):
	inputs = tokenizer(text, return_tensors='pt')
	with torch.no_grad():
		outputs = model(**inputs, labels=inputs["input_ids"])
	n_tokens = inputs["input_ids"].shape[1]
	return outputs.loss.item() * n_tokens  # total loss, not average

def quadgram_loss(text, quadgram_probs):
	loss = 0.0
	n_quadgrams = 0
	for i in range(len(text) - 3):
		quadgram = text[i:i+4]
		prob = quadgram_probs.get(quadgram, 1e-10)  # Avoid log(0)
		loss -= torch.log(torch.tensor(prob))
		n_quadgrams += 1
	return loss.item() / n_quadgrams if n_quadgrams > 0 else float('inf')

def vowel_percentage(text):
	vowels = 'aeiou'
	count = sum(1 for char in text if char in vowels)
	return count / len(text) if len(text) > 0 else 0

def main():
	NUM_SAMPLES = 100

	try:
		with open('quadgram.json', 'r') as f:
			quadgram_probs = json.load(f)
	except FileNotFoundError:
		print("quadgram.json not found. Please run quadgram.py first.")
		return
		
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name)

	# 1. Pick samples from corpus
	with open('data/corpus_vn.txt', 'r', encoding='utf-8') as f:
		words = f.read().split()

	print("Picking samples...")
	samples = []
	for _ in tqdm(range(NUM_SAMPLES)):
		target_len = random.randint(50, 150)
		start_idx = random.randint(0, max(0, len(words) - (target_len // 5)))
		
		current_sample = []
		current_len = 0
		idx = start_idx
		while current_len < target_len and idx < len(words):
			word = words[idx]
			# +1 for space
			if current_len + len(word) + (1 if current_sample else 0) <= target_len:
				current_sample.append(word)
				current_len += len(word) + (1 if current_sample else 0)
			else:
				break
			idx += 1
		if current_sample:
			samples.append(" ".join(current_sample))

	# 2. Process samples, generate ciphers and calculate losses
	print("Generating dataset...")
	X = []
	y = []
	groups = []

	for group_id, sample in enumerate(tqdm(samples)):
		flat_sample = unidecode(sample).lower()
		clean_sample = re.sub(r'[^a-z ]', '', flat_sample)
		clean_sample = ' '.join(clean_sample.split()) # normalize spaces

		if not clean_sample:
			continue

		ciphertexts = []
		ciphertexts.extend(caesar_cipher_solve(clean_sample))
		ciphertexts.extend(atbash_solve(clean_sample))
		
		# Deduplicate while preserving order, ensuring clean_sample is first
		unique_texts = [clean_sample]
		for text in ciphertexts:
			if text not in unique_texts:
				unique_texts.append(text)
		
		# Rank using semantic loss
		scored_texts = []
		for text in unique_texts[1:]:
			try:
				loss = semantic_loss(model, tokenizer, text)
			except Exception:
				loss = float('inf')
			scored_texts.append({'text': text, 'semantic_loss': loss})
			
		scored_texts.sort(key=lambda x: x['semantic_loss'])
		
		# Generate ground truth rankings
		ranks = {clean_sample: 1} # Original is always 1
		for i, item in enumerate(scored_texts):
			ranks[item['text']] = i + 2
		
		# Extract features for learning to rank
		for text in unique_texts:
			q_loss = quadgram_loss(text, quadgram_probs)
			v_perc = vowel_percentage(text)
			token_count = len(tokenizer.encode(text))
			
			X.append([q_loss, v_perc, token_count])
			# Relevance scores typically descend. NDCG handles them.
			# Using inverse rank as relevance
			y.append(1.0 / ranks[text]) 
			groups.append(group_id)

	X = np.array(X)
	y = np.array(y)

	# 3. Train simple regression model (PyTorch)
	print("Normalizing features...")
	X_mean = X.mean(axis=0)
	X_std = X.std(axis=0) + 1e-8
	X_norm = (X - X_mean) / X_std

	print("Training simple regression model...")
	
	class SimpleRanker(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.linear = torch.nn.Linear(3, 1)
			self.activation = torch.nn.Sigmoid()

		def forward(self, x):
			return self.activation(self.linear(x))

	ranker = SimpleRanker()
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(ranker.parameters(), lr=0.1)

	X_tensor = torch.tensor(X_norm, dtype=torch.float32)
	y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

	for epoch in range(250):
		optimizer.zero_grad()
		outputs = ranker(X_tensor)
		loss = criterion(outputs, y_tensor)
		loss.backward()
		optimizer.step()
		
		# Optional: Print loss occasionally
		# if (epoch + 1) % 50 == 0:
		#     print(f"Epoch [{epoch + 1}/250], Loss: {loss.item():.4f}")

	print("Training finished!")
	
	weights = ranker.linear.weight.data.numpy()[0]
	bias = ranker.linear.bias.data.numpy()[0]
	
	print("\n=== Model Parameters ===")
	print("Activation Function: Sigmoid")
	print("Note: Features were standard-normalized before training.")
	print(f"Feature Means: [Quadgram: {X_mean[0]:.4f}, Vowel %: {X_mean[1]:.4f}, Token Count: {X_mean[2]:.4f}]")
	print(f"Feature Stds:  [Quadgram: {X_std[0]:.4f}, Vowel %: {X_std[1]:.4f}, Token Count: {X_std[2]:.4f}]")
	print("\nThe simple equation is:")
	print(f"Z = ({weights[0]:.4f} * Norm(Quadgram Loss)) + ")
	print(f"    ({weights[1]:.4f} * Norm(Vowel Percentage)) + ")
	print(f"    ({weights[2]:.4f} * Norm(Token Count)) + ")
	print(f"    ({bias:.4f})")
	print("Score = 1 / (1 + e^-Z)")
	
	# Save weights and bias to JSON for easy loading later
	model_params = {
		"weights": [float(w) for w in weights],
		"bias": float(bias),
		"feature_means": [float(m) for m in X_mean],
		"feature_stds": [float(s) for s in X_std]
	}
	with open("ranker_params.json", "w") as f:
		json.dump(model_params, f, indent=4)
	print("\nSaved parameters to ranker_params.json")

if __name__ == "__main__":
	main()