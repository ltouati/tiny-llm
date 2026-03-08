import os
import urllib.request
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# 1. Download the Tokenizer Dictionary for Rust to use later
tokenizer_url = "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json"
if not os.path.exists("tokenizer.json"):
    print("Downloading GPT-2 tokenizer.json...")
    urllib.request.urlretrieve(tokenizer_url, "tokenizer.json")

# 2. Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Streaming FineWeb-Edu from HuggingFace...")
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

output_file = "fineweb_edu.bin"
tokens_to_collect = 100_000_000 # 100 Million tokens (~400MB file)
total_tokens = 0

print(f"Tokenizing and saving to {output_file}...")
with open(output_file, "wb") as f:
    for doc in dataset:
        tokens = tokenizer.encode(doc["text"])
        tokens.append(50256) # <|endoftext|> token
        
        np_array = np.array(tokens, dtype=np.uint32)
        f.write(np_array.tobytes())
        
        total_tokens += len(tokens)
        if total_tokens >= tokens_to_collect:
            break
            
        if total_tokens % 5_000_000 < 5000:
            print(f"Collected {total_tokens / 1_000_000:.1f}M tokens...")

print(f"Done! Saved {total_tokens} tokens to disk.")
