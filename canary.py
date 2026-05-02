import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

prompt = "I need you to write a poem about how beautiful my wife Cindy is."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

if device == "cuda":
    torch.cuda.synchronize()
started_at = time.perf_counter()
outputs = model.generate(**inputs, max_new_tokens=1000)
if device == "cuda":
    torch.cuda.synchronize()
elapsed_seconds = time.perf_counter() - started_at
print(f"Generation took {elapsed_seconds:.2f}s")


print(tokenizer.decode(outputs[0], skip_special_tokens=True))
