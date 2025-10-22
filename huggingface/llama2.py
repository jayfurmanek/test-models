import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf") # Using Llama-2-7b-hf as a placeholder for demonstration
                                                                    # Replace with "meta-llama/Meta-Llama-3-8B" for Llama 3

# Define a custom device map
# This example distributes layers across two GPUs (cuda:0 and cuda:1)
# and places the embedding and final output layers on cuda:0.
# The exact mapping will depend on your model's architecture and available VRAM.
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": "cpu",
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    "model.layers.6": 0,
    "model.layers.7": 0,
    "model.layers.8": 0,
    "model.layers.9": 0,
    "model.layers.10": 0,
    "model.layers.11": 0,
    "model.layers.12": 0,
    "model.layers.13": 0,
    "model.layers.14": 0,
    "model.layers.15": 0,
    "model.layers.16": 0,
    "model.layers.17": 0,
    "model.layers.18": 0,
    "model.layers.19": 0,
    "model.layers.20": 0,
    "model.layers.21": 0,
    "model.layers.22": 0,
    "model.layers.23": 0,
    "model.layers.24": 0,
    "model.layers.25": 0,
    "model.layers.26": 0,
    "model.layers.27": 0,
    "model.layers.28": 0,
    "model.layers.29": 0,
    "model.layers.30": 0,
    "model.layers.31": 0,
    "model.norm": 0,
    "lm_head": 0,
}

# Load the model with the custom device_map
# The `device_map` argument automatically distributes the model according to the map.
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", # Replace with "meta-llama/Meta-Llama-3-8B" for Llama 3
    device_map=device_map,
    torch_dtype=torch.bfloat16 # Recommended for Llama 3 for memory and performance
)

# Verify the device of some layers
print(f"Embedding layer device: {model.model.embed_tokens.weight.device}")
print(f"First layer device: {model.model.layers[0].self_attn.q_proj.weight.device}")
print(f"Last layer device: {model.model.layers[-1].self_attn.q_proj.weight.device}")
print(f"LM Head device: {model.lm_head.weight.device}")

# Example of inference
prompt = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0") # Move inputs to a specific device

# Generate text
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{generated_text}")
