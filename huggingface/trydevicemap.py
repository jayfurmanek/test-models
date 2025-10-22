import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

device_map = {
        "model.embed_tokens": "cpu",
        "model.norm": "cpu",
        "lm_head": "cpu",
        }

for i in range(1):
    device_map[f"model.layers.{i}"] = "cuda:0"

for i in range(1, 32):
    device_map[f"model.layers.{i}"] = "cpu"


model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        )

print(model.hf_device_map)
