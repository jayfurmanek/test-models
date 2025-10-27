from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model
import torch

# Model + tokenizer
model_name = "NataliaH/gpt2-tiny-shakespeare"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define device map — only first layer (h.0) on GPU
device_map = {
    "transformer.wte": "cuda",  # embeddings
    "transformer.wpe": "cuda",
    "transformer.h.0": "cuda",  # only this layer on GPU
}

# Assign all other layers to CPU automatically
for i in range(1, model.config.n_layer):
    device_map[f"transformer.h.{i}"] = "cpu"

# Remaining parts on CPU
device_map.update({
    "transformer.ln_f": "cpu",
    "lm_head": "cpu"
})

# Use Accelerate to distribute layers
model = dispatch_model(model, device_map=device_map)

# Prepare input
prompt = "To be, or not to be:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # send input to GPU for first layer

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        use_cache=False,
        do_sample=False,
        output_attentions=False,
        output_hidden_states=False,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

