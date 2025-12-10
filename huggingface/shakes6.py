import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import dispatch_model
import torch

# Model + tokenizer
model_name = "NataliaH/gpt2-tiny-shakespeare"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define device map — only first layer (h.0) on GPU
device_map = {
    "transformer.wte": "cpu",  # embeddings
    "transformer.wpe": "cpu",
#    "transformer.h.0": "cuda",  # only this layer on GPU
}

# Assign all other layers to CPU automatically
for i in range(0, model.config.n_layer):
    device_map[f"transformer.h.{i}"] = "cpu" if i>0 else "cuda"
    #for x in ("ln_1",  "ln_2"):
    #    device_map[f"transformer.h.{i}."+x] = "cpu" if i>0 else "cuda"

# Remaining parts on CPU
device_map.update({
    "transformer.ln_f": "cpu",
    "lm_head": "cpu"
})

# Use Accelerate to distribute layers
model = dispatch_model(model, device_map=device_map, main_device="cpu")

# Prepare input
prompt = "A rose"
inputs = tokenizer(prompt, return_tensors="pt") # send input to GPU for first layer
print(inputs.input_ids.shape)
t1=time.time()
# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        use_cache=False,
        do_sample=False,
        output_attentions=False,
        output_hidden_states=False,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

t2=time.time()
print("Generated in", t2-t1, "s")
exit(0)
prompt = "Yet who would have thought the old man to have had so much"

inputs = tokenizer(prompt, return_tensors="pt") #.to("meta")  # send input to GPU for first layer
print(inputs.input_ids.shape)
t1=time.time()
# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        use_cache=False,
        do_sample=False,
        output_attentions=False,
        output_hidden_states=False,
        temperature=0.5,
        top_k=50,
        top_p=0.95,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

t2=time.time()
print("Generated in", t2-t1, "s")


