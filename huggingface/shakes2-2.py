from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "NataliaH/gpt2-tiny-shakespeare"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Encode prompt and run generation
prompt = "To be, or not to be:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=1,
    temperature=0.8,
    top_p=0.9,
    do_sample=True
)

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

