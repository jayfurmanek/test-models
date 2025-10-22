import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import os
HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN[:3]+'...'  # 'hf_...'

# Specify the local directory for the download
local_dir = "./llama3-8b-instruct"

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Download the model files
weights_location = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    token=HF_TOKEN, # Use your token for authentication
)

# Create a device_map where most things are on CPU
device_map = {
        "model.embed_tokens": "cpu",
        }

for i in range(32):
    if i == 0:
        # Do one layer on GPU
        device_map[f"model.layers.{i}"] = "cuda:0"
    else:
        device_map[f"model.layers.{i}"] = "cpu"

device_map[f"model.norm"] = "cpu"
device_map[f"model.rotary_emb"] = "cpu"
device_map[f"lm_head"] = "cpu"

model = AutoModelForCausalLM.from_pretrained(
        model_id,
#        device_map="auto",
#        device_map=device_map,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        )

#print(model.hf_device_map)

# Initialize model with empty weights to save memory during loading
#with init_empty_weights():
#    model = AutoModelForCausalLM.from_pretrained(
#        model_id,
#        torch_dtype=torch.bfloat16, # Or torch.float16, depending on your hardware
#        low_cpu_mem_usage=True,
#    )

model = load_checkpoint_and_dispatch(
    model, 
    checkpoint=weights_location, 
    device_map=device_map, 
    #offload_buffers=True,
)

print("Device Map: ")
print(model.hf_device_map)

tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    token=HF_TOKEN, 
)

# https://huggingface.co/docs/transformers/main/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {
        "role": "user",
        "content": "Write me a mail.",
    },
    {
        "role": "assistant",
        "content": "To your friend Andrew?",
    },
    {
        "role": "user",
        "content": "Yes.",
    },
]
# Print the assistant's response

temp_output = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,    # <- ask to return the attn mask as well as the input ids
).to("cpu")  # <- this causes a warning, but we want most everything on CPU here

outputs = model.generate(
    input_ids=temp_output["input_ids"],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    attention_mask=temp_output["attention_mask"]
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
