import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

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

# Keep only the first 8 layers (for example)
model.model.layers = model.model.layers[:1]
model.config.num_hidden_layers = 1
print(model.config)

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
        "content": "Hi",
    },
]
# Print the assistant's response

temp_output = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,    # <- ask to return the attn mask as well as the input ids
    #).to("cuda:0")  # <- this causes a warning, but we want most everything on CPU here
    ).to("cpu")  # <- this causes a warning, but we want most everything on CPU here

print(temp_output["input_ids"])


#with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#            on_trace_ready=tensorboard_trace_handler('./log/profiler_output')) as p:
outputs = model.generate(
    input_ids=temp_output["input_ids"],
    max_new_tokens=1,
    use_cache=False,
    do_sample=False,
    output_attentions=False,
    output_hidden_states=False,
    temperature=0.7,
    top_k=50,
    attention_mask=temp_output["attention_mask"]
)

#p.export_chrome_trace("llama_profile.json")

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
