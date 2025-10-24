import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

# --- Config ---
model_name = "meta-llama/Meta-Llama-3-8B"
checkpoint_folder = snapshot_download(model_name)
torch.set_grad_enabled(False)

# --- Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer("Hello", return_tensors="pt").to("cpu")  # single-token input

# --- Build empty model & map devices ---
config = AutoConfig.from_pretrained(model_name)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Keep only embed + rotary + first layer on GPU
device_map = {
    #"model.embed_tokens": "cuda",
    #"model.rotary_emb": "cuda",
    #"model.model.rotary_emb": "cuda",
    "model.embed_tokens": "cpu",
    "model.rotary_emb": "cpu",
    "model.layers.0": "cuda",
    "model.norm": "cpu",
    "lm_head": "cpu",
}
for i in range(1, config.num_hidden_layers):
    device_map[f"model.layers.{i}"] = "cpu"

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=checkpoint_folder,
    device_map=device_map,
    dtype=torch.float16,
    offload_folder="offload_folder",
    offload_buffers=True,
    no_split_module_classes=["LlamaDecoderLayer", "LlamaRotaryEmbedding"],
)

model.config.use_cache = False
model.eval()
print("Loaded model. GPU components:", {k: v for k, v in model.hf_device_map.items() if v == "cuda"})

# --- Optional: fuse GPU layer for fewer kernel launches ---
#try:
#    model.model.layers[0] = torch.compile(model.model.layers[0], mode="reduce-overhead")
#    print("Compiled first layer for minimal kernel launches.")
#except Exception as e:
#    print("torch.compile not available or failed:", e)

# --- Minimal forward ---
with torch.inference_mode():
    outputs = model(**inputs, use_cache=False, output_attentions=False, output_hidden_states=False)
    logits = outputs.logits[:, -1, :]  # only last token logits
    next_token = torch.argmax(logits, dim=-1)

decoded = tokenizer.decode(next_token[0])
print("Next token:", decoded)

