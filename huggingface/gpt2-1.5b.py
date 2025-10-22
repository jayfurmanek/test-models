from huggingface_hub import snapshot_download
checkpoint = "marcsun13/gpt2-xl-linear-sharded"
weights_location = snapshot_download(repo_id=checkpoint)

from accelerate import init_empty_weights
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = 50257
model_config.block_size = 1024

with init_empty_weights():
    model = GPT(model_config)

from accelerate import load_checkpoint_and_dispatch

device_map = {
    "transformer.wte": 0,
    "transformer.wpe": 0,
    "transformer.drop": 0,
}
for i in range(47): # Assuming 12 transformer layers in GPT-2
    device_map[f"transformer.h.{i}"] = "cpu"

# Put last layer on GPU
device_map[f"transformer.h.47"] = "cuda:0"
device_map[f"transformer.ln_f"] = 0
device_map[f"lm_head"] = 0


model = load_checkpoint_and_dispatch(
    model, checkpoint=weights_location, device_map=device_map
)

print(model.hf_device_map)

from mingpt.bpe import BPETokenizer
tokenizer = BPETokenizer()
inputs = tokenizer("Hello, my name is").to(0)

outputs = model.generate(inputs, max_new_tokens=10, do_sample=False)[0]
output = tokenizer.decode(outputs.cpu().squeeze())
print(output)
