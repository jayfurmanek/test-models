import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_id = "NataliaH/gpt2-tiny-shakespeare"

#device_map = {
#        "model.embed_tokens": "cpu",
#        "model.norm": "cpu",
#        "lm_head": "cpu",
#        }
#
#for i in range(1):
#    device_map[f"model.layers.{i}"] = "cuda:0"
#
#for i in range(1, 32):
#    device_map[f"model.layers.{i}"] = "cpu"


model = GPT2LMHeadModel.from_pretrained(
        model_id,
        #device_map="auto",
        #torch_dtype=torch.bfloat16,
        )

tokenizer = GPT2Tokenizer.from_pretrained(model_id)

print(model.device_map)

input_text = 'To be or not to be'
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
