from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate."},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"}
]

# Apply the chat template to format the messages
formatted_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)

print(formatted_chat["input_ids"])
print(tokenizer.decode(formatted_chat["input_ids"]))
