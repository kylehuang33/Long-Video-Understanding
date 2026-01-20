from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/mnt/data-alpha-sg-02/team-agent/ai_glasses/models/Qwen2.5-72B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype="auto", device_map="auto", trust_remote_code=True
)

messages = [
    {"role": "user", "content": "Give me a concise summary of Qwen2.5 in one paragraph."}
]

if hasattr(tokenizer, "apply_chat_template"):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
else:
    inputs = tokenizer(messages[0]["content"], return_tensors="pt")

inputs = {k: v.to(model.device) for k, v in inputs.items()}
generated_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    pad_token_id=tokenizer.eos_token_id,
)
output_ids = generated_ids[0][inputs["input_ids"].shape[-1] :]
output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(output_text)
