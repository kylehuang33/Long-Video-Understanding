import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_id = "Qwen/Qwen3-VL-4B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

img_path = "/home/y84401399/DOING_PROJECTS/video_understanding/MyGGStream/analysis_paper/qualitiative_result/448/frames_ovobench/multi_task_baseline_wrong/video_7a0cbfad-7cb3-44bd-9f0a-926faf5b3479/frame_002_idx00035.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},   # can be a local path
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# 1) Build the text prompt (NO tokenization here)
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 2) Load/prepare vision inputs (this is what actually reads the image)
image_inputs, video_inputs = process_vision_info(messages)

# 3) Create final model inputs
inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to(model.device)

# 4) Generate
generated_ids = model.generate(**inputs, max_new_tokens=256)

# 5) Trim the prompt tokens and decode only new tokens
new_tokens = generated_ids[:, inputs.input_ids.shape[1]:]
output_text = processor.batch_decode(new_tokens, skip_special_tokens=True)
print(output_text[0])
