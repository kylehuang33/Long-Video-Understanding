import os
import base64
from mimetypes import guess_type
from openai import OpenAI

AZURE_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")  # e.g. https://YOUR-RESOURCE-NAME.openai.azure.com
# AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]          # the *deployment name* in Azure AI Foundry / OpenAI Studio
AZURE_DEPLOYMENT="gpt-5.1-chat"



def local_image_to_data_url(image_path: str) -> str:
    mime_type, _ = guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

client = OpenAI(
    api_key=AZURE_API_KEY,
    base_url=f"{AZURE_ENDPOINT}/openai/v1/",
)

# resp = client.chat.completions.create(
#     model=AZURE_DEPLOYMENT,  # deployment name
#     messages=[
#         {"role": "user", "content": "Explain the difference between OpenAI and Azure OpenAI in 20 words."}
#     ],
# )

# print(resp.choices[0].message.content)
# print(resp.model_dump_json(indent=2))


image_data_url = local_image_to_data_url("/home/y84401399/DOING_PROJECTS/video_understanding/MyGGStream/analysis_paper/qualitiative_result/448/frames_ovobench/multi_task_baseline_wrong/video_7a0cbfad-7cb3-44bd-9f0a-926faf5b3479/frame_001_idx00017.jpg")

# resp = client.responses.create(
#     model="gpt-5.1-chat",  # choose a vision-capable model you have access to
#     input=[{
#         "role": "user",
#         "content": [
#             {"type": "input_text", "text": "Write a detailed caption for this image."},
#             {"type": "input_image", "image_url": image_data_url},
#         ],
#     }],
# )

resp = client.chat.completions.create(
    model="gpt-5.1-chat",  # deployment name
    messages=[
        {"role": "system", "content": "You are a precise image captioning assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Write a detailed caption for this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        # optional: "detail": "low" | "high" | "auto"
                    },
                },
            ],
        },
    ],
)

print(resp.choices[0].message.content)
