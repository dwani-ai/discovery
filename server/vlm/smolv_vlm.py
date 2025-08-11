import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load images
image = load_image("https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg")

# Initialize processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
).to(DEVICE)

# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

# Generate outputs
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
"""
Assistant: The image depicts a large, historic statue of liberty, located in New York City. The statue is a green, cylindrical structure with a human figure at the top, holding a torch. The statue is situated on a pedestal that resembles the statue of liberty, which is located on a small island in the middle of a body of water. The water surrounding the island is calm, reflecting the blue sky and the statue.
In the background, there are several tall buildings, including the Empire State Building, which is visible in the distance. These buildings are made of glass and steel, and they are positioned in a grid-like pattern, giving them a modern look. The sky is clear, with a few clouds visible, indicating fair weather.
The statue is surrounded by trees, which are green and appear to be healthy. There are also some small structures, possibly houses or buildings, visible in the distance. The overall scene suggests a peaceful and serene environment, typical of a cityscape.
The image is taken during the daytime, likely during the day of the statue's installation. The lighting is bright, casting a strong shadow on the statue and the water, which enhances the visibility of the statue and the surrounding environment.
To summarize, the image captures a significant historical statue of liberty, situated on a small island in the middle of a body of water, surrounded by trees and buildings. The sky is clear, with a few clouds visible, indicating fair weather. The statue is green and cylindrical, with a human figure holding a torch, and is surrounded by trees, indicating a peaceful and well-maintained environment. The overall scene is one of tranquility and historical significance.
"""
