from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load the model

## TODO Check for cuda / else use CPU 

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    device_map={"": "cuda"},  # Use "cuda" for NVIDIA GPUs
)

# Load your image

image = Image.open("image.jpg")

# 1. Image Captioning

print("Short caption:")
print(model.caption(image, length="short")["caption"])

print("Detailed caption:")
for t in model.caption(image, length="normal", stream=True)["caption"]:
    print(t, end="", flush=True)

# 2. Visual Question Answering

print("Asking questions about the image:")
print(model.query(image, "How many people are in the image?")["answer"])

# 3. Object Detection

print("Detecting objects:")
objects = model.detect(image, "face")["objects"]
print(f"Found {len(objects)} face(s)")

# 4. Visual Pointing

print("Locating objects:")
points = model.point(image, "person")["points"]
print(f"Found {len(points)} person(s)")


image = Image.open("image.jpg")
encoded_image = model.encode_image(image)

# Reuse the encoded image for each inference

print(model.caption(encoded_image, length="short")["caption"])
print(model.query(encoded_image, "How many people are in the image?")["answer"])