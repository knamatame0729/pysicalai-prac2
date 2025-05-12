from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch

# Download CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Get Images
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image = Image.open(requests.get(url, stream=True).raw)

texts = ["a photo of a cat", "a photo of a dog", "a photo of a robot"]

# Pre Process and prediction
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

# Display results
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.4f}")