"""
References:
- Claude Sonnet 5
- https://huggingface.co/facebook/sam3/blob/main/README.md

Adapted for BlueBEAR (CUDA).
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib
from transformers import Sam3Processor, Sam3Model

PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal",
)
IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "test_3.png")
TEXT_PROMPT = "person and violins"

# Cuda on BlueBEAR, falls back to MPS or CPU if tested elsewhere
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version (torch built against): {torch.version.cuda}")

#   export HF_HUB_CACHE=/rds/projects/x/yourprojectname/hf_cache
print(
    f"HF cache location: {os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub (default)')}"
)

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")
print("Model loaded successfully.")

# load image - images need to be in RGB format (requires fixed input shape - 3 channels)
image = Image.open(IMAGE_PATH).convert("RGB")

# segment using text prompt
inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.3,
    mask_threshold=0.3,
    target_sizes=inputs.get("original_sizes").tolist(),
)[0]


# helper provided by sam3
def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)  # type: ignore
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]
    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


# claude sonnet 5 helper function
def extract_mask(image, masks):
    background_color = (255, 255, 255)
    image = image.convert("RGB")
    image_np = np.array(image)
    background = np.full_like(image_np, background_color, dtype=np.uint8)
    masks = masks.cpu().numpy() > 0.5  # convert once, to boolean, before the loop
    for mask in masks:
        background[mask] = image_np[mask]
    return Image.fromarray(background)


result_image = overlay_masks(image, results["masks"])
result_image.save(os.path.join(PROJECT_ROOT, "sam3_overlay.png"))

result_image = extract_mask(image, results["masks"])
result_image.save(os.path.join(PROJECT_ROOT, "sam3_background_removal.png"))

print("Saved segmented image to sam3_overlay.png and sam3_background_removal.png")
