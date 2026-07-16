"""
Prompt trial — runs a single text prompt against the test image,
saving output named after the prompt for easy comparison across runs.
Adapted for BlueBEAR (CUDA).
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model

PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal",
)
IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "test_3.png")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prompt_trials")

if len(sys.argv) < 2:
    raise SystemExit("Usage: python sam3_prompt_trial.py \"your, text, prompt\"")
TEXT_PROMPT = sys.argv[1]

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
device = "cuda"
print(f"Using device: {device}")
print(f"Prompt: '{TEXT_PROMPT}'")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.3,
    mask_threshold=0.3,
    target_sizes=inputs.get("original_sizes").tolist(),
)[0]

n_masks = results["masks"].shape[0]
print(f"{n_masks} mask(s) found")


def extract_mask(image, masks, background_color=(255, 255, 255)):
    image = image.convert("RGB")
    image_np = np.array(image)
    background = np.full_like(image_np, background_color, dtype=np.uint8)
    masks = masks.cpu().numpy() > 0.5
    for mask in masks:
        background[mask] = image_np[mask]
    return Image.fromarray(background)


# turn "person, violin, bow" into "person_violin_bow" for a clean filename
safe_name = TEXT_PROMPT.replace(",", "").replace(" ", "_")

result_image = extract_mask(image, results["masks"])
out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
result_image.save(out_path)
print(f"Saved to {out_path}")