"""
References:
- Claude Sonnet 5
- https://huggingface.co/facebook/sam3/blob/main/README.md

Adapted for BlueBEAR (CUDA).
"""


import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

parser = argparse.ArgumentParser()

parser.add_argument("video")
parser.add_argument("prompt")

args = parser.parse_args()

# Paths
PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal",
)

VIDEO_NAME = args.video
TEXT_PROMPT = args.prompt

FRAME_DIR = os.path.join(PROJECT_ROOT, "data_out", VIDEO_NAME, "frames")
MASK_DIR = os.path.join(PROJECT_ROOT, "data_out", VIDEO_NAME, "masks")

os.makedirs(MASK_DIR, exist_ok=True)

# Load device
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
device = "cuda"
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")


# Load model
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

model.eval()

print("Model loaded successfully.")

# Frame paths
frame_files = sorted(
    f for f in os.listdir(FRAME_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)

print(f"Found {len(frame_files)} frames.")

for i, frame_file in enumerate(frame_files):
    
    frame_path = os.path.join(FRAME_DIR, frame_file)
    mask_path = os.path.join(MASK_DIR, frame_file)
    
    # skip frames already processed
    if os.path.exists(mask_path):
        continue
    
    image = Image.open(frame_path).convert("RGB")
    
    inputs = processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.3,
        mask_threshold=0.3,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    # Save segmentation mask
    mask = results["segmentation"].cpu().numpy().astype(np.uint8)

    Image.fromarray(mask).save(mask_path)
    
    # Free GPU memory used by this frame
    del image
    del inputs
    del outputs
    del results
    del mask
    
    if device == "cuda":
        torch.cuda.empty_cache()

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(frame_files)} frames.")
        
print("Finished.")