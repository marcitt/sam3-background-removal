"""
References:
- Claude Sonnet 5
- https://huggingface.co/facebook/sam3
"""
import os
import subprocess
import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoConfig
from transformers.video_utils import load_video
import imageio_ffmpeg

import shutil

from datetime import datetime


PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/sam3-background-removal",
)

VIDEO = "C6_T1.MOV"

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data_in", VIDEO)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tmp", "frames_out")

video_name = os.path.splitext(VIDEO)[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "data_out", f"{video_name}_{timestamp}.mp4")

CONCEPTS = ["person", "violin", "violin bow"]

# CONCEPTS = ["person, violin, bow"]              

MAX_FRAMES = 500
FPS = 30

# how many pixels to grow the combined mask by, to close small gaps
# between adjacent concepts (e.g. hand gripping the bow). Start small —
# too high will eat into background around the true edges.
CLOSING_ITERATIONS = 3

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
device = "cuda"
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_config = Sam3VideoConfig.from_pretrained("facebook/sam3")

model_config.score_threshold_detection = 0.15   # confidence to start detecting an object at all
model_config.new_det_thresh = 0.7   # footage has a fixed, known set of objects,s o be strict about starting to track anything "new"

model_config.assoc_iou_thresh = 0.05             # how much overlap counts as "same object as last frame"

model_config.fill_hole_area = 32                # fills small internal gaps within a mask

model_config.hotstart_unmatch_thresh = 4 
model_config.min_trk_keep_alive = -1


model = Sam3VideoModel.from_pretrained("facebook/sam3", config=model_config,  dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print("Model loaded successfully.")

video_frames, _ = load_video(VIDEO_PATH)
video_frames = video_frames[:MAX_FRAMES]

# accumulated across all concepts
combined_masks = {}
# may be too large over time

def get_frame_shape():
    first = video_frames[0]
    arr = first if isinstance(first, np.ndarray) else np.array(first)
    return arr.shape[:2]  # (H, W)

# added downscaling
TARGET_H = 1080
orig_h, orig_w = get_frame_shape() 
scale = TARGET_H / orig_h
target_size = (int(orig_w * scale), TARGET_H)  # (W, H) for PIL resize

video_frames = [
    np.array(Image.fromarray(np.array(f)).resize(target_size, Image.BILINEAR))
    for f in video_frames
]

frame_h, frame_w = get_frame_shape()

for concept in CONCEPTS:

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session, text=concept
    )

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=MAX_FRAMES
    ):
        frame_idx = model_outputs.frame_idx #label
        
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        masks = processed_outputs["masks"].cpu().numpy() > 0.5  # (num_instances, H, W)

        # if the frame is not already stored in combined masks it gets stored as an empty array
        if frame_idx not in combined_masks:
            combined_masks[frame_idx] = np.zeros((frame_h, frame_w), dtype=bool) 

        # apply bitwise OR
        for mask in masks:
            combined_masks[frame_idx] |= mask
    
    # clear
    del inference_session
    torch.cuda.empty_cache()


def extract_mask(image, mask, background_color=(255, 255, 255)):
    image = image.convert("RGB")
    image_np = np.array(image)
    background = np.full_like(image_np, background_color, dtype=np.uint8)
    background[mask] = image_np[mask]
    return Image.fromarray(background)


print(f"\nApplying closing ({CLOSING_ITERATIONS} iterations) to close gaps...")
for frame_idx in combined_masks:
    combined_masks[frame_idx] = ndimage.binary_closing(
        combined_masks[frame_idx], iterations=CLOSING_ITERATIONS
    )


frame_count = 0
# sort masks in combined_masks in order of frame_idx
for frame_idx in sorted(combined_masks.keys()):
    frame_np = video_frames[frame_idx] #get the real video frame
    frame_image = (
        frame_np if isinstance(frame_np, Image.Image) else Image.fromarray(np.array(frame_np))
    )
    frame_image = frame_image.convert("RGB")

    result_image = extract_mask(frame_image, combined_masks[frame_idx])
    result_image.save(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.png"))
    frame_count += 1


# Merging frames
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run(
    [ffmpeg_path, "-y", "-framerate", str(FPS),
     "-i", os.path.join(OUTPUT_DIR, "frame_%05d.png"),
     "-c:v", "libx264", "-pix_fmt", "yuv420p", OUTPUT_VIDEO],
    check=True,
)
print(f"Saved final video to {OUTPUT_VIDEO}")

shutil.rmtree(OUTPUT_DIR)
print(f"Removed intermediate frames directory: {OUTPUT_DIR}")