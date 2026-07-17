"""
Multi-concept video segmentation: runs each concept as a separate SAM3 video
tracking pass, merges masks per frame, applies dilation to close small gaps
between adjacent concepts (e.g. hand touching bow), then renders and stitches
the final video.
Adapted for BlueBEAR (CUDA).
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

PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal",
)
VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "IMG_5097")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "frames_out_combined_2")
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "sam3_multi_concept_3.mp4")

CONCEPTS = ["person", "violin", "violin bow"]
MAX_FRAMES = 50
FPS = 30

# how many pixels to grow the combined mask by, to close small gaps
# between adjacent concepts (e.g. hand gripping the bow). Start small —
# too high will eat into background around the true edges.
DILATION_ITERATIONS = 3

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
device = "cuda"
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_config = Sam3VideoConfig.from_pretrained("facebook/sam3")
model = Sam3VideoModel.from_pretrained("facebook/sam3", config=model_config).to(device)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print("Model loaded successfully.")

video_frames, _ = load_video(VIDEO_PATH)
print(f"Loaded {len(video_frames)} frames from {VIDEO_PATH}")

# frame_idx -> boolean mask (H, W), accumulated across all concepts
combined_masks = {}


def get_frame_shape():
    first = video_frames[0]
    arr = first if isinstance(first, np.ndarray) else np.array(first)
    return arr.shape[:2]  # (H, W)


frame_h, frame_w = get_frame_shape()

for concept in CONCEPTS:
    print(f"\n=== Tracking concept: '{concept}' ===")

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
        frame_idx = model_outputs.frame_idx
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        masks = processed_outputs["masks"].cpu().numpy() > 0.5  # (num_instances, H, W)

        if frame_idx not in combined_masks:
            combined_masks[frame_idx] = np.zeros((frame_h, frame_w), dtype=bool)

        for mask in masks:
            combined_masks[frame_idx] |= mask

    print(f"  -> tracked across {len(combined_masks)} frame(s) so far")


def extract_mask(image, mask, background_color=(255, 255, 255)):
    image = image.convert("RGB")
    image_np = np.array(image)
    background = np.full_like(image_np, background_color, dtype=np.uint8)
    background[mask] = image_np[mask]
    return Image.fromarray(background)


print(f"\nApplying dilation ({DILATION_ITERATIONS} iterations) to close gaps...")
for frame_idx in combined_masks:
    combined_masks[frame_idx] = ndimage.binary_dilation(
        combined_masks[frame_idx], iterations=DILATION_ITERATIONS
    )

print("Rendering merged frames...")
frame_count = 0
for frame_idx in sorted(combined_masks.keys()):
    frame_np = video_frames[frame_idx]
    frame_image = (
        frame_np if isinstance(frame_np, Image.Image) else Image.fromarray(np.array(frame_np))
    )
    frame_image = frame_image.convert("RGB")

    result_image = extract_mask(frame_image, combined_masks[frame_idx])
    result_image.save(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.png"))
    frame_count += 1

print(f"Rendered {frame_count} merged frames.")

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run(
    [ffmpeg_path, "-y", "-framerate", str(FPS),
     "-i", os.path.join(OUTPUT_DIR, "frame_%05d.png"),
     "-c:v", "libx264", "-pix_fmt", "yuv420p", OUTPUT_VIDEO],
    check=True,
)
print(f"Saved final video to {OUTPUT_VIDEO}")