"""
References:
- Claude Sonnet 5
- https://huggingface.co/facebook/sam3

ARCHITECTURE NOTE (streaming rewrite):
Previously this script used load_video() to decode the entire video into memory
upfront, then ran 3 separate full passes (one per concept) over the stored frames.
This caused OOMs on full-length 4K footage, since even after downscaling, the raw
4K decode had to complete and be held in memory before any resizing happened.

This version decodes the video ONCE, frame by frame, via PyAV directly. For each
frame: downscale immediately, then run all 3 concept sessions on that single frame,
merge their masks, composite, and write the PNG straight away — before moving to
the next frame. At any moment, only one frame's data is in memory, regardless of
video length.

Tradeoff (see DECISIONS.md): this uses SAM3's "streaming" inference mode, which
disables hotstart heuristics that clean up duplicate/false-positive object tracks
(those heuristics need access to future frames, which streaming mode doesn't have
by design). Expect slightly noisier masks than the old pre-loaded approach in
exchange for the memory safety.
"""
import os
import subprocess
import torch
import numpy as np
import av
from PIL import Image
from scipy import ndimage
from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoConfig
import imageio_ffmpeg

import shutil

from datetime import datetime


PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/sam3-background-removal",
)

VIDEO = "IMG_5097.mov"

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data_in", VIDEO)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tmp", "frames_out")

video_name = os.path.splitext(VIDEO)[0]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "data_out", f"{video_name}_{timestamp}.mp4")

CONCEPTS = ["person", "violin", "violin bow"]

# CONCEPTS = ["person, violin, bow"]              

MAX_FRAMES = 1292  # cap on how many frames to process; safe to set this to the
# video's actual frame count or higher now, since frames are streamed one at a
# time rather than all decoded upfront
FPS = 30

# how many pixels to grow the combined mask by, to close small gaps
# between adjacent concepts (e.g. hand gripping the bow). Start small —
# too high will eat into background around the true edges.
CLOSING_ITERATIONS = 3

TARGET_H = 1080  # downscale target — SAM3 doesn't need native 4K to segment

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

model_config.hotstart_unmatch_thresh = 4  # NOTE: inert in streaming mode — hotstart
# heuristics (which this config feeds into) are disabled when frames are provided
# one-by-one, since they require future-frame lookahead. Kept here since it's a
# harmless no-op now, not because it does anything in this mode.
model_config.min_trk_keep_alive = -1


model = Sam3VideoModel.from_pretrained("facebook/sam3", config=model_config, low_cpu_mem_usage=True).to(device)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")
print("Model loaded successfully.")


def iter_downscaled_frames(video_path, max_frames, target_h):
    """
    Decodes the video one frame at a time via PyAV and downscales each frame
    immediately, before the next frame is decoded. Never holds more than one
    frame in memory. Yields (frame_idx, frame_array, target_w) tuples.
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    orig_h, orig_w = stream.height, stream.width
    scale = target_h / orig_h
    target_size = (int(orig_w * scale), target_h)  # (W, H) for PIL resize

    count = 0
    for frame in container.decode(video=0):
        if count >= max_frames:
            break
        img = frame.to_image().resize(target_size, Image.BILINEAR)
        yield count, np.array(img), target_size[0]
        count += 1
    container.close()


def extract_mask(image, mask, background_color=(255, 255, 255)):
    image = image.convert("RGB")
    image_np = np.array(image)
    background = np.full_like(image_np, background_color, dtype=np.uint8)
    background[mask] = image_np[mask]
    return Image.fromarray(background)


# one streaming session per concept, all initialized empty and kept open for
# the whole video — this is what lets tracking stay continuous across frames
# even though frames arrive one at a time
sessions = {}
for concept in CONCEPTS:
    session = processor.init_video_session(
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
    )
    session = processor.add_text_prompt(inference_session=session, text=concept)
    sessions[concept] = session

frame_count = 0
frame_h = TARGET_H
frame_w = None  # set on first frame

for frame_idx, frame_arr, target_w in iter_downscaled_frames(VIDEO_PATH, MAX_FRAMES, TARGET_H):
    if frame_w is None:
        frame_w = target_w

    combined_mask = np.zeros((frame_h, frame_w), dtype=bool)

    for concept in CONCEPTS:
        session = sessions[concept]

        inputs = processor(images=frame_arr, device=device, return_tensors="pt")
        model_outputs = model(
            inference_session=session, frame=inputs.pixel_values[0], reverse=False
        )
        processed_outputs = processor.postprocess_outputs(
            session, model_outputs, original_sizes=inputs.original_sizes
        )
        masks = processed_outputs["masks"].cpu().numpy() > 0.5  # (num_instances, H, W)

        for mask in masks:
            combined_mask |= mask

    # apply closing per-frame now, since we no longer hold all frames'
    # masks at once to batch this step at the end
    combined_mask = ndimage.binary_closing(combined_mask, iterations=CLOSING_ITERATIONS)

    frame_image = Image.fromarray(frame_arr).convert("RGB")
    result_image = extract_mask(frame_image, combined_mask)
    result_image.save(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.png"))
    frame_count += 1

    if (frame_idx + 1) % 50 == 0:
        print(f"Processed {frame_idx + 1} frames...")

for session in sessions.values():
    del session
torch.cuda.empty_cache()

print(f"\nProcessed {frame_count} frames total.")

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