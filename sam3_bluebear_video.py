"""
References:
- Claude Sonnet 5
- https://huggingface.co/docs/transformers/model_doc/sam3_video
Adapted for BlueBEAR (CUDA).
"""

import os
import subprocess
import torch
import numpy as np
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoConfig
from transformers.video_utils import load_video
import imageio_ffmpeg

PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal",
)

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "test_video.mov")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "frames_out")
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "sam3_background_removed.mp4")

TEXT_PROMPT = "person, violin, bow"
MAX_FRAMES = 50
FPS = 30

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
device = "cuda"
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model_config = Sam3VideoConfig.from_pretrained("facebook/sam3")
model = Sam3VideoModel.from_pretrained("facebook/sam3", config=model_config).to(device)
processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

video_frames, _ = load_video(VIDEO_PATH)
print(f"Loaded {len(video_frames)} frames from {VIDEO_PATH}")

inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
)
inference_session = processor.add_text_prompt(
    inference_session=inference_session, text=TEXT_PROMPT
)


def extract_mask(image, masks, background_color=(255, 255, 255)):
    image = image.convert("RGB")
    image_np = np.array(image)
    background = np.full_like(image_np, background_color, dtype=np.uint8)
    masks = masks.cpu().numpy() > 0.5
    for mask in masks:
        background[mask] = image_np[mask]
    return Image.fromarray(background)


frame_count = 0
for model_outputs in model.propagate_in_video_iterator(
    inference_session=inference_session, max_frame_num_to_track=MAX_FRAMES
):
    frame_idx = model_outputs.frame_idx
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    masks = processed_outputs["masks"]

    frame_np = video_frames[frame_idx]
    frame_image = (
        frame_np if isinstance(frame_np, Image.Image) else Image.fromarray(np.array(frame_np))
    )
    frame_image = frame_image.convert("RGB")

    result_image = extract_mask(frame_image, masks)
    result_image.save(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}.png"))
    frame_count += 1

print(f"Finished processing {frame_count} frames.")

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run(
    [ffmpeg_path, "-y", "-framerate", str(FPS),
     "-i", os.path.join(OUTPUT_DIR, "frame_%05d.png"),
     "-c:v", "libx264", "-pix_fmt", "yuv420p", OUTPUT_VIDEO],
    check=True,
)

print(f"Saved final video to {OUTPUT_VIDEO}")