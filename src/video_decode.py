import os
import subprocess
import argparse
import imageio_ffmpeg

PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/sam3-background-removal",
)

parser = argparse.ArgumentParser(
    description="Extract frames from a video using FFmpeg."
)

parser.add_argument(
    "video",
    help="Video filename or path."
)

args = parser.parse_args()

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data_in", args.video)

video_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_out", video_name)
FRAME_DIR = os.path.join(OUTPUT_DIR, "frames")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([
    ffmpeg_path,
    "-i", VIDEO_PATH,
    "-vsync", "0",
    os.path.join(FRAME_DIR, "frame_%06d.png"),
], check=True)