import os
import subprocess
import imageio_ffmpeg

PROJECT_ROOT = os.environ.get(
    "SAM3_PROJECT_ROOT",
    "/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal",
)
FRAMES_DIR = os.path.join(PROJECT_ROOT, "frames_out")
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "sam3_background_removed_test.mp4")
FPS = 30

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
print(f"Using ffmpeg at: {ffmpeg_path}")

subprocess.run(
    [ffmpeg_path, "-y", "-framerate", str(FPS),
     "-i", os.path.join(FRAMES_DIR, "frame_%05d.png"),
     "-c:v", "libx264", "-pix_fmt", "yuv420p", OUTPUT_VIDEO],
    check=True,
)