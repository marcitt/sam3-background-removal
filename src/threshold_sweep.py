"""
Standalone threshold sweep for per-concept SAM3 image segmentation.

Runs Sam3Model/Sam3Processor once per (threshold, mask_threshold) combination
for a single concept prompt against a single image, and logs a connected-component
count (messiness proxy) plus mask pixel coverage for each combo to a CSV.

Does not touch video.py or run_frame_segment.sh — standalone comparison tool.

Usage:
    python threshold_sweep.py IMG_5097 "violin" \
        --thresholds 0.2 0.3 0.4 0.5 \
        --mask_thresholds 0.2 0.3 0.4 0.5
"""

import argparse
import csv
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from transformers import Sam3Model, Sam3Processor

MODEL_ID = "facebook/sam3"


def count_components(binary_mask: np.ndarray) -> int:
    labeled, n = ndimage.label(binary_mask) # type: ignore
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_id", help="Image identifier, e.g. IMG_5097")
    parser.add_argument("concept", help="Text prompt for this sweep, e.g. 'violin'")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--mask_thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--data_in", default="data_in")
    parser.add_argument("--data_out", default="data_out/threshold_sweep")
    args = parser.parse_args()

    os.makedirs(args.data_out, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — refusing to silently fall back to CPU")
    device = "cuda"

    image_path = os.path.join(args.data_in, f"{args.image_id}.jpg")
    image = Image.open(image_path).convert("RGB")

    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # Text embeddings are constant across the sweep — compute once.
    inputs = processor(images=image, text=args.concept, return_tensors="pt").to(device)

    results_path = os.path.join(args.data_out, f"{args.image_id}_{args.concept.replace(' ', '_')}_sweep.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "threshold", "mask_threshold", "n_components", "mask_pixel_fraction"])

        for det_thresh in args.thresholds:
            for mask_thresh in args.mask_thresholds:
                with torch.no_grad():
                    outputs = model(**inputs)

                processed = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=det_thresh,
                    mask_threshold=mask_thresh,
                    target_sizes=[image.size[::-1]],
                )[0]

                if len(processed["masks"]) == 0:
                    n_components = 0
                    pixel_fraction = 0.0
                else:
                    # Union of all instance masks returned for this concept/pass.
                    combined = np.zeros(image.size[::-1], dtype=bool)
                    for m in processed["masks"]:
                        combined |= m.cpu().numpy().astype(bool)
                    n_components = count_components(combined)
                    pixel_fraction = combined.mean()

                    out_name = f"{args.image_id}_{args.concept.replace(' ', '_')}_t{det_thresh}_mt{mask_thresh}.png"
                    Image.fromarray((combined * 255).astype(np.uint8)).save(
                        os.path.join(args.data_out, out_name)
                    )

                writer.writerow([args.concept, det_thresh, mask_thresh, n_components, f"{pixel_fraction:.4f}"])
                print(f"threshold={det_thresh} mask_threshold={mask_thresh} -> "
                      f"components={n_components} pixel_fraction={pixel_fraction:.4f}")

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()