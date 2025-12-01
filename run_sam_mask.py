import os
import argparse
import random
from glob import glob

import numpy as np
import torch
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def parse_args():
    parser = argparse.ArgumentParser("Run SAM mask generation on BDD test images")
    parser.add_argument(
        "--bdd-root",
        default=os.environ.get("BDD_ROOT", "dataset"),
        help="BDD100K dataset root (default: env BDD_ROOT or repo dataset)",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Explicit path to test images; overrides --bdd-root if provided",
    )
    parser.add_argument(
        "--output-dir",
        default="output_masks",
        help="Directory to store mask visualizations",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of test images to sample",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="checkpoints/sam_decoder_bdd_best.pth",
        help="Path to SAM checkpoint (fine-tuned weights)",
    )
    parser.add_argument(
        "--model-type",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type",
    )
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="Mask generator points_per_side",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.86,
        help="Mask generator pred_iou_thresh",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.92,
        help="Mask generator stability_score_thresh",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=100,
        help="Mask generator min_mask_region_area",
    )
    return parser.parse_args()


def visualize_with_opencv(image, anns, save_path):
    h, w = image.shape[:2]
    color_mask = np.zeros_like(image, dtype=np.uint8)

    for ann in anns:
        m = ann["segmentation"]
        color = np.random.randint(0, 255, size=3, dtype=np.uint8)
        color_mask[m] = color

    overlay = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Saved to", save_path)


def process_one_image(img_path, mask_generator, output_dir):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"Warning: failed to read {img_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)
    if len(masks) == 0:
        print(f"No masks generated for {img_path}")
        return

    stem = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"{stem}_mask.png")
    visualize_with_opencv(image_rgb, masks, save_path)


def main():
    args = parse_args()

    test_dir = args.test_dir or os.path.join(args.bdd_root, "10k", "test")
    if not os.path.isdir(test_dir):
        raise RuntimeError(f"Test directory not found: {test_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_area,
    )

    image_paths = sorted(glob(os.path.join(test_dir, "*.jpg")))
    if len(image_paths) == 0:
        raise RuntimeError(f"No test images found in {test_dir}")

    sample_count = min(args.num_samples, len(image_paths))
    sampled = random.sample(image_paths, sample_count)
    print(f"Processing {sample_count} test images ...")

    for img_path in sampled:
        process_one_image(img_path, mask_generator, args.output_dir)


if __name__ == "__main__":
    main()
