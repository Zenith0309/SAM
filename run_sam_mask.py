import os
import random
from glob import glob
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ---------------- 基础配置 ----------------
BDD_ROOT = "/data1/tzs/segment-anything-main/dataset"
TEST_DIR = os.path.join(BDD_ROOT, "10k", "test")
OUTPUT_DIR = "output_masks_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SAMPLES = 5
sam_checkpoint = "checkpoints/sam_decoder_bdd_best.pth"  # 使用微调得到的最优权重
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 加载 SAM ----------------
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    min_mask_region_area=100,
)


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


def process_one_image(img_path):
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
    save_path = os.path.join(OUTPUT_DIR, f"{stem}_mask.png")
    visualize_with_opencv(image_rgb, masks, save_path)


def main():
    image_paths = sorted(glob(os.path.join(TEST_DIR, "*.jpg")))
    if len(image_paths) == 0:
        raise RuntimeError(f"No test images found in {TEST_DIR}")

    sample_count = min(NUM_SAMPLES, len(image_paths))
    sampled = random.sample(image_paths, sample_count)
    print(f"Processing {sample_count} test images ...")

    for img_path in sampled:
        process_one_image(img_path)


if __name__ == "__main__":
    main()
