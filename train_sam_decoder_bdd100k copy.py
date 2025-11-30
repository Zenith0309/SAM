import os
import cv2
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor


# ========================
#      BDD100K Dataset
# ========================
class BDDRoadDataset(Dataset):
   

    def __init__(
        self,
        img_dir,
        label_dir,
        split="train",
        img_ext=".jpg",
        label_ext=".png",
        label_suffix="_train_id",
        max_samples=None,
    ):
        self.img_dir = os.path.join(img_dir, split)
        self.label_dir = os.path.join(label_dir, split)

        self.img_paths = sorted(glob(os.path.join(self.img_dir, f"*{img_ext}")))
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.samples = []
        missing_labels = 0
        for img_path in self.img_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_name = f"{basename}{label_suffix}{label_ext}"
            label_path = os.path.join(self.label_dir, label_name)

            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))
            else:
                missing_labels += 1

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No labels matched for split '{split}'. Check label_dir={self.label_dir}."
            )

        if max_samples is not None:
            self.samples = self.samples[: max(1, max_samples)]

        if missing_labels > 0:
            print(
                f"Warning: {missing_labels} image(s) under {self.img_dir} are missing"
                f" corresponding labels in {self.label_dir}. They were skipped."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ========================
#        Dice Loss
# ========================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (H,W), raw
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


# ========================
#   SAM Forward Helper
# ========================
def sample_positive_point(fg_mask, deterministic=False):
    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0:
        return None
    if deterministic:
        idx = 0
    else:
        idx = np.random.randint(0, len(xs))
    point = np.array([[xs[idx], ys[idx]]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)
    return point, label


def select_class_ids(seg_label, max_classes=3, ignore_label=255, deterministic=False):
    class_ids = [int(cid) for cid in np.unique(seg_label) if cid != ignore_label]
    if len(class_ids) == 0:
        return []
    if max_classes is None or max_classes >= len(class_ids):
        selected = class_ids
    else:
        if deterministic:
            selected = sorted(class_ids)[:max_classes]
        else:
            selected = list(np.random.choice(class_ids, max_classes, replace=False))
    return selected


def forward_with_points(predictor, point, label, device):
    """Run prompt encoder + mask decoder for the already-set image."""
    transformed_point = predictor.transform.apply_coords(point, predictor.original_size)
    point_coords = torch.as_tensor(transformed_point, dtype=torch.float32, device=device)
    point_labels = torch.as_tensor(label, dtype=torch.int64, device=device)
    point_coords = point_coords[None, :, :]
    point_labels = point_labels[None, :]

    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
        points=(point_coords, point_labels), boxes=None, masks=None
    )

    low_res_masks, _ = predictor.model.mask_decoder(
        image_embeddings=predictor.features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    upscaled_masks = predictor.model.postprocess_masks(
        low_res_masks, predictor.input_size, predictor.original_size
    )

    return upscaled_masks[0, 0, :, :]


def align_logits_with_target(logits, target_shape):
    if logits.shape == target_shape:
        return logits
    logits_4d = logits.unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        logits_4d,
        size=target_shape,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0]


# ========================
#      Training Loop
# ========================
def train_one_epoch(
    sam,
    predictor,
    dataloader,
    optimizer,
    bce,
    dice,
    device,
    max_classes_per_image=3,
):
    sam.train()
    total_loss = 0.0
    total_steps = 0

    for img_path_batch, label_path_batch in tqdm(
        dataloader, desc="Train dataloader", leave=False
    ):
        # batch_size=1，这里取 [0]
        img_path = img_path_batch[0]
        label_path = label_path_batch[0]

        # ----- 读图像 -----
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ----- 读语义分割标签 (单通道, class id) -----
        seg_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if seg_label is None:
            print(f"Warning: label not found for {img_path}, skip.")
            continue

        class_ids = select_class_ids(seg_label, max_classes=max_classes_per_image)
        if not class_ids:
            continue

        # ----- 设置 SAM 输入 -----
        predictor.set_image(image)

        for class_id in class_ids:
            fg_mask = (seg_label == class_id).astype(np.uint8)
            if fg_mask.sum() == 0:
                continue

            point, label = sample_positive_point(fg_mask, deterministic=False)
            if point is None:
                continue
            pred_logits = forward_with_points(predictor, point, label, device)

            gt = torch.from_numpy(fg_mask).float().to(device)
            pred_logits = align_logits_with_target(pred_logits, gt.shape)

            loss_bce = bce(pred_logits, gt)
            loss_dice = dice(pred_logits, gt)
            loss = loss_bce + loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        #     print(
        #         f"[Train] img={os.path.basename(img_path)} class={class_id} "
        #         f"loss={loss.item():.4f}"
        #     )

    return total_loss / max(1, total_steps)


# ========================
#      Validation Loop
# ========================
def evaluate(
    sam,
    predictor,
    dataloader,
    bce,
    dice,
    device,
    max_classes_per_image=3,
):
    sam.eval()
    total_loss = 0.0
    dice_scores = 0.0
    iou_scores = 0.0
    valid_batches = 0

    smooth = dice.smooth

    with torch.no_grad():
        for img_path_batch, label_path_batch in tqdm(
            dataloader, desc="Val dataloader", leave=False
        ):
            img_path = img_path_batch[0]
            label_path = label_path_batch[0]

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            seg_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if seg_label is None:
                continue

            class_ids = select_class_ids(
                seg_label,
                max_classes=max_classes_per_image,
                deterministic=True,
            )
            if not class_ids:
                continue

            predictor.set_image(image)
            for class_id in class_ids:
                fg_mask = (seg_label == class_id).astype(np.uint8)
                if fg_mask.sum() == 0:
                    continue

                sample = sample_positive_point(fg_mask, deterministic=True)
                if sample is None:
                    continue
                point, label = sample
                pred_logits = forward_with_points(predictor, point, label, device)
                gt = torch.from_numpy(fg_mask).float().to(device)
                pred_logits = align_logits_with_target(pred_logits, gt.shape)

                loss = bce(pred_logits, gt) + dice(pred_logits, gt)
                total_loss += loss.item()

                probs = torch.sigmoid(pred_logits)
                pred_mask = (probs > 0.5).float()

                intersection = (pred_mask * gt).sum()
                dice_score = (2 * intersection + smooth) / (
                    pred_mask.sum() + gt.sum() + smooth
                )
                dice_scores += dice_score.item()

                union = pred_mask.sum() + gt.sum() - intersection
                iou = (intersection + smooth) / (union + smooth)
                iou_scores += iou.item()

                valid_batches += 1

                # print(
                #     f"[Val] img={os.path.basename(img_path)} class={class_id} "
                #     f"loss={loss.item():.4f} dice={dice_score.item():.4f} "
                #     f"iou={iou.item():.4f}"
                # )

    if valid_batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / valid_batches,
        dice_scores / valid_batches,
        iou_scores / valid_batches,
    )


# ========================
#           Main
# ========================
def main():
    # --------- 路径配置（按你自己的实际路径改）---------
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    BDD_ROOT = "/data1/tzs/segment-anything-main/dataset"    # <- 改成你的根路径
    IMG_DIR = os.path.join(BDD_ROOT, "10k")
    LABEL_DIR = os.path.join(BDD_ROOT, "labels")

    # --------- 训练超参数 ---------
    lr = 1e-4
    batch_size = 1          # SamPredictor 不适合大 batch，这里固定 1
    num_epochs = 5
    max_classes_per_image = 4
    debug_sample_count = 16  # 设置为 None 或 0 即可使用完整数据集
    if debug_sample_count in (None, 0):
        debug_sample_count = None
    debug_val_sample_count = debug_sample_count

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------- 数据集 & DataLoader ---------
    train_dataset = BDDRoadDataset(
        IMG_DIR,
        LABEL_DIR,
        split="train",
        max_samples=debug_sample_count,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,           # SAM predictor forward 只能用 1
        shuffle=True,
        num_workers=16,          # 根据 CPU 核数调整
        pin_memory=True,        # GPU 加速
        persistent_workers=True # 长时间训练更稳定
    )

    val_dataset = BDDRoadDataset(
        IMG_DIR,
        LABEL_DIR,
        split="val",
        max_samples=debug_val_sample_count,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    # --------- 加载 SAM 模型 ---------
    sam_checkpoint = os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    # 冻结 image_encoder & prompt_encoder
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    # 只训练 mask_decoder
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=lr)

    # SamPredictor 包一层
    predictor = SamPredictor(sam)

    # 损失函数
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_val_metric = -float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "sam_decoder_bdd_best.pth")

    # --------- 训练循环 ---------
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            sam,
            predictor,
            train_loader,
            optimizer,
            bce,
            dice,
            device,
            max_classes_per_image=max_classes_per_image,
        )
        val_loss, val_dice, val_iou = evaluate(
            sam,
            predictor,
            val_loader,
            bce,
            dice,
            device,
            max_classes_per_image=max_classes_per_image,
        )

        print(
            f"Epoch [{epoch+1}/{num_epochs}]  train_loss={avg_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  val_iou={val_iou:.4f}"
        )

        # 仅在验证指标提升时保存最优权重
        if val_dice > best_val_metric:
            best_val_metric = val_dice
            torch.save(sam.state_dict(), best_ckpt_path)
            print(f"  -> New best checkpoint saved to {best_ckpt_path} (val_dice={val_dice:.4f})")


if __name__ == "__main__":
    main()
