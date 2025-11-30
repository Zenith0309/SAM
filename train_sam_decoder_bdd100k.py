import os
import argparse
import cv2
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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


def save_sam_checkpoint(sam, path):
    """Safely save SAM weights even if mask_decoder is wrapped in DDP."""
    original_decoder = None
    if isinstance(sam.mask_decoder, DDP):
        original_decoder = sam.mask_decoder
        sam.mask_decoder = original_decoder.module
    torch.save(sam.state_dict(), path)
    if original_decoder is not None:
        sam.mask_decoder = original_decoder


def run_dummy_backward_step(sam, optimizer):
    """Ensure one backward/step happens so DDP ranks stay in sync."""
    params = [p for p in sam.mask_decoder.parameters() if p.requires_grad]
    if not params:
        return
    dummy_loss = params[0].sum() * 0.0
    optimizer.zero_grad()
    dummy_loss.backward()
    optimizer.step()


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
    disable_tqdm=False,
):
    sam.train()
    total_loss = 0.0
    total_steps = 0

    for img_path_batch, label_path_batch in tqdm(
        dataloader,
        desc="Train dataloader",
        leave=False,
        disable=disable_tqdm,
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
            run_dummy_backward_step(sam, optimizer)
            total_steps += 1
            continue

        class_ids = select_class_ids(seg_label, max_classes=max_classes_per_image)
        if not class_ids:
            run_dummy_backward_step(sam, optimizer)
            total_steps += 1
            continue

        # ----- 设置 SAM 输入 -----
        predictor.set_image(image)

        loss_sum = None
        class_count = 0

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

            if loss_sum is None:
                loss_sum = loss
            else:
                loss_sum = loss_sum + loss
            class_count += 1

        if loss_sum is None or class_count == 0:
            run_dummy_backward_step(sam, optimizer)
            total_steps += 1
            continue

        loss_mean = loss_sum / class_count

        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        total_loss += loss_mean.item()
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
    disable_tqdm=False,
):
    sam.eval()
    total_loss = 0.0
    dice_scores = 0.0
    iou_scores = 0.0
    valid_batches = 0

    smooth = dice.smooth

    with torch.no_grad():
        for img_path_batch, label_path_batch in tqdm(
            dataloader,
            desc="Val dataloader",
            leave=False,
            disable=disable_tqdm,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SAM mask decoder on BDD100K")
    parser.add_argument("--bdd-root", default="/data1/tzs/segment-anything-main/dataset")
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-classes-per-image", type=int, default=4)
    parser.add_argument("--debug-train-samples", type=int, default=16, help="<=0 means use full train set")
    parser.add_argument("--debug-val-samples", type=int, default=16, help="<=0 means use full val set")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--sam-checkpoint", default=None, help="Path to pre-trained SAM weights")
    parser.add_argument("--best-ckpt-name", default="sam_decoder_bdd_best.pth")
    parser.add_argument("--distributed", action="store_true", help="Enable multi-GPU DDP training")
    parser.add_argument("--world-size", type=int, default=max(1, torch.cuda.device_count()))
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:29500")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep dataloader workers alive between epochs")
    return parser.parse_args()


def resolve_paths(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = args.checkpoint_dir or os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    sam_checkpoint = args.sam_checkpoint or os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
    best_ckpt_path = os.path.join(checkpoint_dir, args.best_ckpt_name)
    img_dir = os.path.join(args.bdd_root, "10k")
    label_dir = os.path.join(args.bdd_root, "labels")

    return {
        "project_root": project_root,
        "checkpoint_dir": checkpoint_dir,
        "sam_checkpoint": sam_checkpoint,
        "best_ckpt_path": best_ckpt_path,
        "img_dir": img_dir,
        "label_dir": label_dir,
    }


def create_dataloader(dataset, batch_size=1, num_workers=0, sampler=None, shuffle=True, persistent=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent and num_workers > 0,
    )


# ========================
#    Single GPU Training
# ========================
def run_single_gpu_training(args):
    cfg = resolve_paths(args)

    debug_train = None if args.debug_train_samples <= 0 else args.debug_train_samples
    debug_val = None if args.debug_val_samples <= 0 else args.debug_val_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = BDDRoadDataset(
        cfg["img_dir"],
        cfg["label_dir"],
        split="train",
        max_samples=debug_train,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        persistent=args.persistent_workers,
    )

    val_dataset = BDDRoadDataset(
        cfg["img_dir"],
        cfg["label_dir"],
        split="val",
        max_samples=debug_val,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        persistent=args.persistent_workers,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=cfg["sam_checkpoint"])
    sam.to(device)

    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr)
    predictor = SamPredictor(sam)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_val_metric = -float("inf")

    for epoch in range(args.num_epochs):
        avg_loss = train_one_epoch(
            sam,
            predictor,
            train_loader,
            optimizer,
            bce,
            dice,
            device,
            max_classes_per_image=args.max_classes_per_image,
        )
        val_loss, val_dice, val_iou = evaluate(
            sam,
            predictor,
            val_loader,
            bce,
            dice,
            device,
            max_classes_per_image=args.max_classes_per_image,
        )

        print(
            f"Epoch [{epoch+1}/{args.num_epochs}]  train_loss={avg_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  val_iou={val_iou:.4f}"
        )

        if val_dice > best_val_metric:
            best_val_metric = val_dice
            save_sam_checkpoint(sam, cfg["best_ckpt_path"])
            print(
                f"  -> New best checkpoint saved to {cfg['best_ckpt_path']} "
                f"(val_dice={val_dice:.4f})"
            )


# ========================
#   Distributed Training
# ========================
def distributed_worker(rank, world_size, args, cfg):
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank,
    )

    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

        debug_train = None if args.debug_train_samples <= 0 else args.debug_train_samples
        debug_val = None if args.debug_val_samples <= 0 else args.debug_val_samples

        train_dataset = BDDRoadDataset(
            cfg["img_dir"],
            cfg["label_dir"],
            split="train",
            max_samples=debug_train,
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            sampler=train_sampler,
            persistent=args.persistent_workers,
        )

        val_dataset = BDDRoadDataset(
            cfg["img_dir"],
            cfg["label_dir"],
            split="val",
            max_samples=debug_val,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            sampler=val_sampler,
            shuffle=False,
            persistent=args.persistent_workers,
        )

        if rank == 0:
            print(f"Distributed training with world_size={world_size}")
            print("Train samples:", len(train_dataset))
            print("Val samples:", len(val_dataset))

        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=cfg["sam_checkpoint"])
        sam.to(device)

        for p in sam.image_encoder.parameters():
            p.requires_grad = False
        for p in sam.prompt_encoder.parameters():
            p.requires_grad = False

        sam.mask_decoder = DDP(
            sam.mask_decoder,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr)
        predictor = SamPredictor(sam)

        bce = nn.BCEWithLogitsLoss()
        dice = DiceLoss()

        best_val_metric = -float("inf")

        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch)

            avg_loss = train_one_epoch(
                sam,
                predictor,
                train_loader,
                optimizer,
                bce,
                dice,
                device,
                max_classes_per_image=args.max_classes_per_image,
                disable_tqdm=rank != 0,
            )

            avg_loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = (avg_loss_tensor / world_size).item()

            val_loss, val_dice, val_iou = evaluate(
                sam,
                predictor,
                val_loader,
                bce,
                dice,
                device,
                max_classes_per_image=args.max_classes_per_image,
                disable_tqdm=rank != 0,
            )

            metrics_tensor = torch.tensor([val_loss, val_dice, val_iou], device=device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            val_loss, val_dice, val_iou = (metrics_tensor / world_size).tolist()

            if rank == 0:
                print(
                    f"[DDP] Epoch [{epoch+1}/{args.num_epochs}] train_loss={avg_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
                )
                if val_dice > best_val_metric:
                    best_val_metric = val_dice
                    save_sam_checkpoint(sam, cfg["best_ckpt_path"])
                    print(
                        f"  -> New best checkpoint saved to {cfg['best_ckpt_path']} "
                        f"(val_dice={val_dice:.4f})"
                    )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed_training(args):
    world_size = max(1, args.world_size)
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices")
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"world_size={world_size} exceeds available GPUs={torch.cuda.device_count()}"
        )
    cfg = resolve_paths(args)
    mp.spawn(
        distributed_worker,
        args=(world_size, args, cfg),
        nprocs=world_size,
        join=True,
    )


def main():
    args = parse_args()
    if args.distributed:
        run_distributed_training(args)
    else:
        run_single_gpu_training(args)


if __name__ == "__main__":
    main()
