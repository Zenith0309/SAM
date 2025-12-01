import os
import argparse
import cv2
import numpy as np
from glob import glob
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class BDDRoadTensorDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        split: str = "train",
        img_ext: str = ".jpg",
        label_ext: str = ".png",
        label_suffix: str = "_train_id",
        max_samples: int = None,
    ) -> None:
        self.img_dir = os.path.join(img_dir, split)
        self.label_dir = os.path.join(label_dir, split)

        img_paths = sorted(glob(os.path.join(self.img_dir, f"*{img_ext}")))
        if not img_paths:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.samples: List[Tuple[str, str]] = []
        for img_path in img_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, f"{basename}{label_suffix}{label_ext}")
            if os.path.exists(label_path):
                self.samples.append((img_path, label_path))

        if not self.samples:
            raise RuntimeError(
                f"No labels matched for split '{split}'. Check label_dir={self.label_dir}."
            )

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[: max(1, max_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        img_path, label_path = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        seg_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if seg_label is None:
            raise RuntimeError(f"Failed to read label {label_path}")

        return {
            "image": image,
            "label": seg_label,
            "image_path": img_path,
        }


def collate_samples(samples: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    return samples


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1 - dice


def save_sam_checkpoint(sam, path: str):
    original_decoder = None
    if isinstance(sam.mask_decoder, DDP):
        original_decoder = sam.mask_decoder
        sam.mask_decoder = original_decoder.module
    torch.save(sam.state_dict(), path)
    if original_decoder is not None:
        sam.mask_decoder = original_decoder


def run_dummy_backward_step(sam, optimizer):
    params = [p for p in sam.mask_decoder.parameters() if p.requires_grad]
    if not params:
        return
    dummy_loss = params[0].sum() * 0.0
    optimizer.zero_grad()
    dummy_loss.backward()
    optimizer.step()


def sample_positive_point(mask: np.ndarray, deterministic: bool = False):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    idx = 0 if deterministic else np.random.randint(0, len(xs))
    point = np.array([[xs[idx], ys[idx]]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)
    return point, label


def select_class_ids(
    seg_label: np.ndarray,
    max_classes: int = 3,
    ignore_label: int = 255,
    deterministic: bool = False,
) -> List[int]:
    class_ids = [int(c) for c in np.unique(seg_label) if c != ignore_label]
    if not class_ids:
        return []
    if max_classes is None or max_classes >= len(class_ids):
        return class_ids
    if deterministic:
        return sorted(class_ids)[:max_classes]
    chosen = np.random.choice(class_ids, max_classes, replace=False)
    return list(int(x) for x in chosen)


def preprocess_batch(
    samples: List[Dict[str, np.ndarray]],
    sam,
    device: torch.device,
    resize_transform: ResizeLongestSide,
):
    processed_images = []
    meta = []
    for sample in samples:
        image = sample["image"]
        seg = sample["label"]
        original_size = image.shape[:2]
        transformed = resize_transform.apply_image(image)
        input_size = transformed.shape[:2]

        image_torch = torch.as_tensor(transformed, device=device)
        image_torch = image_torch.permute(2, 0, 1).contiguous().float() / 255.0
        with torch.no_grad():
            preprocessed = sam.preprocess(image_torch)
        processed_images.append(preprocessed)
        meta.append(
            {
                "original_size": original_size,
                "input_size": input_size,
                "label": torch.from_numpy(seg).float().to(device),
                "label_np": seg,
            }
        )

    image_batch = torch.stack(processed_images, dim=0)
    with torch.no_grad():
        embeddings = sam.image_encoder(image_batch)
    return embeddings, meta


def train_one_epoch(
    sam,
    dataloader,
    optimizer,
    bce,
    dice_loss,
    device,
    max_classes_per_image: int,
):
    sam.train()
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    dense_pe = sam.prompt_encoder.get_dense_pe()
    total_loss = 0.0
    total_steps = 0

    for samples in dataloader:
        image_embeddings, meta = preprocess_batch(samples, sam, device, resize_transform)
        batch_loss = None
        class_counter = 0

        for idx, sample in enumerate(samples):
            seg_np = meta[idx]["label_np"]
            class_ids = select_class_ids(seg_np, max_classes=max_classes_per_image)
            if not class_ids:
                continue

            original_size = meta[idx]["original_size"]
            input_size = meta[idx]["input_size"]
            seg_tensor = meta[idx]["label"]
            image_embedding = image_embeddings[idx : idx + 1]

            for class_id in class_ids:
                fg_mask = (seg_np == class_id).astype(np.uint8)
                sample_point = sample_positive_point(fg_mask, deterministic=False)
                if sample_point is None:
                    continue
                point, label = sample_point
                point_coords = resize_transform.apply_coords(point, original_size)
                point_coords = torch.as_tensor(point_coords, dtype=torch.float32, device=device)
                point_labels = torch.as_tensor(label, dtype=torch.int64, device=device)
                point_coords = point_coords[None, :, :]
                point_labels = point_labels[None, :]

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=(point_coords, point_labels), boxes=None, masks=None
                )
                low_res_masks, _ = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=dense_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = sam.postprocess_masks(
                    low_res_masks,
                    input_size=input_size,
                    original_size=original_size,
                )
                pred_logits = upscaled_masks[:, 0, :, :]
                gt = (seg_tensor == float(class_id)).float()

                loss = bce(pred_logits, gt) + dice_loss(pred_logits, gt)
                batch_loss = loss if batch_loss is None else batch_loss + loss
                class_counter += 1

        if batch_loss is None or class_counter == 0:
            run_dummy_backward_step(sam, optimizer)
            continue

        loss_mean = batch_loss / class_counter
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        total_loss += loss_mean.item()
        total_steps += 1

    return total_loss / max(1, total_steps)


def evaluate(
    sam,
    dataloader,
    bce,
    dice_loss,
    device,
    max_classes_per_image: int,
):
    sam.eval()
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    dense_pe = sam.prompt_encoder.get_dense_pe()
    total_loss = 0.0
    dice_scores = 0.0
    iou_scores = 0.0
    steps = 0

    with torch.no_grad():
        for samples in dataloader:
            image_embeddings, meta = preprocess_batch(samples, sam, device, resize_transform)
            for idx, sample in enumerate(samples):
                seg_np = meta[idx]["label_np"]
                class_ids = select_class_ids(
                    seg_np,
                    max_classes=max_classes_per_image,
                    deterministic=True,
                )
                if not class_ids:
                    continue

                original_size = meta[idx]["original_size"]
                input_size = meta[idx]["input_size"]
                seg_tensor = meta[idx]["label"]
                image_embedding = image_embeddings[idx : idx + 1]

                for class_id in class_ids:
                    fg_mask = (seg_np == class_id).astype(np.uint8)
                    sample_point = sample_positive_point(fg_mask, deterministic=True)
                    if sample_point is None:
                        continue

                    point, label = sample_point
                    point_coords = resize_transform.apply_coords(point, original_size)
                    point_coords = torch.as_tensor(point_coords, dtype=torch.float32, device=device)
                    point_labels = torch.as_tensor(label, dtype=torch.int64, device=device)
                    point_coords = point_coords[None, :, :]
                    point_labels = point_labels[None, :]

                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=(point_coords, point_labels), boxes=None, masks=None
                    )
                    low_res_masks, _ = sam.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=dense_pe,
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    upscaled_masks = sam.postprocess_masks(
                        low_res_masks,
                        input_size=input_size,
                        original_size=original_size,
                    )
                    pred_logits = upscaled_masks[:, 0, :, :]
                    gt = (seg_tensor == float(class_id)).float()

                    loss = bce(pred_logits, gt) + dice_loss(pred_logits, gt)
                    total_loss += loss.item()

                    probs = torch.sigmoid(pred_logits)
                    pred_mask = (probs > 0.5).float()
                    intersection = (pred_mask * gt).sum()
                    dice = (2 * intersection + dice_loss.smooth) / (
                        pred_mask.sum() + gt.sum() + dice_loss.smooth
                    )
                    union = pred_mask.sum() + gt.sum() - intersection
                    iou = (intersection + dice_loss.smooth) / (union + dice_loss.smooth)

                    dice_scores += dice.item()
                    iou_scores += iou.item()
                    steps += 1

    return total_loss, dice_scores, iou_scores, steps


def parse_args():
    parser = argparse.ArgumentParser("Batched SAM decoder finetuning")
    parser.add_argument("--bdd-root", default="/data1/tzs/segment-anything-main/dataset")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-classes-per-image", type=int, default=4)
    parser.add_argument("--debug-train-samples", type=int, default=0)
    parser.add_argument("--debug-val-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--best-ckpt-name", default="sam_decoder_bdd_batched_best.pth")
    parser.add_argument("--distributed", action="store_true", help="Enable DDP training")
    parser.add_argument("--world-size", type=int, default=max(1, torch.cuda.device_count()))
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:29502")
    return parser.parse_args()


def build_dataset(bdd_root: str, split: str, max_samples: int):
    img_dir = os.path.join(bdd_root, "10k")
    label_dir = os.path.join(bdd_root, "labels")
    max_samples = None if max_samples is None or max_samples <= 0 else max_samples
    return BDDRoadTensorDataset(img_dir, label_dir, split=split, max_samples=max_samples)


def create_dataloader(
    dataset,
    batch_size,
    num_workers,
    sampler=None,
    shuffle=True,
    persistent=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_samples,
        persistent_workers=persistent and num_workers > 0,
    )


def build_dataloaders(args):
    train_ds = build_dataset(args.bdd_root, "train", args.debug_train_samples)
    val_ds = build_dataset(args.bdd_root, "val", args.debug_val_samples)
    train_loader = create_dataloader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent=args.persistent_workers,
        shuffle=True,
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent=args.persistent_workers,
        shuffle=False,
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def resolve_paths(args):
    project_root = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = args.checkpoint_dir or os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    sam_ckpt = args.sam_checkpoint or os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
    best_ckpt = os.path.join(checkpoint_dir, args.best_ckpt_name)
    return {
        "checkpoint_dir": checkpoint_dir,
        "sam_ckpt": sam_ckpt,
        "best_ckpt": best_ckpt,
    }


def run_single_training(args, paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_len, val_len = build_dataloaders(args)
    print(f"Train samples: {train_len}")
    print(f"Val samples: {val_len}")

    sam = sam_model_registry["vit_h"](checkpoint=paths["sam_ckpt"])
    sam.to(device)
    for p in sam.image_encoder.parameters():
        p.requires_grad = False
    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    best_metric = -float("inf")
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(
            sam,
            train_loader,
            optimizer,
            bce,
            dice_loss,
            device,
            max_classes_per_image=args.max_classes_per_image,
        )
        val_loss_sum, val_dice_sum, val_iou_sum, val_steps = evaluate(
            sam,
            val_loader,
            bce,
            dice_loss,
            device,
            max_classes_per_image=args.max_classes_per_image,
        )
        val_steps = max(1, val_steps)
        val_loss = val_loss_sum / val_steps
        val_dice = val_dice_sum / val_steps
        val_iou = val_iou_sum / val_steps
        print(
            f"Epoch [{epoch+1}/{args.num_epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
        )
        if val_dice > best_metric:
            best_metric = val_dice
            save_sam_checkpoint(sam, paths["best_ckpt"])
            print(
                f"  -> New best checkpoint saved to {paths['best_ckpt']} "
                f"(val_dice={val_dice:.4f})"
            )


def distributed_worker(rank, world_size, args, paths):
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=world_size,
        rank=rank,
    )
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)

        train_dataset = build_dataset(args.bdd_root, "train", args.debug_train_samples)
        val_dataset = build_dataset(args.bdd_root, "val", args.debug_val_samples)

        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

        train_loader = create_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=train_sampler,
            persistent=args.persistent_workers,
            shuffle=True,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=val_sampler,
            persistent=args.persistent_workers,
            shuffle=False,
        )

        if rank == 0:
            print(f"Distributed training with world_size={world_size}")
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")

        sam = sam_model_registry["vit_h"](checkpoint=paths["sam_ckpt"])
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
            find_unused_parameters=False,
        )

        optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=args.lr)
        bce = nn.BCEWithLogitsLoss()
        dice_loss = DiceLoss()

        best_metric = -float("inf")
        for epoch in range(args.num_epochs):
            train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                sam,
                train_loader,
                optimizer,
                bce,
                dice_loss,
                device,
                max_classes_per_image=args.max_classes_per_image,
            )
            train_loss_tensor = torch.tensor([train_loss], device=device)
            dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = (train_loss_tensor / world_size).item()

            val_loss_sum, val_dice_sum, val_iou_sum, val_steps = evaluate(
                sam,
                val_loader,
                bce,
                dice_loss,
                device,
                max_classes_per_image=args.max_classes_per_image,
            )
            metrics_tensor = torch.tensor(
                [val_loss_sum, val_dice_sum, val_iou_sum, float(val_steps)],
                device=device,
            )
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_steps = max(1.0, metrics_tensor[3].item())
            val_loss = metrics_tensor[0].item() / total_steps
            val_dice = metrics_tensor[1].item() / total_steps
            val_iou = metrics_tensor[2].item() / total_steps

            if rank == 0:
                print(
                    f"[DDP] Epoch [{epoch+1}/{args.num_epochs}] train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f}"
                )
                if val_dice > best_metric:
                    best_metric = val_dice
                    save_sam_checkpoint(sam, paths["best_ckpt"])
                    print(
                        f"  -> New best checkpoint saved to {paths['best_ckpt']} "
                        f"(val_dice={val_dice:.4f})"
                    )
    finally:
        dist.destroy_process_group()


def run_distributed_training(args, paths):
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices")
    world_size = max(1, args.world_size)
    if world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"world_size={world_size} exceeds available GPUs={torch.cuda.device_count()}"
        )
    mp.spawn(
        distributed_worker,
        args=(world_size, args, paths),
        nprocs=world_size,
        join=True,
    )


def main():
    args = parse_args()
    paths = resolve_paths(args)
    if args.distributed:
        run_distributed_training(args, paths)
    else:
        run_single_training(args, paths)


if __name__ == "__main__":
    main()
