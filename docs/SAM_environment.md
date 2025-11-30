# SAM 环境配置指南

本指南说明如何在全新机器上搭建用于 Segment Anything (SAM) 微调与推理的 `SAM` Conda 环境，包括系统依赖、Python 包和常用命令。

## 1. 系统与硬件要求

- **操作系统**：建议使用 Linux (Ubuntu 20.04/22.04) 或 WSL2。Windows/macOS 也可，但命令需自适应。
- **GPU**：NVIDIA GPU，驱动版本需兼容目标 CUDA 版本（如 CUDA 11.8+）。
- **CUDA / cuDNN**：建议安装官方 CUDA Toolkit；若使用 `pip install torch==...+cu118`，可依赖 PyTorch 自带的 CUDA runtime。
- **磁盘空间**：数据集与 checkpoint 可能超过 20GB，请确保 `dataset/` 与 `checkpoints/` 有足够空间。

## 2. 创建 Conda 环境

```bash
# 安装 Miniconda 或 Anaconda 并初始化 shell 后执行：
conda create -n SAM python=3.10 -y
conda activate SAM
```

如需在脚本内自动激活，可设置 `CONDA_ENV_NAME=SAM` 环境变量（`scripts/start_tmux_training.sh` 已支持）。

## 3. Python 依赖

在 `SAM` 环境中执行：

```bash
# 基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy tqdm pillow matplotlib

# 项目本身
pip install -e .
```

> 可按需补充 `onnxruntime`, `onnx`, `pycocotools` 等。若使用不同 CUDA 版本，请替换 PyTorch 安装命令。

## 4. 项目目录约定

- `dataset/`：存放 BDD100K 影像与标签（默认结构：`10k/train`, `10k/val`, `labels/train`, `labels/val`）。该目录已在 `.gitignore` 中，请勿提交。
- `checkpoints/`：存放预训练权重 (`sam_vit_h_4b8939.pth`) 与微调产物 (`sam_decoder_bdd_best.pth`)。
- `logs/`：`scripts/start_tmux_training.sh` 自动写入训练日志。

示例目录：
```
segment-anything-main/
├── checkpoints/
│   ├── sam_vit_h_4b8939.pth
│   └── sam_decoder_bdd_best.pth
├── dataset/
│   ├── 10k/train
│   ├── 10k/val
│   ├── labels/train
│   └── labels/val
└── logs/
```

## 5. 训练与推理

### 单卡训练
```bash
python train_sam_decoder_bdd100k.py \
  --num-epochs 10 \
  --debug-train-samples 0 \
  --debug-val-samples 0 \
  --max-classes-per-image 4
```

### 多卡训练（DDP）
```bash
TORCH_DISTRIBUTED_DEBUG=INFO \
python train_sam_decoder_bdd100k.py \
  --distributed \
  --world-size 4 \
  --num-epochs 20 \
  --debug-train-samples 0 \
  --debug-val-samples 0 \
  --max-classes-per-image 6
```

> 确保 `world-size` 不超过可用 GPU 数，必要时改用 `torchrun --nproc_per_node=4 ...`。

### tmux 一键启动
```bash
./scripts/start_tmux_training.sh
# Window0: 训练 + 日志; Window1: watch nvidia-smi
```

### 推理/可视化
```bash
python run_sam_mask.py \
  --checkpoint checkpoints/sam_decoder_bdd_best.pth \
  --input-dir dataset/10k/val \
  --output-dir output_masks
```

## 6. 常见问题

| 问题 | 解决方案 |
| --- | --- |
| DDP 卡死或警告 `destroy_process_group` | 确保所有 rank 路径一致；退出前调用 `dist.destroy_process_group()`；避免强行 `kill` 进程。 |
| `find_unused_parameters` 警告 | 若确认每次前向都会用到全部参数，可将 DDP 构造里的 `find_unused_parameters=False`；否则可忽略。 |
| 无法读取 dataset | 检查 `BDD_ROOT` 路径、`labels/*_train_id.png` 是否存在；可通过 `--debug-train-samples` 快速验证。 |
| tmux 训练窗口无进度条 | 只在 rank0 输出；确保 attach 到 rank0 日志窗口或在命令前导出 `RANK=0`。 |

## 7. 版本管理提示

- `dataset/` 已加入 `.gitignore`，若历史提交中仍包含数据，可执行 `git rm -r --cached dataset` 并重新提交。
- 推荐在 `logs/` 中保留训练命令与环境信息，便于复现。

依据上述步骤搭建环境后，即可运行训练脚本、分布式微调与推理脚本。若在安装或运行中遇到问题，可将错误日志与环境信息一并提供以便排查。
