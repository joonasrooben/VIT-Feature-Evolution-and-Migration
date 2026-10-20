from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from PIL import Image

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    train_dir: str | None
    val_dir: str | None
    data_root: str | None = None
    train_imglist: str | None = None
    val_imglist: str | None = None
    output_dir: str = "./runs/vit_tiny_in1k"
    epochs: int = 300
    warmup_epochs: int = 5
    batch_size: int = 512          # per GPU
    eval_batch_size: int = 512
    grad_accum_steps: int = 1
    base_lr: float = 1e-3          # paper-style base LR before scaling
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    num_workers: int = 8
    prefetch_factor: int = 4
    image_size: int = 224
    crop_pct: float = 224 / 256
    model_name: str = "vit_tiny_patch16_224"
    drop_path: float = 0.0
    mixup_alpha: float = 0.2
    color_jitter: float = 0.3
    randaug_num_ops: int = 10
    randaug_magnitude: int = 15    # approx mapping for paper's RandAug(10, 0.5)
    layer_decay: float = 0.75      # tunable; paper uses LLRD but does not pin one scratch value
    amp_dtype: str = "bf16"       # bf16 on A100/H100; fp16 on V100/T4
    channels_last: bool = True
    compile_model: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    save_every: int = 10
    seed: int = 42


# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed() -> tuple[torch.device, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        dist.barrier()
        return torch.device("cuda", local_rank), local_rank

    if not torch.cuda.is_available():
        raise RuntimeError("This script is intended for CUDA training.")

    return torch.device("cuda", 0), 0


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n


def reduce_tensor(t: torch.Tensor, average: bool = True) -> torch.Tensor:
    if not is_dist():
        return t
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
class OpenOODImglistDataset(Dataset):
    """
    Minimal ImageNet-style imglist dataset compatible with OpenOOD text files.
    Expected line formats:
      relative/path.jpg 123
      relative/path.jpg {"label": 123, ...}
    The path is resolved relative to cfg.data_root.
    """
    def __init__(self, imglist_path: str, data_root: str, transform=None):
        import ast
        self.transform = transform
        self.samples = []

        with open(imglist_path, "r") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                image_name, extra_str = raw_line.split(" ", 1)

                extra = ast.literal_eval(extra_str)
                if isinstance(extra, dict):
                    label = int(extra["label"])
                else:
                    label = int(extra)

                full_path = os.path.join(data_root, image_name)
                self.samples.append((full_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]

        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    


def build_transforms(cfg: TrainConfig):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.image_size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=cfg.color_jitter,
            contrast=cfg.color_jitter,
            saturation=cfg.color_jitter,
            hue=0.0,
        ),
        transforms.RandAugment(
            num_ops=cfg.randaug_num_ops,
            magnitude=cfg.randaug_magnitude,
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    resize_size = int(cfg.image_size / cfg.crop_pct)
    val_transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return train_transform, val_transform


def build_dataloaders(cfg: TrainConfig):
    train_tf, val_tf = build_transforms(cfg)

    use_imglists = cfg.train_imglist is not None and cfg.val_imglist is not None
    if use_imglists:
        if cfg.data_root is None:
            raise ValueError("--data-root is required when using --train-imglist/--val-imglist")
        train_set = OpenOODImglistDataset(cfg.train_imglist, cfg.data_root, transform=train_tf)
        val_set = OpenOODImglistDataset(cfg.val_imglist, cfg.data_root, transform=val_tf)
    else:
        if cfg.train_dir is None or cfg.val_dir is None:
            raise ValueError("Either provide --train-dir/--val-dir or provide --data-root with --train-imglist/--val-imglist")
        train_set = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
        val_set = datasets.ImageFolder(cfg.val_dir, transform=val_tf)

    train_sampler = DistributedSampler(train_set, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if is_dist() else None

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(2, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


# -----------------------------------------------------------------------------
# Model / optimizer
# -----------------------------------------------------------------------------
def get_num_layers_for_vit(model: nn.Module) -> int:
    if hasattr(model, "blocks"):
        return len(model.blocks) + 2
    raise ValueError("Expected a timm ViT with .blocks")


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
    if name in {"cls_token", "pos_embed"} or name.startswith("patch_embed"):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers - 1


def param_groups_layer_decay(model: nn.Module, lr: float, weight_decay: float, layer_decay: float):
    num_layers = get_num_layers_for_vit(model)
    groups: dict[str, dict] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        use_decay = param.ndim > 1 and not name.endswith(".bias")
        decay_tag = "decay" if use_decay else "no_decay"
        this_wd = weight_decay if use_decay else 0.0

        layer_id = get_layer_id_for_vit(name, num_layers)
        lr_scale = layer_decay ** (num_layers - 1 - layer_id)
        group_name = f"layer_{layer_id}_{decay_tag}"

        if group_name not in groups:
            groups[group_name] = {
                "params": [],
                "weight_decay": this_wd,
                "lr": lr * lr_scale,
            }
        groups[group_name]["params"].append(param)

    return list(groups.values())


def build_model(cfg: TrainConfig, num_classes: int) -> nn.Module:
    model = timm.create_model(
        cfg.model_name,
        pretrained=False,
        num_classes=num_classes,
        embed_dim=192,
        num_heads=12,
        depth=12,
        drop_path_rate = cfg.drop_path,
    )
    
    return model


# -----------------------------------------------------------------------------
# Schedulers
# -----------------------------------------------------------------------------
def cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 1e-5) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mixup_fn: Mixup,
    device: torch.device,
    epoch: int,
    cfg: TrainConfig,
    scaler: torch.amp.GradScaler | None,
    total_steps: int,
    base_lrs: list[float],
) -> dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    data_meter = AverageMeter()
    step_meter = AverageMeter()

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    world_size = get_world_size()
    steps_per_epoch = len(train_loader)
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    end = time.time()
    optimizer.zero_grad(set_to_none=True)

    for it, (images, targets) in enumerate(train_loader):
        data_time = time.time() - end
        global_step = epoch * steps_per_epoch + it

        lr_scale = cosine_warmup_lr(global_step, total_steps, warmup_steps)
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base_lr * lr_scale

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if cfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        images, targets = mixup_fn(images, targets)

        step_start = time.time()
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(images)

            loss = criterion(outputs, targets)
            loss = loss / cfg.grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
            if (it + 1) % cfg.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if (it + 1) % cfg.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_time = time.time() - step_start

        reduced_loss = reduce_tensor(loss.detach() * cfg.grad_accum_steps, average=True)
        batch_size = images.shape[0] * world_size
        loss_meter.update(reduced_loss.item(), batch_size)
        data_meter.update(data_time)
        step_meter.update(step_time)

        if is_main_process() and (it % 10 == 0 or it == steps_per_epoch - 1):
            imgs_per_sec = batch_size / max(step_time, 1e-8)
            print(
                f"epoch={epoch:03d} iter={it:04d}/{steps_per_epoch} "
                f"loss={loss_meter.avg:.4f} data_t={data_meter.avg:.4f}s "
                f"step_t={step_meter.avg:.4f}s imgs/s={imgs_per_sec:.1f}",
                flush=True,
            )

        end = time.time()

    return {
        "train_loss": loss_meter.avg,
        "train_data_time": data_meter.avg,
        "train_step_time": step_meter.avg,
    }


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device, cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss().to(device)
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if cfg.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        loss = reduce_tensor(loss, average=True)
        acc1 = reduce_tensor(acc1, average=True)
        acc5 = reduce_tensor(acc5, average=True)

        n = images.size(0) * get_world_size()
        loss_meter.update(loss.item(), n)
        acc1_meter.update(acc1.item(), n)
        acc5_meter.update(acc5.item(), n)

    return {
        "val_loss": loss_meter.avg,
        "val_acc1": acc1_meter.avg,
        "val_acc5": acc5_meter.avg,
    }


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(cfg: TrainConfig) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    device, local_rank = setup_distributed()
    seed_everything(cfg.seed + get_rank())

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train_loader, val_loader, train_sampler = build_dataloaders(cfg)
    if hasattr(train_loader.dataset, "classes"):
        num_classes = len(train_loader.dataset.classes)
        print(num_classes)
    else:
        num_classes = 1000

    model = build_model(cfg, num_classes=num_classes).to(device)
    print(model)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.compile_model:
        model = torch.compile(model)

    if is_dist():
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)

    model_for_optim = model.module if isinstance(model, DDP) else model

    effective_batch = cfg.batch_size * get_world_size() * cfg.grad_accum_steps
    scaled_lr = cfg.base_lr * (effective_batch / 256.0)

    param_groups = param_groups_layer_decay(
        model_for_optim,
        lr=scaled_lr,
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay,
    )

    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=cfg.betas,
            eps=1e-8,
            fused=True,
        )
    except TypeError:
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=cfg.betas,
            eps=1e-8,
        )

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    mixup_fn = Mixup(
        mixup_alpha=cfg.mixup_alpha,
        cutmix_alpha=0.0,
        prob=1.0 if cfg.mixup_alpha > 0 else 0.0,
        switch_prob=0.0,
        mode="batch",
        label_smoothing=0.0,
        num_classes=num_classes,
    )

    criterion = SoftTargetCrossEntropy().to(device)

    scaler = None
    if cfg.amp_dtype == "fp16":
        scaler = torch.amp.GradScaler("cuda")

    total_steps = cfg.epochs * len(train_loader)

    if is_main_process():
        print(f"model={cfg.model_name}")
        print(f"num_classes={num_classes}")
        print(f"effective_batch={effective_batch}")
        print(f"scaled_lr={scaled_lr:.8f}")
        print(f"output_dir={cfg.output_dir}")

    best_acc1 = -1.0
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            mixup_fn=mixup_fn,
            device=device,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler,
            total_steps=total_steps,
            base_lrs=base_lrs,
        )
        val_stats = evaluate(model, val_loader, device, cfg)

        if is_main_process():
            acc1 = val_stats["val_acc1"]
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_stats['train_loss']:.4f} "
                f"val_loss={val_stats['val_loss']:.4f} "
                f"val_acc1={val_stats['val_acc1']:.2f} "
                f"val_acc5={val_stats['val_acc5']:.2f}",
                flush=True,
            )

            state = {
                "epoch": epoch,
                "model": (model.module if isinstance(model, DDP) else model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": cfg.__dict__,
                "best_acc1": best_acc1,
            }

            save_checkpoint(output_dir / f"epoch_{epoch:03d}.ckpt", state)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Optimized ViT-Tiny ImageNet-1K training pipeline")
    parser.add_argument("--train-dir", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--train-imglist", type=str, default=None)
    parser.add_argument("--val-imglist", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="./results/vit_tiny_in_mixed")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument("--randaug-num-ops", type=int, default=10)
    parser.add_argument("--randaug-magnitude", type=int, default=15)
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        data_root=args.data_root,
        train_imglist=args.train_imglist,
        val_imglist=args.val_imglist,
        output_dir=args.output_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        image_size=args.image_size,
        model_name=args.model_name,
        drop_path=args.drop_path,
        mixup_alpha=args.mixup_alpha,
        color_jitter=args.color_jitter,
        randaug_num_ops=args.randaug_num_ops,
        randaug_magnitude=args.randaug_magnitude,
        layer_decay=args.layer_decay,
        amp_dtype=args.amp_dtype,
        channels_last=args.channels_last,
        compile_model=args.compile_model,
        pin_memory=not args.no_pin_memory,
        persistent_workers=not args.no_persistent_workers,
        save_every=args.save_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
