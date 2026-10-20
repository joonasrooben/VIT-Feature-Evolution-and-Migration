# vit_tiny_imagenet_train_ffcv.py
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import timm
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.transforms import (
    ToTensor,
    ToDevice,
    ToTorchImage,
    Convert,
    NormalizeImage,
    RandomHorizontalFlip,
    RandomBrightness,
    RandomContrast,
    RandomSaturation,
    Squeeze,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    train_beton: str
    val_beton: str
    train_samples: int
    val_samples: int
    num_classes: int = 1000
    output_dir: str = "./results/vit_tiny_ffcv"

    epochs: int = 300
    warmup_epochs: int = 5
    batch_size: int = 1024
    eval_batch_size: int = 512
    grad_accum_steps: int = 1

    base_lr: float = 1e-3
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)

    num_workers: int = 8
    image_size: int = 224
    crop_pct: float = 224 / 256

    model_name: str = "vit_tiny_patch16_224"
    drop_path: float = 0.0
    layer_decay: float = 0.75

    mixup_alpha: float = 0.2

    # Approximation to ColorJitter(0.3); no hue term here
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    color_p: float = 0.8

    amp_dtype: str = "bf16"
    channels_last: bool = True
    compile_model: bool = False
    os_cache: bool = True
    save_every: int = 1
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
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
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
# FFCV data
# -----------------------------------------------------------------------------
def build_ffcv_loaders(cfg: TrainConfig, device: torch.device):
    device_for_loader = torch.device(f"cuda:{device.index}")

    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    std = np.array([0.229, 0.224, 0.225]) * 255.0

    train_image_pipeline = [
        RandomResizedCropRGBImageDecoder((cfg.image_size, cfg.image_size)),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomBrightness(cfg.brightness, p=cfg.color_p),
        RandomContrast(cfg.contrast, p=cfg.color_p),
        RandomSaturation(cfg.saturation, p=cfg.color_p),
        ToTensor(),
        ToDevice(device_for_loader, non_blocking=True),
        ToTorchImage(channels_last=cfg.channels_last),
        NormalizeImage(mean, std, np.float16),
    ]

    val_image_pipeline = [
        CenterCropRGBImageDecoder((cfg.image_size, cfg.image_size), ratio=cfg.crop_pct),
        ToTensor(),
        ToDevice(device_for_loader, non_blocking=True),
        ToTorchImage(channels_last=cfg.channels_last),
        NormalizeImage(mean, std, np.float16),
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        ToDevice(device_for_loader, non_blocking=True),
        Squeeze(),
    ]

    order = OrderOption.RANDOM if not is_dist() else OrderOption.RANDOM

    train_loader = Loader(
        cfg.train_beton,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        order=order,
        os_cache=cfg.os_cache,
        drop_last=True,
        pipelines={
            "image": train_image_pipeline,
            "label": label_pipeline,
        },
        distributed=is_dist(),
    )

    val_loader = Loader(
        cfg.val_beton,
        batch_size=cfg.eval_batch_size,
        num_workers=max(2, cfg.num_workers // 2),
        order=OrderOption.SEQUENTIAL,
        os_cache=cfg.os_cache,
        drop_last=False,
        pipelines={
            "image": val_image_pipeline,
            "label": label_pipeline,
        },
        distributed=is_dist(),
    )

    global_train_batch = cfg.batch_size * get_world_size()
    global_val_batch = cfg.eval_batch_size * get_world_size()
    train_steps = cfg.train_samples // global_train_batch
    val_steps = math.ceil(cfg.val_samples / global_val_batch)

    return train_loader, val_loader, train_steps, val_steps


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
        drop_path_rate=cfg.drop_path,
    )
    return model


# -----------------------------------------------------------------------------
# Scheduler
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
    train_loader,
    train_steps: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mixup_fn: Mixup,
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
    steps_per_epoch = train_steps
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    end = time.time()
    optimizer.zero_grad(set_to_none=True)

    for it, (images, targets) in enumerate(train_loader):
        if it >= steps_per_epoch:
            break

        data_time = time.time() - end
        global_step = epoch * steps_per_epoch + it

        lr_scale = cosine_warmup_lr(global_step, total_steps, warmup_steps)
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base_lr * lr_scale

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

        torch.cuda.synchronize()
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
def evaluate(
    model: nn.Module,
    val_loader,
    val_steps: int,
    cfg: TrainConfig,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16

    for it, (images, targets) in enumerate(val_loader):
        if it >= val_steps:
            break

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

    train_loader, val_loader, train_steps, val_steps = build_ffcv_loaders(cfg, device)
    num_classes = cfg.num_classes

    model = build_model(cfg, num_classes=num_classes).to(device)

    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.compile_model:
        model = torch.compile(model)

    if is_dist():
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

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

    criterion = SoftTargetCrossEntropy().cuda()

    scaler = None
    if cfg.amp_dtype == "fp16":
        scaler = torch.amp.GradScaler("cuda")

    total_steps = cfg.epochs * train_steps

    if is_main_process():
        print(f"model={cfg.model_name}")
        print(f"num_classes={num_classes}")
        print(f"effective_batch={effective_batch}")
        print(f"scaled_lr={scaled_lr:.8f}")
        print(f"train_steps={train_steps}")
        print(f"val_steps={val_steps}")
        print(f"output_dir={cfg.output_dir}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc1 = -1.0

    for epoch in range(cfg.epochs):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            train_steps=train_steps,
            optimizer=optimizer,
            criterion=criterion,
            mixup_fn=mixup_fn,
            epoch=epoch,
            cfg=cfg,
            scaler=scaler,
            total_steps=total_steps,
            base_lrs=base_lrs,
        )

        val_stats = evaluate(
            model=model,
            val_loader=val_loader,
            val_steps=val_steps,
            cfg=cfg,
        )

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

            if acc1 > best_acc1:
                best_acc1 = acc1
                save_checkpoint(output_dir / "best.ckpt", state)

            
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.ckpt", state)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="ViT-Tiny ImageNet-1K training with FFCV")
    parser.add_argument("--train-beton", type=str, required=True)
    parser.add_argument("--val-beton", type=str, required=True)
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--val-samples", type=int, required=True)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--output-dir", type=str, default="./results/vit_tiny_ffcv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--grad-accum-steps", type=int, default=1)

    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--layer-decay", type=float, default=0.75)

    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--brightness", type=float, default=0.3)
    parser.add_argument("--contrast", type=float, default=0.3)
    parser.add_argument("--saturation", type=float, default=0.3)
    parser.add_argument("--color-p", type=float, default=0.8)

    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-os-cache", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        train_beton=args.train_beton,
        val_beton=args.val_beton,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
        model_name=args.model_name,
        drop_path=args.drop_path,
        layer_decay=args.layer_decay,
        mixup_alpha=args.mixup_alpha,
        brightness=args.brightness,
        contrast=args.contrast,
        saturation=args.saturation,
        color_p=args.color_p,
        amp_dtype=args.amp_dtype,
        channels_last=args.channels_last,
        compile_model=args.compile_model,
        os_cache=not args.no_os_cache,
        save_every=args.save_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)