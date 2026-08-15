from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import yaml

# OpenOOD
from openood.datasets import get_dataloader
from openood.utils.config import Config

# You need timm for the MAE backbone:
# pip install timm
import timm


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _to_device(batch: Dict[str, Any], device: torch.device):
    x = batch["data"].float().to(device, non_blocking=True)
    y = batch["label"].long().to(device, non_blocking=True)
    return x, y


def _find_config_path(args: Dict[str, Any]) -> Path:
    """
    Try explicit config_path first; otherwise fall back to common OpenOOD result paths.
    """
    openood_root = Path(args.get("openood_root", "../openood/OpenOOD"))

    if args.get("config_path", None) is not None:
        return Path(args["config_path"])

    ind_name = args["ind_name"]
    seed = args.get("seed", 0)

    candidates = [
        openood_root / f"results/{ind_name}_vit-mae/s{seed}/config_gen.yml",
        openood_root / f"results/{ind_name}_vit-mae_base_e{args.get('total_epochs', 100)}_lr0.1_default/s{seed}/config_gen.yml",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Could not find config_gen.yml automatically. "
        "Pass args['config_path'] explicitly."
    )


def load_openood_config(args: Dict[str, Any]) -> Config:
    cfg_path = _find_config_path(args)
    with open(cfg_path, "r") as f:
        conf = Config(yaml.safe_load(f))
    return conf


# ---------------------------------------------------------
# MAE model loader
# ---------------------------------------------------------
def load_vit_base_mae(
    device: torch.device,
    mae_name: str = "vit_base_patch16_224.mae",
    pretrained: bool = True,
    ckpt_path: Optional[str] = None,
    num_classes: int = 0,
) -> nn.Module:
    """
    Load a ViT-Base MAE model via timm.

    Notes:
    - num_classes=0 gives a feature extractor without classifier head.
    - If you have a fine-tuned checkpoint with a classifier head, set num_classes accordingly
      and load ckpt_path.
    """
    model = timm.create_model(
        mae_name,
        pretrained=pretrained if ckpt_path is None else False,
        num_classes=num_classes,
    )

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Common checkpoint formats
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip possible prefixes
        cleaned = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[load_vit_base_mae] missing keys: {len(missing)}")
        print(f"[load_vit_base_mae] unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# Token extraction for timm ViT / MAE-style backbones
# ---------------------------------------------------------
@torch.no_grad()
def forward_tokens_per_block_timm_vit(
    model: nn.Module,
    x: torch.Tensor,
    return_logits: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor]]:
    """
    Forward a timm ViT-style model manually and collect hidden states after each block.

    Returns:
        cls_list:    list of [B, D]
        patch_list:  list of [B, Npatch, D]
        logits:      [B, C] or None
    """
    # Patch embedding
    x = model.patch_embed(x)  # [B, Npatch, D]

    # Add cls token + pos embedding
    if hasattr(model, "_pos_embed"):
        x = model._pos_embed(x)
    else:
        # Fallback for older timm variants
        B = x.shape[0]
        if hasattr(model, "cls_token") and model.cls_token is not None:
            cls_token = model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            x = x + model.pos_embed

    # Optional patch dropout / norm_pre
    if hasattr(model, "patch_drop") and model.patch_drop is not None:
        x = model.patch_drop(x)
    if hasattr(model, "norm_pre") and model.norm_pre is not None:
        x = model.norm_pre(x)

    cls_list = []
    patch_list = []

    # Transformer blocks
    for blk in model.blocks:
        x = blk(x)                      # [B, 1+Npatch, D]
        cls_list.append(x[:, 0].detach())
        patch_list.append(x[:, 1:].detach())

    # Final norm
    if hasattr(model, "norm") and model.norm is not None:
        x = model.norm(x)

    logits = None
    if return_logits:
        # Only meaningful if model has a classifier head
        try:
            if hasattr(model, "forward_head"):
                logits = model.forward_head(x, pre_logits=False).detach()
            elif hasattr(model, "head"):
                pooled = x[:, 0]
                logits = model.head(pooled).detach()
        except Exception as e:
            print(f"[forward_tokens_per_block_timm_vit] Could not compute logits: {e}")
            logits = None

    return cls_list, patch_list, logits


def cls_hidden_states_mae(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    cls_list, _, _ = forward_tokens_per_block_timm_vit(model, x, return_logits=False)
    return cls_list


def patches_hidden_states_mae(model: nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    _, patch_list, _ = forward_tokens_per_block_timm_vit(model, x, return_logits=False)
    return patch_list


# ---------------------------------------------------------
# Data pass
# ---------------------------------------------------------
@torch.no_grad()
def get_model_outputs_mae(
    data_loader,
    model: nn.Module,
    return_name: bool = False,
    patches_on: bool = False,
    train: bool = False,
    return_logits: bool = False,
) -> Dict[str, Any]:
    device = next(model.parameters()).device

    cls_all = []
    logits_all = []
    classes = []
    names = []
    patches_all = []

    model.eval()

    for batch in data_loader:
        if return_name:
            names.extend(batch["image_name"])

        x, y = _to_device(batch, device)
        classes.append(y)

        cls_batch, patch_batch, logits = forward_tokens_per_block_timm_vit(
            model, x, return_logits=return_logits
        )

        # CLS: list[L] of [B, D] -> [B, L, D]
        cls_layers = torch.stack(cls_batch, dim=1)
        cls_all.append(cls_layers)

        if logits is not None:
            logits_all.append(logits)

        if patches_on:
            if train:
                # Sample only 5% of current batch for patch saving to reduce storage
                perm = torch.randperm(x.size(0), device=x.device)
                k = max(1, int(x.size(0) * 0.05))
                idx = perm[:k]
                x_patch = x[idx]
            else:
                x_patch = x

            patch_batch2 = patches_hidden_states_mae(model, x_patch)
            # list[L] of [B, Npatch, D] -> [B, L, Npatch, D]
            patch_layers = torch.stack(patch_batch2, dim=1)
            patches_all.append(patch_layers)

    classes_tensor = torch.cat(classes, dim=0)
    cls_tensor = torch.cat(cls_all, dim=0)

    output = {
        "layer_x": cls_tensor,      # [N, L, D]
        "classes": classes_tensor,  # [N]
    }

    if len(logits_all) > 0:
        output["logit"] = torch.cat(logits_all, dim=0)

    if patches_on and len(patches_all) > 0:
        output["patches"] = torch.cat(patches_all, dim=0)  # [N, L, Npatch, D]

    if return_name:
        output["names"] = names

    return output


# ---------------------------------------------------------
# Main generation function
# ---------------------------------------------------------
def generate_acts_vit_mae(args: Dict[str, Any]):
    """
    Example args:
    args = {
        "openood_root": "../openood/OpenOOD",
        "config_path": None,  # optional explicit path to config_gen.yml
        "ind_name": "imagenet200",
        "seed": 0,

        # MAE settings
        "mae_name": "vit_base_patch16_224.mae",
        "mae_pretrained": True,
        "mae_ckpt_path": None,      # optional local checkpoint
        "mae_num_classes": 0,       # 0 = feature extractor, >0 if fine-tuned classifier
        "save_logits": False,

        # saving / extraction
        "epoch": 0,
        "step": 0,
        "train_patches": False,
        "test_patches": True,
        "out_dir": None,
    }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    openood_root = Path(args.get("openood_root", "../openood/OpenOOD"))
    ind_name = args["ind_name"]

    # 1) OpenOOD config + dataloaders
    conf = load_openood_config(args)
    loader_dict = get_dataloader(conf)
    ind_dict = list(loader_dict.keys())

    # 2) Load MAE
    net = load_vit_base_mae(
        device=device,
        mae_name=args.get("mae_name", "vit_base_patch16_224.mae"),
        pretrained=args.get("mae_pretrained", True),
        ckpt_path=args.get("mae_ckpt_path", None),
        num_classes=args.get("mae_num_classes", 0),
    )

    # 3) Output path
    if args.get("out_dir", None) is None:
        path = openood_root / f"vit_mae_{ind_name}_s{args.get('seed', 0)}" / f"epoch{args.get('epoch', 0)}_step{args.get('step', 0)}"
    else:
        path = Path(args["out_dir"])

    path.mkdir(parents=True, exist_ok=True)
    print(f"[generate_acts_vit_mae] Saving to: {path}")

    # Some OpenOOD datasets store names
    return_name = ind_name in ["shapes3d", "imagenet_mixed10_balanced"]

    for key in ind_dict:
        print(f"[split] {key}")

        if args.get("train_patches", False) and key == "train":
            buf_data = get_model_outputs_mae(
                loader_dict[key],
                net,
                return_name=return_name,
                patches_on=True,
                train=True,
                return_logits=args.get("save_logits", False),
            )
            np.save(path / f"vit_mae_{ind_name}_{key}_patches.npy",
                    buf_data["patches"].cpu().numpy())

        elif args.get("test_patches", False) and key == "test":
            buf_data = get_model_outputs_mae(
                loader_dict[key],
                net,
                return_name=return_name,
                patches_on=True,
                train=False,
                return_logits=args.get("save_logits", False),
            )
            np.save(path / f"vit_mae_{ind_name}_{key}_patches.npy",
                    buf_data["patches"].cpu().numpy())

        else:
            buf_data = get_model_outputs_mae(
                loader_dict[key],
                net,
                return_name=return_name,
                patches_on=False,
                train=False,
                return_logits=args.get("save_logits", False),
            )

        np.save(path / f"vit_mae_{ind_name}_{key}.npy",
                buf_data["layer_x"].cpu().numpy())
        np.save(path / f"vit_mae_{ind_name}_{key}_y.npy",
                buf_data["classes"].cpu().numpy())

        if "logit" in buf_data:
            np.save(path / f"vit_mae_{ind_name}_{key}_logits.npy",
                    buf_data["logit"].cpu().numpy())

        if return_name:
            np.save(path / f"vit_mae_{ind_name}_{key}_names.npy",
                    np.array(buf_data["names"]))
