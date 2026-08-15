import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
import timm

from openood.datasets import get_dataloader
from openood.utils.config import Config

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from PIL import Image
from tqdm import tqdm


sys.path.append("..")
sys.path.append("../openood/OpenOOD/")

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
                self.samples.append((full_path, label, image_name))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label, name = self.samples[index]

        with Image.open(path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, name
    


def build_transforms(image_size):
    
    resize_size = int(256)
    transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    return transform


def build_dataloaders(cfg):
    tf = build_transforms(cfg["img_size"])
    #data_root = "/gpfs/space/projects/mlgroup/data/images_largescale/"
    data_root = "/tmp/imagenet_1k/"
    return_dict = {}
    train_set = OpenOODImglistDataset(cfg["train_imglist"], data_root, transform=tf)
    test_set = OpenOODImglistDataset(cfg["test_imglist"], data_root, transform=tf)

    batch_size = 128

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
    )
    try:
        val_set = OpenOODImglistDataset(cfg["val_imglist"], data_root, transform=tf)
        val_loader = DataLoader(
            val_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
    )
        return_dict['val'] = val_loader
    except:
        print("no val split")

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        sampler=None,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
    )

    return_dict['train'] = train_loader
    return_dict['test'] = test_loader

    return return_dict




def load_timm_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "model" not in ckpt:
        raise KeyError(f"No 'model' key found in checkpoint. Top-level keys: {list(ckpt.keys())}")

    state_dict = ckpt["model"]

    # optional cleanup in case training used DDP / wrappers
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Missing keys: {missing[:20]}")
    print(f"Unexpected keys: {unexpected[:20]}")

def collect_hidden_states_timm(model, x, need_patches=False):
    """
    For timm VisionTransformer models.

    Returns
    -------
    cls_layers   : [B, L, D]
    patch_layers : [B, L, Npatch, D] or None
    logits       : [B, C]
    """
    model.eval()
    cls_list = []
    patch_list = []
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            # out: [B, 1 + Npatch, D]
            cls_list.append(out[:, 0].detach())
            if need_patches:
                patch_list.append(out[:, 1:].detach())
        return hook

    # timm ViT uses model.blocks
    for blk in model.blocks:
        hooks.append(blk.register_forward_hook(make_hook()))

    with torch.no_grad():
        logits = model(x)

    for h in hooks:
        h.remove()

    cls_layers = torch.stack(cls_list, dim=1)  # [B, L, D]
    patch_layers = torch.stack(patch_list, dim=1) if need_patches else None

    return cls_layers, patch_layers, logits


def get_model_outputs(
    data_loader,
    model,
    return_name=False,
    patches_on=False,
    train=False,
):
    device = next(model.parameters()).device

    cls_all = []
    logits_all = []
    classes_all = []
    names_all = []
    patches_all = []

    model.eval()
    with torch.no_grad():
        for x, y, names in tqdm(data_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if return_name:
                names_all.extend(names)

            classes_all.append(y)

            cls_layers, _, logits = collect_hidden_states_timm(
                model, x, need_patches=False
            )
            cls_all.append(cls_layers)
            logits_all.append(logits)

            if patches_on:
                if train:
                    perm = torch.randperm(x.size(0), device=x.device)
                    k = max(1, int(x.size(0) * 0.05))
                    idx = perm[:k]
                    x_patch = x[idx]
                else:
                    x_patch = x

                _, patch_layers, _ = collect_hidden_states_timm(
                    model, x_patch, need_patches=True
                )
                patches_all.append(patch_layers)

    out = {
        "logit": torch.cat(logits_all, dim=0),
        "layer_x": torch.cat(cls_all, dim=0),
        "classes": torch.cat(classes_all, dim=0),
    }

    if patches_on and len(patches_all) > 0:
        out["patches"] = torch.cat(patches_all, dim=0)

    if return_name:
        out["names"] = names_all

    return out


def generate_acts_vit(args):
    openood_path = Path("../openood/OpenOOD")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ind_name = args["ind_name"]
    num_classes = args["num_classes"]

    #ckpt_path = openood_path / f"results/vit_tiny_a1_mixed/epoch_{args['epoch']:03d}.ckpt"
    ckpt_path = openood_path / f"results/vit_tiny_ffcv/epoch_{args['epoch']:03d}.ckpt"


    # Rebuild the SAME architecture that produced the checkpoint.
    net = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        embed_dim=192,
        num_heads=12,
        depth=12,
        drop_path_rate=0.0,          # for loading/inference this usually does not matter
    )

    load_timm_checkpoint(net, ckpt_path)
    net = net.to(device)
    net.eval()

    #with open(openood_path / "results/vit_tiny_a1_mixed/config_gen.yml", "r") as stream:
    #    conf = Config(yaml.safe_load(stream))

    loader_dict = build_dataloaders(args)
    ind_dict = list(loader_dict.keys())

    save_path = openood_path / f"vit_{ind_name}_s{args['seed']}/epoch{args['epoch']}_step{args['step']}"
    save_path.mkdir(parents=True, exist_ok=True)

    return_name = ind_name in ["shapes3d", "imagenet_mixed10_balanced"]

    for key in ind_dict:
        print(f"Processing {key}")

        if args["train_patches"] and key == "train":
            buf_data = get_model_outputs(
                loader_dict[key],
                net,
                return_name=return_name,
                patches_on=True,
                train=True,
            )
            np.save(save_path / f"vit_{ind_name}_{key}_patches.npy",
                    buf_data["patches"].cpu().numpy())

        elif args["test_patches"] and key == "test":
            buf_data = get_model_outputs(
                loader_dict[key],
                net,
                return_name=return_name,
                patches_on=True,
                train=False,
            )
            np.save(save_path / f"vit_{ind_name}_{key}_patches.npy",
                    buf_data["patches"].cpu().numpy())

        else:
            buf_data = get_model_outputs(
                loader_dict[key],
                net,
                return_name=return_name,
                patches_on=False,
                train=False,
            )

        np.save(save_path / f"vit_{ind_name}_{key}.npy",
                buf_data["layer_x"].cpu().numpy())
        np.save(save_path / f"vit_{ind_name}_{key}_y.npy",
                buf_data["classes"].cpu().numpy())
        np.save(save_path / f"vit_{ind_name}_{key}_logits.npy",
                buf_data["logit"].cpu().numpy())

        if return_name:
            np.save(save_path / f"vit_{ind_name}_{key}_names.npy",
                    np.array(buf_data["names"], dtype=object))