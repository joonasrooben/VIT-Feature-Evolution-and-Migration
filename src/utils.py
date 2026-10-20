import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Sequence, Literal
import math
import torch
from pathlib import Path

def denormalize(normalized, basis):
    return basis.min() + (basis.max() - basis.min())*((normalized + 1)/2)

def normalize(unnormalized, basis):
    return (2 * (unnormalized - basis.min()) / (basis.max() - basis.min()) - 1)

def normalize_feat_wise(unnormalized, basis):
    return (2 * (unnormalized - basis.min(0)) / (basis.max(0) - basis.min(0)) - 1)

def denormalize_feat_wise(normalized, basis):
    return basis.min(0) + (basis.max(0) - basis.min(0))*((normalized + 1)/2)

def denormalize_std(normalized, basis):
    return normalized * basis.std(0) + basis.mean(0)

def normalize_std(unnormalized, basis):
    return (unnormalized - basis.mean(0))/basis.std(0)

def feature_normalize(X, axis=1, eps=1e-12):
    """Normalize along axis (rows if axis=1) with numerical stability. From mahalanobis ++"""
    X = np.asarray(X)
    norms = np.linalg.norm(X, ord=2, axis=axis, keepdims=True)
    return X / np.clip(norms, eps, None)

def split_names_to_dict(name_list, append = False, labels_dict = None ):
    if ~append:
        labels_dict = {"fh":[],"wh":[],"oh":[],"sc":[],"sh":[] ,"or":[] }
    for name in name_list:
        spltted_name = name.split("_")
        labels_dict["fh"].append(int(spltted_name[3][2:]))
        labels_dict["wh"].append(int(spltted_name[4][2:]))
        labels_dict["oh"].append(int(spltted_name[5][2:]))
        labels_dict["sc"].append(int(spltted_name[6][2:]))
        labels_dict["sh"].append(int(spltted_name[7][2:]))
        labels_dict["or"].append(int(spltted_name[8][2:].split(".")[0]))

    return labels_dict

def load_all_data(epoch, step, seed, normalization_on,keywords, layer_nr = 11, backbone = "vit", ind_name = "shapes3d", train_patches =False, test_patches =False,eval_mode=False, eval_split="test", activation_root="../openood/OpenOOD" ): 
    if eval_mode:
        acts = np.load(f"{activation_root}/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{eval_split}.npy")[:,layer_nr,:]

        if normalization_on:
            acts = feature_normalize(acts)
        return acts

    all_data_dict = {}
    for key in keywords:
        patches_acts = None
        if (key == "train") and train_patches:
            patches_acts = np.load(f"{activation_root}/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_patches.npy")[:,layer_nr,:]
            print(patches_acts.shape)
        elif (key == "test") and test_patches:
            patches_acts = np.load(f"{activation_root}/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_patches.npy")[:,layer_nr,:]
            print(patches_acts.shape)


        acts = np.load(f"{activation_root}/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}.npy")[:,layer_nr,:]
        print(acts.shape)
        try:
            logits = np.load(f"{activation_root}/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_logits.npy")
        except:
            logits = None

        if normalization_on:
            acts = feature_normalize(acts)
            if patches_acts is not None:
                normed_patches_acts = feature_normalize(patches_acts.reshape((-1,acts.shape[1]))).reshape((-1 ,196, acts.shape[1]))
            else:
                normed_patches_acts = None
        if ind_name == "shapes3d":
            names = np.load(f"{activation_root}/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_names.npy").flatten() 
            labels = split_names_to_dict(names)
            all_data_dict[key] = {"activations" : acts, "labels": labels, "logits": logits, "names": names, "patches" : normed_patches_acts}
        else:
            labels = np.load(f"{activation_root}/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_y.npy").flatten() 
            all_data_dict[key] = {"activations" : acts, "labels": labels, "logits": logits, "names": None, "patches" : normed_patches_acts}
            
    return all_data_dict


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _device_or_default(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"



def _infer_patch_grid(n_tokens: int) -> Tuple[int, int]:
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError(
            f"Cannot infer square patch grid from {n_tokens} tokens. "
            "Pass patch_grid explicitly."
        )
    return side, side


def _tokens_to_patch_map(
    token_acts: np.ndarray,
    patch_grid: Optional[Tuple[int, int]] = None,
    drop_cls_token: bool = True,
) -> np.ndarray:
    """
    Accepts either:
      - shape [T] where T = H*W or 1+H*W
      - shape [H, W]

    Returns:
      - shape [H, W]
    """
    arr = np.asarray(token_acts, dtype=float)

    if arr.ndim == 2:
        return arr

    if arr.ndim != 1:
        raise ValueError(
            f"token_acts must have shape [T] or [H, W], got {arr.shape}"
        )

    t = arr.shape[0]

    if patch_grid is not None:
        gh, gw = patch_grid
        n_patch = gh * gw

        if t == n_patch:
            return arr.reshape(gh, gw)

        if drop_cls_token and t == n_patch + 1:
            return arr[1:].reshape(gh, gw)

        raise ValueError(
            f"Token count {t} incompatible with patch_grid={patch_grid}."
        )

    # infer automatically
    if drop_cls_token:
        side = int(round(math.sqrt(max(t - 1, 0))))
        if side * side == t - 1:
            return arr[1:].reshape(side, side)

    gh, gw = _infer_patch_grid(t)
    return arr.reshape(gh, gw)

def _reduce_token_scores(
    token_scores: np.ndarray,
    mode: Literal["max", "mean", "sum_positive", "q95"] = "max",
) -> float:
    ts = np.asarray(token_scores, dtype=float).reshape(-1)

    if ts.size == 0:
        return float("nan")

    if mode == "max":
        return float(np.max(ts))
    if mode == "mean":
        return float(np.mean(ts))
    if mode == "sum_positive":
        return float(np.clip(ts, 0.0, None).sum())
    if mode == "q95":
        return float(np.quantile(ts, 0.95))

    raise ValueError(f"Unknown reduction mode: {mode}")

def _normalize_patch_map_for_display(
    patch_map: np.ndarray,
    positive_only: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    pm = np.asarray(patch_map, dtype=float).copy()

    if positive_only:
        pm = np.clip(pm, 0.0, None)
        mx = pm.max()
        if mx > eps:
            pm /= mx
        return pm

    # signed normalization to [0, 1]
    lo, hi = pm.min(), pm.max()
    if hi - lo > eps:
        pm = (pm - lo) / (hi - lo)
    else:
        pm[:] = 0.0
    return pm
