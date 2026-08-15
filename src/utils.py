import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Sequence, Literal
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import math
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

def best_row_matches(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
    row_names_b: Optional[Sequence[str]] = None,
    exclude_cols: int = 2,
    metric: str = "cosine",
    compute_null: bool = False,
    null_samples: int = 100,
    null_percentile: float = 95.0,
    random_state: Optional[int] = None,
    return_threshold: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    For each row in heatmap_a, find the most similar row in heatmap_b.

    Args:
        heatmap_a: Matrix from make_heatmap (rows are concepts, cols are labels).
        heatmap_b: Matrix from make_heatmap (rows are concepts, cols are labels).
        row_names_b: Optional row names for heatmap_b (e.g., row_names from make_heatmap).
        exclude_cols: Number of leading columns to skip (default skips N and Acc).
        metric: "cosine" or "l2" (cosine similarity or negative L2 distance).
        compute_null: If True, estimate a null similarity distribution by random matches.
        null_samples: Number of random matches per row in heatmap_a for the null.
        null_percentile: Percentile of null distribution used as a threshold.
        random_state: Optional RNG seed for reproducibility.
        return_threshold: If True, include the null threshold in the return.

    Returns:
        best_scores: Similarity score for the best match per row in heatmap_a.
        best_row_names: Best-matching concept identifiers from heatmap_b.
    """
    a = np.asarray(heatmap_a)
    b = np.asarray(heatmap_b)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("heatmap_a and heatmap_b must be 2D arrays.")
    if a.shape[1] != b.shape[1]:
        raise ValueError("heatmap_a and heatmap_b must have the same number of columns.")

    if exclude_cols > 0:
        a = a[:, exclude_cols:]
        b = b[:, exclude_cols:]

    if metric == "cosine":
        a_norm = feature_normalize(a, axis=1)
        b_norm = feature_normalize(b, axis=1)
        sims = a_norm @ b_norm.T
    elif metric == "l2":
        a2 = (a**2).sum(axis=1, keepdims=True)
        b2 = (b**2).sum(axis=1, keepdims=True).T
        sims = -(a2 - 2 * (a @ b.T) + b2)
    else:
        raise ValueError("metric must be 'cosine' or 'l2'.")

    best_idx = sims.argmax(axis=1)
    best_scores = sims[np.arange(sims.shape[0]), best_idx]
    if row_names_b is None:
        best_row_names = [str(i) for i in best_idx.tolist()]
    else:
        best_row_names = [row_names_b[i] for i in best_idx.tolist()]

    if compute_null:
        rng = np.random.default_rng(random_state)
        n_a = a.shape[0]
        n_b = b.shape[0]
        if n_b == 0:
            raise ValueError("heatmap_b must have at least one row to compute null.")
        if null_samples <= 0:
            raise ValueError("null_samples must be > 0 when compute_null is True.")
        rand_rows = rng.integers(0, n_b, size=(n_a, null_samples))
        null_scores = sims[np.arange(n_a)[:, None], rand_rows].reshape(-1)
        threshold = np.percentile(null_scores, null_percentile)
    else:
        threshold = None

    if return_threshold:
        return best_scores, best_row_names, threshold
    return best_scores, best_row_names


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

def load_all_data(epoch, step, seed, normalization_on,keywords, layer_nr = 11, backbone = "vit", ind_name = "shapes3d", train_patches =False, test_patches =False,eval_mode=False, eval_split="test" ): 
    #'fh', 'wh', 'oh', 'sc', 'sh', 'or'
    if eval_mode:
        acts = np.load(f"../openood/OpenOOD/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{eval_split}.npy")[:,layer_nr,:]

        if normalization_on:
            acts = feature_normalize(acts)
        return acts

    all_data_dict = {}
    for key in keywords:
        patches_acts = None
        if (key == "train") and train_patches:
            patches_acts = np.load(f"../openood/OpenOOD/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_patches.npy")[:,layer_nr,:]
            print(patches_acts.shape)
        elif (key == "test") and test_patches:
            patches_acts = np.load(f"../openood/OpenOOD/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_patches.npy")[:,layer_nr,:]
            print(patches_acts.shape)


        acts = np.load(f"../openood/OpenOOD/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}.npy")[:,layer_nr,:]
        print(acts.shape)
        try:
            logits = np.load(f"../openood/OpenOOD/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_logits.npy")
        except:
            logits = None

        if normalization_on:
            acts = feature_normalize(acts)
            if patches_acts is not None:
                normed_patches_acts = feature_normalize(patches_acts.reshape((-1,acts.shape[1]))).reshape((-1 ,196, acts.shape[1]))
            else:
                normed_patches_acts = None
        if ind_name == "shapes3d":
            names = np.load(f"../openood/OpenOOD/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_names.npy").flatten() 
            labels = split_names_to_dict(names)
            all_data_dict[key] = {"activations" : acts, "labels": labels, "logits": logits, "names": names, "patches" : normed_patches_acts}
        else:
            labels = np.load(f"../openood/OpenOOD/{backbone}_{ind_name}{seed}/epoch{epoch}_step{step}/{backbone}_{ind_name}_{key}_y.npy").flatten() 
            all_data_dict[key] = {"activations" : acts, "labels": labels, "logits": logits, "names": None, "patches" : normed_patches_acts}
            
    return all_data_dict

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

def plot_top_activating_images_per_class(
    num_classes: int,
    activations: Sequence[float],
    save_path: Path,
    *,
    # Option A: provide labels + image paths directly
    labels: Optional[Sequence[int]] = None,
    img_paths: Optional[Sequence[str]] = None,
    # Option B: provide dataset roots like before
    data_dir: Optional[Path] = None,
    imglist_pth: Optional[Path] = None,
    # plotting options
    top_k: int = 10,
    preview_max_side: int = 96,
    title: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    sort_descending: bool = True,
    show_only_positive_summary: bool = True,
) -> None:
    """
    Creates a plot with:
      1) top row: normalized activation mass per class (bar plot)
      2) below: one row per class, up to top_k images per class, ordered by activation

    Assumptions:
      - `activations[i]` corresponds to image i and label i
      - sorting is descending by default (highest activations first)
      - top summary uses positive activation mass if possible; otherwise abs activation mass

    You can either pass:
      - labels + img_paths
    or
      - data_dir + imglist_pth
    """
    activations = np.asarray(activations, dtype=float)

    # --- load image paths / labels ---
    if labels is not None and img_paths is not None:
        labels_arr = np.asarray(labels, dtype=int)
        resolved_paths = [Path(p) for p in img_paths]
    elif data_dir is not None and imglist_pth is not None:
        rel_paths, lbls = _read_imglist(imglist_pth)
        labels_arr = np.asarray(lbls, dtype=int)
        resolved_paths = [_resolve_image_path(data_dir, rp) for rp in rel_paths]
    else:
        raise ValueError(
            "Provide either (labels and img_paths) or (data_dir and imglist_pth)."
        )

    if len(activations) != len(labels_arr) or len(activations) != len(resolved_paths):
        raise ValueError(
            f"Length mismatch: len(activations)={len(activations)}, "
            f"len(labels)={len(labels_arr)}, len(img_paths)={len(resolved_paths)}"
        )

    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(
            f"len(class_names)={len(class_names)} must match num_classes={num_classes}"
        )

    valid = np.isfinite(activations)
    activations = activations[valid]
    labels_arr = labels_arr[valid]
    resolved_paths = [p for p, v in zip(resolved_paths, valid) if v]

    # --- per-class ordering ---
    class_to_sorted_indices: List[np.ndarray] = []
    for c in range(num_classes):
        idx = np.where(labels_arr == c)[0]

        # keep only positive activations
        idx = idx[activations[idx] > 0]

        if idx.size == 0:
            class_to_sorted_indices.append(idx)
            continue

        order = np.argsort(activations[idx])[::-1]
        class_to_sorted_indices.append(idx[order])

    # --- top summary: normalized activation mass by class ---
    if show_only_positive_summary:
        weights = np.clip(activations, 0.0, None)
        if weights.sum() == 0:
            weights = np.abs(activations)
    else:
        weights = np.abs(activations)

    class_mass = np.bincount(labels_arr, weights=weights, minlength=num_classes).astype(float)
    if class_mass.sum() > 0:
        class_mass /= class_mass.sum()

    # --- figure layout ---
    fig_w = max(14.0, 1.45 * top_k)
    fig_h = max(2.8 + 1.45 * num_classes, 6.0)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=num_classes + 1,
        ncols=top_k,
        height_ratios=[1.3] + [1.0] * num_classes,
        hspace=0.04,
        wspace=0.02,
    )

    # --- top bar plot ---
    ax_top = fig.add_subplot(gs[0, :])
    x = np.arange(num_classes)
    ax_top.bar(x, class_mass)
    ax_top.set_xlim(-0.5, num_classes - 0.5)
    ax_top.set_ylabel("Norm. mass")

    if class_names is None:
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([str(i) for i in range(num_classes)])
    else:
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(class_names, rotation=45, ha="right")

    if title is None:
        ax_top.set_title("Normalized activation mass by class")
    else:
        ax_top.set_title(title)

    ax_top.grid(True, axis="y", alpha=0.25)

    # --- image grid ---
    for c in range(num_classes):
        sorted_idx = class_to_sorted_indices[c]
        n_show = min(top_k, len(sorted_idx))

        for j in range(top_k):
            ax = fig.add_subplot(gs[c + 1, j])
            ax.set_xticks([])
            ax.set_yticks([])

            if j >= n_show:
                ax.set_axis_off()
                continue

            i = sorted_idx[j]
            img_path = resolved_paths[i]
            act = float(activations[i])

            try:
                im = _load_preview_image(img_path, max_side=preview_max_side)
                ax.imshow(im, interpolation="nearest")
            except Exception:
                ax.set_axis_off()
                continue

            # activation overlay
            ax.text(
                0.02,
                0.04,
                f"{act:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.65, pad=1.2, edgecolor="none"),
            )

            # class label on first image in row
            if j == 0:
                row_label = class_names[c] if class_names is not None else f"class {c}"
                ax.set_ylabel(row_label, rotation=0, labelpad=28, va="center", fontsize=9)

    _save_figure_light(fig, save_path, dpi=150)
    plt.close(fig)

