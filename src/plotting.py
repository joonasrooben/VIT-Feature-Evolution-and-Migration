from pathlib import Path
import math
from typing import List, Tuple, Optional, Dict, Any, Sequence, Literal, Callable
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
import seaborn as sns
from PIL import Image


from src.utils import _ensure_dir, _tokens_to_patch_map, _tokens_to_patch_map, _infer_patch_grid, _reduce_token_scores, _normalize_patch_map_for_display

def _second_highest_value(matrix: np.ndarray) -> float:
    """Second-highest *distinct* finite value (fallback to max if not available)."""
    vals = np.asarray(matrix).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    uniq = np.unique(vals)
    uniq.sort()
    return float(uniq[-2] if uniq.size >= 2 else uniq[-1])

def _read_imglist(imglist_pth: Path) -> Tuple[List[str], List[int]]:
    """
    Reads ImglistDataset txt format: each line like '<rel_path> <label>'
    (path may occasionally have spaces; we treat last token as label).
    """
    rel_paths: List[str] = []
    labels: List[int] = []
    with open(imglist_pth, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            label = int(parts[-1])
            rel_path = " ".join(parts[:-1])
            rel_paths.append(rel_path)
            labels.append(label)
    return rel_paths, labels


def _resolve_image_path(data_dir: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (data_dir / p)

def _load_preview_image(
    img_path: Path,
    max_side: int = 96,
) -> Image.Image:
    """
    Load image and downsample it for fast plotting.
    max_side=96 or 128 is usually enough for subplot previews.
    """
    im = Image.open(img_path).convert("RGB")
    im.thumbnail((max_side, max_side), getattr(Image, "Resampling", Image).BILINEAR)
    return im


def _plot_heatmap(
    matrix: np.ndarray,
    epochs: List[int],
    layers: List[int],
    title: str,
    save_path: Path,
    *,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 1.0,
    cmap: str = "viridis",
) -> None:
    fig_w = max(6.0, 0.7 * len(epochs))
    fig_h = max(4.0, 0.5 * len(layers))
    plt.figure(figsize=(fig_w, fig_h))
    vmax = _second_highest_value(matrix)
    sns.heatmap(
        matrix,
        xticklabels=[str(e) for e in epochs],
        yticklabels=[str(l) for l in layers],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar=True,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    _ensure_dir(save_path.parent)
    plt.savefig(save_path, dpi=300)
    plt.close()

def _plot_two_heatmaps_with_acc_and_images(
    matrix_top: np.ndarray,
    matrix_mid: np.ndarray,
    matrix_bottom: np.ndarray,
    accs: Sequence[float],
    epochs: List[int],
    layers: List[int],
    titles: Sequence[str],   # expected: [suptitle, top, mid, bottom]
    save_path: Path,
    *,
    cmap: str = "viridis",
    img_indices: Optional[Sequence[int]] = None,
    data_dir: Optional[Path] = None,
    imglist_pth: Optional[Path] = None,
    show_image_labels: bool = False,
    acc_ylabel: str = "Accuracy",
    acc_ylim = None,
    acts_values = None,
    heatmap_limits=None,
) -> None:
    want_images = (
        img_indices is not None
        and len(img_indices) > 0
        and data_dir is not None
        and imglist_pth is not None
    )

    n_epochs = len(epochs)
    n_layers = len(layers)
    if type(matrix_mid) == type(None):
        matrices = [matrix_top, matrix_bottom]
    else:
        matrices = [matrix_top, matrix_mid, matrix_bottom]

    # --- saner figure geometry ---
    fig_w = min(22.0, max(10.0, 0.18 * n_epochs))
    heat_h = min(5.5, max(2.5, 0.28 * n_layers))
    acc_h = 1.6
    images_h = 2.2 if want_images else 0.0
    label_h = 0.5 if want_images else 0.0
    fig_h = acc_h + len(matrices) * heat_h + images_h + label_h

    nrows = (len(matrices) + 3) if want_images else (len(matrices) + 1)
    height_ratios = [acc_h] + [heat_h]*len(matrices) + ([images_h, label_h] if want_images else [])

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.suptitle(titles[0], fontsize=18)

    gs = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=height_ratios)

    # --- accuracy axis ---
    ax_acc = fig.add_subplot(gs[0, 0])
    x = np.arange(n_epochs)
    ax_acc.plot(x, np.asarray(accs, dtype=float), linewidth=1.5)
    ax_acc.set_xlim(-0.5, n_epochs - 0.5)
    ax_acc.set_ylabel(acc_ylabel)
    if acc_ylim is not None:
        ax_acc.set_ylim(*acc_ylim)
    ax_acc.grid(True, axis="y", alpha=0.3)
    ax_acc.tick_params(axis="x", labelbottom=False)

    # --- heatmap axes ---
    ax_top = fig.add_subplot(gs[1, 0])
    if type(matrix_mid) == type(None):
        ax_bottom = fig.add_subplot(gs[2, 0], sharex=ax_top, sharey=ax_top)
        heat_axes = [ax_top, ax_bottom]
    else:
        ax_mid = fig.add_subplot(gs[2, 0], sharex=ax_top, sharey=ax_top)
        ax_bottom = fig.add_subplot(gs[3, 0], sharex=ax_top, sharey=ax_top)
        heat_axes = [ax_top, ax_mid, ax_bottom]

    # use one common color range
    global_vmin = 0 #min(min(m.min(), 0.0) for m in matrices)
    global_vmax = 1 #max(_second_highest_value(m) for m in matrices)
    limits = (
        [(global_vmin, global_vmax)] * len(matrices)
        if heatmap_limits is None else heatmap_limits
    )
    if len(limits) != len(matrices):
        raise ValueError("Provide one (vmin, vmax) pair per heatmap.")

    last_im = None
    for ax, mat, title, (vmin, vmax) in zip(
        heat_axes, matrices, titles[1:], limits
    ):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        last_im = im
        if heatmap_limits is not None:
            fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
        ax.set_title(title)
        ax.set_ylabel("Layer")
        ax.set_xlim(-0.5, n_epochs - 0.5)
        ax.set_ylim(n_layers - 0.5, -0.5)

        # sparse ticks only
        xstep = max(1, n_epochs // 20)
        ystep = max(1, n_layers // 12)

        xticks = np.arange(0, n_epochs, xstep)
        yticks = np.arange(0, n_layers, ystep)

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(epochs[i]) for i in xticks], rotation=90)
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(layers[i]) for i in yticks])

    ax_bottom.set_xlabel("Epoch")

    # one shared colorbar for all heatmaps
    if last_im is not None and heatmap_limits is None:
        fig.colorbar(last_im, ax=heat_axes, shrink=0.9, pad=0.01)
    # --- images strip ---
    if want_images:
        n_imgs = len(img_indices)
        rel_paths, labels = _read_imglist(imglist_pth)

        sub = gs[len(matrices) + 1, 0].subgridspec(nrows=1, ncols=n_imgs, wspace=0.02)
        for k, idx in enumerate(img_indices):
            ax_img = fig.add_subplot(sub[0, k])
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.axis("off")

            if not (0 <= idx < len(rel_paths)):
                continue

            img_path = _resolve_image_path(data_dir, rel_paths[idx])
            try:
                im = _load_preview_image(img_path, max_side=96)
                ax_img.imshow(im, interpolation="nearest")
                ax_img.text(
                0.02,
                0.04,
                f"{acts_values[k]:.2f}",
                transform=ax_img.transAxes,
                fontsize=7,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(
                    facecolor="black",
                    alpha=0.65,
                    pad=1.2,
                    edgecolor="none",
                ),
                )
                if show_image_labels:
                    ax_img.set_title(f"i={idx}, y={labels[idx]}", fontsize=6, pad=2)
            except Exception:
                pass

        ax_cat = fig.add_subplot(gs[len(matrices)+2, 0])
        ax_cat.set_xlim(-0.5, n_imgs - 0.5)
        ax_cat.set_ylim(0, 1)
        ax_cat.set_yticks([])

        # Example grouping into 3 equal groups if divisible by 3
        if n_imgs % 3 == 0:
            group_size = n_imgs // 3
            centers = [group_size * i + (group_size - 1) / 2 for i in range(3)]
            ax_cat.set_xticks(centers)
            ax_cat.set_xticklabels(["low", "mid", "high"])

            for cut in range(group_size, n_imgs, group_size):
                ax_cat.axvline(cut - 0.5, ymin=0.0, ymax=1.0, alpha=0.35)

        for spine in ("top", "left", "right"):
            ax_cat.spines[spine].set_visible(False)
        ax_cat.tick_params(axis="x", length=0)
        ax_cat.set_xlabel("Category")

    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def _save_figure_light(fig: plt.Figure, save_path: Path, dpi: int = 150) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = save_path.suffix.lower()
    if suffix in [".jpg", ".jpeg"]:
        fig.savefig(save_path, dpi=dpi, quality=85, bbox_inches="tight")
    else:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

def plot_top_activating_images_per_class(
    num_classes: int,
    activations: Optional[Sequence[float]],   # CLS-based scores if ranking_source="cls"
    save_path: Path,
    *,
    # Option A: provide labels + image paths directly
    labels: Optional[Sequence[int]] = None,
    img_paths: Optional[Sequence[str]] = None,
    # Option B: provide dataset roots like before
    data_dir: Optional[Path] = None,
    imglist_pth: Optional[Path] = None,
    # Ranking mode
    ranking_source: Literal["cls", "patch"] = "cls",
    patch_rank_reduce: Literal["max", "mean", "sum_positive", "q95"] = "max",
    drop_cls_for_patch_ranking: bool = False,
    # Memory-efficient patch-SAE support
    sae: Optional[Any] = None,
    patch_latents: Optional[Sequence[np.ndarray]] = None,  # raw token embeddings [T,D] or [1+T,D]
    feature_idx: Optional[int] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    drop_cls_for_heatmap: bool = False,
    sae_device: str = "cpu",
    sae_token_batch_size: int = 4096,
    sae_encode_feature_fn: Optional[Callable[[Any, torch.Tensor, int], torch.Tensor]] = None,
    # Or pass precomputed patch activations
    patch_activations: Optional[Sequence[np.ndarray]] = None,
    heatmap_alpha: float = 0.38,
    heatmap_cmap: str = "jet",
    heatmap_positive_only: bool = True,
    # plotting options
    top_k: int = 10,
    preview_max_side: int = 96,
    title: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    sort_descending: bool = True,
    show_only_positive_summary: bool = True,
    max_plotted_classes: int = 20,
) -> None:
    """
    Memory-efficient version:
      - rank by CLS scores or patch scores
      - patch heatmaps are computed only for selected images

    ranking_source:
      - "cls": use `activations`
      - "patch": derive one scalar score per image from patch activations

    If num_classes > max_plotted_classes, only classes with the highest
    single-image positive activation are plotted.

    This means that a class with only one strongly activating image can be shown
    instead of a large class whose images are all weakly activated.
    """

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

    n = len(labels_arr)

    if class_names is not None and len(class_names) != num_classes:
        raise ValueError(
            f"len(class_names)={len(class_names)} must match num_classes={num_classes}"
        )

    # --- prepare ranking scores ---
    cls_scores = None

    if activations is not None:
        cls_scores = np.asarray(activations, dtype=float)

        if len(cls_scores) != n:
            raise ValueError(
                f"Length mismatch: len(activations)={len(cls_scores)}, len(labels)={n}"
            )

    if ranking_source == "cls":
        if cls_scores is None:
            raise ValueError("`activations` must be provided when ranking_source='cls'.")

        ranking_scores = cls_scores.copy()

    elif ranking_source == "patch":
        ranking_scores = _compute_patch_ranking_scores(
            patch_latents=patch_latents,
            patch_activations=patch_activations,
            sae=sae,
            feature_idx=feature_idx,
            patch_grid=patch_grid,
            drop_cls_for_patch_ranking=drop_cls_for_patch_ranking,
            patch_rank_reduce=patch_rank_reduce,
            device=sae_device,
            sae_token_batch_size=sae_token_batch_size,
            sae_encode_feature_fn=sae_encode_feature_fn,
        )

        if len(ranking_scores) != n:
            raise ValueError(
                f"Length mismatch: len(ranking_scores)={len(ranking_scores)}, len(labels)={n}"
            )

    else:
        raise ValueError(f"Unknown ranking_source: {ranking_source}")

    # --- validity mask ---
    valid = np.isfinite(ranking_scores)

    ranking_scores = ranking_scores[valid]
    labels_arr = labels_arr[valid]
    resolved_paths = [p for p, v in zip(resolved_paths, valid) if v]

    if cls_scores is not None:
        cls_scores = cls_scores[valid]

    if patch_latents is not None:
        patch_latents = [pl for pl, v in zip(patch_latents, valid) if v]

    if patch_activations is not None:
        patch_activations = [pa for pa, v in zip(patch_activations, valid) if v]

    # --- per-class ordering using chosen ranking scores ---
    class_to_sorted_indices: List[np.ndarray] = []

    # Used only for selecting which classes to show when there are too many.
    # For each class, this stores the maximum positive activation in that class.
    class_peak_activation = np.full(num_classes, -np.inf, dtype=float)

    for c in range(num_classes):
        idx = np.where(labels_arr == c)[0]

        # Keep only positive scores.
        idx = idx[ranking_scores[idx] > 0.0]

        if idx.size == 0:
            class_to_sorted_indices.append(idx)
            continue

        class_peak_activation[c] = float(np.max(ranking_scores[idx]))

        order = np.argsort(ranking_scores[idx])

        if sort_descending:
            order = order[::-1]

        class_to_sorted_indices.append(idx[order])

    # --- choose classes to plot ---
    # If there are more than max_plotted_classes, select classes by their
    # strongest single activating image, not by how many images the class has.
    valid_classes = np.where(np.isfinite(class_peak_activation))[0]

    if len(valid_classes) == 0:
        raise ValueError("No classes with positive valid activations to plot.")

    if num_classes > max_plotted_classes:
        class_order = valid_classes[
            np.argsort(class_peak_activation[valid_classes])[::-1]
        ]
        plotted_classes = class_order[:max_plotted_classes]
    else:
        plotted_classes = valid_classes

    n_plot_classes = len(plotted_classes)

    # --- identify only the images that will be shown ---
    selected_indices: List[int] = []

    for c in plotted_classes:
        sorted_idx = class_to_sorted_indices[c]
        n_show = min(top_k, len(sorted_idx))
        selected_indices.extend(sorted_idx[:n_show].tolist())

    selected_indices = sorted(set(selected_indices))

    # --- build patch maps only for selected images ---
    patch_map_dict: Dict[int, np.ndarray] = {}

    if patch_activations is not None:
        for i in selected_indices:
            pm = _tokens_to_patch_map(
                np.asarray(patch_activations[i]),
                patch_grid=patch_grid,
                drop_cls_token=drop_cls_for_heatmap,
            )
            patch_map_dict[i] = pm

    elif sae is not None and patch_latents is not None and feature_idx is not None:
        patch_map_dict = _encode_patch_maps_for_selected(
            selected_indices=selected_indices,
            patch_latents=patch_latents,
            sae=sae,
            feature_idx=feature_idx,
            patch_grid=patch_grid,
            drop_cls_for_heatmap=drop_cls_for_heatmap,
            device=sae_device,
            sae_token_batch_size=sae_token_batch_size,
            sae_encode_feature_fn=sae_encode_feature_fn,
        )

    # --- top summary: normalized activation/ranking mass by class ---
    if show_only_positive_summary:
        weights = np.clip(ranking_scores, 0.0, None)

        if weights.sum() == 0:
            weights = np.abs(ranking_scores)
    else:
        weights = np.abs(ranking_scores)

    class_mass = np.bincount(
        labels_arr,
        weights=weights,
        minlength=num_classes,
    ).astype(float)

    if class_mass.sum() > 0:
        class_mass /= class_mass.sum()

    class_mass_plot = class_mass[plotted_classes]

    # --- figure layout ---
    fig_w = max(14.0, 1.45 * top_k)
    fig_h = max(2.8 + 1.45 * n_plot_classes, 6.0)

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)

    gs = fig.add_gridspec(
        nrows=n_plot_classes + 1,
        ncols=top_k,
        height_ratios=[1.3] + [1.0] * n_plot_classes,
        hspace=0.04,
        wspace=0.02,
    )

    # --- top bar plot ---
    ax_top = fig.add_subplot(gs[0, :])

    x = np.arange(n_plot_classes)

    ax_top.bar(x, class_mass_plot)
    ax_top.set_xlim(-0.5, n_plot_classes - 0.5)
    ax_top.set_ylabel("Norm. mass")

    if class_names is None:
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([str(c) for c in plotted_classes])
    else:
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(
            [class_names[c] for c in plotted_classes],
            rotation=45,
            ha="right",
        )

    if title is not None:
        ax_top.set_title(title)
    else:
        if num_classes > max_plotted_classes:
            ax_top.set_title(
                f"Normalized activation mass by class "
                f"({ranking_source} ranking, top {n_plot_classes} classes by peak activation)"
            )
        else:
            ax_top.set_title(
                f"Normalized activation mass by class ({ranking_source} ranking)"
            )

    ax_top.grid(True, axis="y", alpha=0.25)

    # --- image grid ---
    for row_idx, c in enumerate(plotted_classes):
        sorted_idx = class_to_sorted_indices[c]
        n_show = min(top_k, len(sorted_idx))

        for j in range(top_k):
            ax = fig.add_subplot(gs[row_idx + 1, j])
            ax.set_xticks([])
            ax.set_yticks([])

            if j >= n_show:
                ax.set_axis_off()
                continue

            i = sorted_idx[j]
            img_path = resolved_paths[i]
            score = float(ranking_scores[i])

            try:
                im = _load_preview_image(img_path, max_side=preview_max_side)
                im = np.asarray(im)
                h_img, w_img = im.shape[:2]
                ax.imshow(im, interpolation="nearest")

            except Exception:
                ax.set_axis_off()
                continue

            if i in patch_map_dict:
                pm = _normalize_patch_map_for_display(
                    patch_map_dict[i],
                    positive_only=heatmap_positive_only,
                )

                ax.imshow(
                    pm,
                    cmap=heatmap_cmap,
                    alpha=heatmap_alpha,
                    interpolation="bilinear",
                    extent=(0, w_img, h_img, 0),
                )

            ax.text(
                0.02,
                0.04,
                f"{score:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                ha="left",
                va="bottom",
                bbox=dict(
                    facecolor="black",
                    alpha=0.65,
                    pad=1.2,
                    edgecolor="none",
                ),
            )

            if j == 0:
                row_label = class_names[c] if class_names is not None else f"class {c}"

                ax.set_ylabel(
                    row_label,
                    rotation=0,
                    labelpad=28,
                    va="center",
                    fontsize=12,
                )

    _save_figure_light(fig, save_path, dpi=300)
    plt.close(fig)

def _plot_three_heatmaps(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    matrix3: np.ndarray,
    accs: Sequence[float],
    epochs: List[int],
    layers: List[int],
    titles: Tuple[str, str, str],
    save_path: Path,
    *,
    vmin: Optional[float] = 0.0,
    cmap: str = "viridis",
    acc_ylabel: str = "Accuracy",
    acc_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    if len(accs) != len(epochs):
        raise ValueError(f"len(accs)={len(accs)} must match len(epochs)={len(epochs)}")
    
    fig_w = max(6.0, 0.7 * len(epochs))
    base_h = max(2.8, 0.5 * len(layers))
    acc_h = max(1.4, 0.35 * base_h)  # dedicated height for the accuracy strip
    fig_h = base_h * 3.0 + acc_h

    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
    gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[acc_h, base_h, base_h, base_h])

    # --- accuracy axis (above the top heatmap) ---
    ax_acc = fig.add_subplot(gs[0, 0])

    # Heatmap axes (share x/y among themselves)
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax1, sharey=ax1)
    heat_axes = (ax1, ax2, ax3)

    # x positions that align with heatmap column centers (0.5, 1.5, ..., n-0.5)
    x = np.arange(len(epochs)) + 0.5
    ax_acc.plot(x, np.asarray(accs, dtype=float))
    ax_acc.set_xlim(0, len(epochs))
    ax_acc.set_ylabel(acc_ylabel)
    if acc_ylim is not None:
        ax_acc.set_ylim(*acc_ylim)
    ax_acc.grid(True, axis="y", alpha=0.3)
    ax_acc.tick_params(axis="x", labelbottom=False)  # keep epoch labels only on bottom heatmap

    # --- heatmaps ---
    for ax, mat, title in zip(heat_axes, (matrix1, matrix2, matrix3), titles):
        vmax = _second_highest_value(mat)
        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=[str(e) for e in epochs],
            yticklabels=[str(l) for l in layers],
            vmin=min(mat.min(),0.0),
            vmax=vmax,
            cmap=cmap,
            cbar=True,
        )
        ax.set_title(title)
        ax.set_ylabel("Layer")

    heat_axes[-1].set_xlabel("Epoch")

    _ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def _default_sae_encode_feature(
    sae: Any,
    x_tokens: torch.Tensor,   # [M, D]
    feature_idx: int,
) -> torch.Tensor:
    """
    Default assumption:
      - sae.encode(x_tokens) returns [M, F]
      or
      - sae(x_tokens) returns [M, F]
    Adapt this if your SAE API differs.
    """
    if hasattr(sae, "encode"):
        z = sae.encode(x_tokens)
    else:
        z = sae(x_tokens)

    if isinstance(z, tuple):
        z = z[0]

    if z.ndim != 2:
        raise ValueError(
            f"Expected SAE output shape [num_tokens, num_features], got {tuple(z.shape)}"
        )

    return z[:, feature_idx]   # [M]


def _compute_patch_ranking_scores(
    *,
    patch_latents: Optional[Sequence[np.ndarray]] = None,   # raw [T,D]
    patch_activations: Optional[Sequence[np.ndarray]] = None,  # precomputed [T], [H,W], or [1+T]
    sae: Optional[Any] = None,
    feature_idx: Optional[int] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    drop_cls_for_patch_ranking: bool = True,
    patch_rank_reduce: Literal["max", "mean", "sum_positive", "q95"] = "max",
    device: str = "cpu",
    sae_token_batch_size: int = 4096,
    sae_encode_feature_fn: Optional[Callable[[Any, torch.Tensor, int], torch.Tensor]] = None,
) -> np.ndarray:
    """
    Returns one scalar patch-based ranking score per image.

    Memory-efficient:
      - does NOT store patch maps for all images
      - only stores one scalar per image
    """
    if sae_encode_feature_fn is None:
        sae_encode_feature_fn = _default_sae_encode_feature

    # Case 1: patch activations are already given
    if patch_activations is not None:
        scores = np.empty(len(patch_activations), dtype=float)

        for i, pa in enumerate(patch_activations):
            pm = _tokens_to_patch_map(
                np.asarray(pa),
                patch_grid=patch_grid,
                drop_cls_token=drop_cls_for_patch_ranking,
            )
            scores[i] = _reduce_token_scores(pm, mode=patch_rank_reduce)

        return scores

    # Case 2: compute patch feature activations from raw patch latents through SAE
    if patch_latents is None or sae is None or feature_idx is None:
        raise ValueError(
            "For patch-based ranking, provide either `patch_activations` "
            "or (`patch_latents`, `sae`, and `feature_idx`)."
        )

    scores = np.empty(len(patch_latents), dtype=float)

    for i, toks in enumerate(patch_latents):
        if isinstance(toks, torch.Tensor):
            toks_np = toks.detach().cpu().numpy()
        else:
            toks_np = np.asarray(toks)

        if toks_np.ndim != 2:
            raise ValueError(
                f"patch_latents[{i}] must have shape [T,D] for patch ranking, got {toks_np.shape}"
            )

        toks_np_spatial = toks_np[1:] if drop_cls_for_patch_ranking else toks_np
        num_tokens = toks_np_spatial.shape[0]

        token_scores_chunks = []
        for start in range(0, num_tokens, sae_token_batch_size):
            chunk = toks_np_spatial[start:start + sae_token_batch_size]
            chunk_t = torch.as_tensor(chunk, dtype=torch.float32, device=device)

            with torch.no_grad():
                feat_scores = sae_encode_feature_fn(sae, chunk_t, feature_idx)

            token_scores_chunks.append(feat_scores.detach().cpu().numpy())

        token_scores = np.concatenate(token_scores_chunks, axis=0)  # [T]
        scores[i] = _reduce_token_scores(token_scores, mode=patch_rank_reduce)

    return scores


def _encode_patch_maps_for_selected(
    selected_indices: Sequence[int],
    patch_latents: Sequence[np.ndarray],
    sae: Any,
    feature_idx: int,
    *,
    patch_grid: Optional[Tuple[int, int]] = None,
    drop_cls_for_heatmap: bool = False,
    device: str = "cpu",
    sae_token_batch_size: int = 4096,
    sae_encode_feature_fn: Optional[Callable[[Any, torch.Tensor, int], torch.Tensor]] = None,
) -> Dict[int, np.ndarray]:
    """
    Encodes patch tokens through the SAE only for selected images.

    Returns:
      dict: image_index -> patch_map [H, W]
    """
    if sae_encode_feature_fn is None:
        sae_encode_feature_fn = _default_sae_encode_feature

    patch_map_dict: Dict[int, np.ndarray] = {}

    for i in selected_indices:
        toks = patch_latents[i]
        if isinstance(toks, torch.Tensor):
            toks_np = toks.detach().cpu().numpy()
        else:
            toks_np = np.asarray(toks)

        if toks_np.ndim != 2:
            raise ValueError(
                f"patch_latents[{i}] must have shape [T,D], got {toks_np.shape}"
            )

        # split off CLS for spatial map
        if drop_cls_for_heatmap:
            toks_np_spatial = toks_np[1:] if toks_np.shape[0] >= 2 else toks_np
        else:
            toks_np_spatial = toks_np

        num_tokens = toks_np_spatial.shape[0]

        if patch_grid is None:
            side = int(round(np.sqrt(num_tokens)))
            if side * side != num_tokens:
                raise ValueError(
                    f"Cannot infer square grid from {num_tokens} patch tokens for image {i}. "
                    f"Pass patch_grid explicitly."
                )
            gh, gw = side, side
        else:
            gh, gw = patch_grid
            if gh * gw != num_tokens:
                raise ValueError(
                    f"patch_grid={patch_grid} incompatible with {num_tokens} spatial tokens "
                    f"for image {i}."
                )

        token_scores_chunks = []
        for start in range(0, num_tokens, sae_token_batch_size):
            chunk = toks_np_spatial[start:start + sae_token_batch_size]
            chunk_t = torch.as_tensor(chunk, dtype=torch.float32, device=device)

            with torch.no_grad():
                feat_scores = sae_encode_feature_fn(sae, chunk_t, feature_idx)

            token_scores_chunks.append(feat_scores.detach().cpu().numpy())

        token_scores = np.concatenate(token_scores_chunks, axis=0)  # [T]
        patch_map = token_scores.reshape(gh, gw)
        patch_map_dict[i] = patch_map

    return patch_map_dict


