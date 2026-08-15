from pathlib import Path
import math
from typing import List, Tuple, Optional, Dict, Any, Sequence
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn

def make_heatmap(sae,all_data_dict,arguments_dict, threshold = 1e-3, type_data = "train", backbone = "vit", folder_name = "", device = "cpu" ):
    sae_type = sae.sae_type
    latents = torch.from_numpy(all_data_dict[type_data]["activations"]).float().to(device)
    train_latents = sae(latents)[1].detach().cpu().numpy()
    y_pred_tr = nn.functional.softmax(torch.tensor(all_data_dict[type_data]["logits"])).argmax(1)

    y_tr = np.array(all_data_dict[type_data]["labels"]["sh"])
    N, C = train_latents.shape
    heat_map_matric,col_names,row_names = get_heat_map_matrix(all_data_dict[type_data]["labels"],train_latents,y_pred_tr,y_tr,threshold)
    file_name = f'{sae_type} Activation_heatmap_threshold_{threshold}_{type_data}.png'
    plot_concept_label_heatmap(heat_map_matric[:250],label_names=col_names,concept_names=row_names, sort_rows_by_mean=True,file_name = file_name,N=N,vmax=0.5,folder_name=folder_name)
    return heat_map_matric, col_names, row_names

def get_heat_map_matrix(labels,latents,y_pred_tr,y_tr,threshold):
    heat_map_matric = []
    row_names = []
    col_names = ["N","Acc"]
    min_lens = []
    N, C = latents.shape
    for lab in ["fh","wh","oh","sc","sh","or"]:
        min_len = np.array(labels[lab]).max()+1
        a = np.arange(min_len)
        min_lens.append(min_len)
        b= [ lab + str(ai)  for ai in a]
        col_names.extend(b)

    for i, cid in enumerate(range(C)):
        row = []
        z = latents[:, cid]
        active_mask = z > threshold
        if sum(active_mask) < 10:
            continue
        row_names.append(str(cid))
        row.append(np.mean(active_mask))
        pos_acts = z[active_mask]
        acc = accuracy_score(y_tr[active_mask],y_pred_tr[active_mask])
        row.append(acc)
        # Label distribution -------------------------------------------------------
        fhs = np.bincount(np.array(labels["fh"])[active_mask],minlength=min_lens[0])/sum(active_mask)
        whs = np.bincount(np.array(labels["wh"])[active_mask],minlength=min_lens[1])/sum(active_mask)
        ohs = np.bincount(np.array(labels["oh"])[active_mask],minlength=min_lens[2])/sum(active_mask)
        scs = np.bincount(np.array(labels["sc"])[active_mask],minlength=min_lens[3])/sum(active_mask)
        shs = np.bincount(np.array(labels["sh"])[active_mask],minlength=min_lens[4])/sum(active_mask)
        ors = np.bincount(np.array(labels["or"])[active_mask],minlength=min_lens[5])/sum(active_mask)
        row.extend(fhs)
        row.extend(whs)
        row.extend(ohs)
        row.extend(scs)
        row.extend(shs)
        row.extend(ors)
        heat_map_matric.append(row)
    return heat_map_matric, col_names, row_names

def plot_concept_label_heatmap(
    acc_matrix,
    concept_names=None,
    label_names=None,
    sort_rows_by_mean=False,
    sort_cols_by_mean=False,
    vmin=0.0,
    vmax=1.0,
    file_name = "",
    folder_name = "",
    N=10000
):
    """
    Plot a heatmap of concept × label accuracies.

    Parameters
    ----------
    acc_matrix : array-like of shape (n_concepts, n_labels)
        Accuracy (or any scalar metric) per concept/label pair.
    concept_names : list of str or None
        Names for concepts (rows). Length = n_concepts.
    label_names : list of str or None
        Names for labels (columns). Length = n_labels.
    sort_rows_by_mean : bool
        If True, sort concepts by descending row-mean.
    sort_cols_by_mean : bool
        If True, sort labels by descending column-mean.
    vmin, vmax : float
        Color scale limits (e.g. 0–1 for accuracy).
    """
    acc = np.asarray(acc_matrix)
    n_concepts, n_labels = acc.shape

    # ---- Sorting ----
    row_order = np.arange(n_concepts)
    col_order = np.arange(n_labels)

    if sort_rows_by_mean:
        row_order = np.argsort(-acc.T[0].T)  # descending
    if sort_cols_by_mean:
        col_order = np.argsort(-acc.mean(axis=0))  # descending

    acc_sorted = acc[row_order][:, col_order]

    if concept_names is not None:
        concept_names = [concept_names[i] for i in row_order]
    if label_names is not None:
        label_names = [label_names[j] for j in col_order]

    # ---- Figure size heuristic ----
    # Scale width with number of labels, height with number of concepts
    fig_width = min(16, 4 + 0.15 * n_labels)
    fig_height = min(18, 4 + 0.08 * n_concepts)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        acc_sorted,
        aspect='auto',
        origin='lower',
        vmin=vmin,
        vmax=vmax
    )
    ax.set_xlabel("Label")
    ax.set_ylabel("Concept")

    # ---- X-axis labels ----
    ax.tick_params(axis='y', labelsize=6)  # or 4, 8, etc.

    if label_names is not None:
        ax.set_xticks(np.arange(n_labels))
        ax.set_xticklabels(label_names, rotation=90)
    else:
        ax.set_xticks([])

    # ---- Y-axis labels ----
    if concept_names is not None and n_concepts <= 300:
        # Only label all concepts if there aren't too many
        ax.set_yticks(np.arange(n_concepts))
        ax.set_yticklabels(concept_names)
    elif concept_names is not None:
        # For many concepts, only label every Nth
        step = max(1, n_concepts // 40)
        tick_positions = np.arange(0, n_concepts, step)
        tick_labels = [concept_names[i] for i in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    else:
        ax.set_yticks([])

    
    for i in range(n_concepts):
            value = acc_sorted[i, 0]*N
            ax.text(
                0, i,                          # x, y: note order (col=x, row=y)
                f"{value:.0f}",                # text: format as you like
                ha="center", va="center",      # centered in the cell
                color="white", size=4,bbox=dict(boxstyle="round,pad=0.2", fc=(0,0,0,0.5), ec="none")               # or "white" if background is dark
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")


    plt.title(file_name)
    plt.tight_layout()
    #os.makedirs(f"{folder_name}", exist_ok=True)
    plt.savefig(f"{folder_name}/{file_name}", dpi=600)
    plt.show()


def show_images_grid(
    data,
    filenames: Sequence[str],
    max_cols: int = 10,
    titles: Optional[Sequence[str]] = None,
    figsize_per_cell: float = 2.0,
    turn_off_axes: bool = True,
    save_path: Optional[str] = None,
):
    """
    Display images in a grid with up to `max_cols` columns.

    Args:
        root: Root directory for images (can be "" if `filenames` are absolute).
        filenames: List of image paths (relative to `root`, or absolute paths).
        max_cols: Maximum number of columns in the grid.
        titles: Optional list of titles per image (same length as filenames).
        figsize_per_cell: Size multiplier for each cell in inches.
        turn_off_axes: If True, hides axes.
        save_path: If provided, saves the figure to this path.

    Returns:
        (fig, axes) from matplotlib for further customization.
    """
    
    paths = []
    infos = []
    for name in filenames:
        p = int(name.split("/")[2].split("_")[0])
        infos.append(name.split("/")[2])
        paths.append(p)

    # Load images (RGB), skipping missing ones with a warning
    imgs, kept_paths, kept_titles = [], [], []
    for i, p in enumerate(paths):
        try:
            img = data[p]
            imgs.append(img)
            kept_paths.append(infos[i])
            if titles is not None and i < len(titles):
                kept_titles.append(infos[i])
            else:
                kept_titles.append(None)
        except Exception as e:
            print(f"[warn] Could not load {p}: {e}")

    n = len(imgs)
    if n == 0:
        print("No images to display.")
        return None, None

    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig_w = max(1, cols * figsize_per_cell)
    fig_h = max(1, rows * figsize_per_cell)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes, dtype=object).reshape(rows, cols)  # handles 1D cases

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < n:
                ax.imshow(imgs[idx])
                if turn_off_axes:
                    ax.axis("off")
                if kept_titles[idx]:
                    ax.set_title(str(kept_titles[idx]), fontsize=9)
            else:
                ax.axis("off")
            idx += 1

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()

    return fig, axes
