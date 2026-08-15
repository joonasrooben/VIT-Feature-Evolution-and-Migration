from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[:, None]
    return x


def _fit_logreg_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 2000,
    class_weight: Optional[str] = "balanced",
    score_fn: Callable[[np.ndarray, np.ndarray], float] = balanced_accuracy_score,
) -> float:
    X_train = _as_2d(np.asarray(X_train, dtype=float))
    X_test = _as_2d(np.asarray(X_test, dtype=float))
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    if np.unique(y_train).size < 2:
        return np.nan
    if np.unique(y_test).size < 2:
        return np.nan
    if np.allclose(X_train.std(axis=0), 0.0):
        return np.nan

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="liblinear",
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
        ),
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return float(score_fn(y_test, y_pred))


def _columnwise_minmax_for_heatmap(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)

    for j in range(values.shape[1]):
        col = values[:, j]
        mask = np.isfinite(col)
        if not np.any(mask):
            continue

        cmin = np.min(col[mask])
        cmax = np.max(col[mask])

        if np.isclose(cmin, cmax):
            out[mask, j] = 0.5
        else:
            out[mask, j] = (col[mask] - cmin) / (cmax - cmin)

    return out


def _make_annotation_strings(values: np.ndarray, col_names: Sequence[str]) -> np.ndarray:
    annot = np.empty(values.shape, dtype=object)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                annot[i, j] = "-"
            elif "count" in col_names[j].lower() or "nr" in col_names[j].lower():
                annot[i, j] = f"{int(round(v))}"
            else:
                annot[i, j] = f"{v:.3f}"

    return annot

def precompute_shared_subclass_metrics(
    *,
    subclass_names: Sequence[str],
    subclass_labels_train: Sequence[Union[int, str]],
    subclass_labels_test: Sequence[Union[int, str]],
    full_activation_train: np.ndarray,
    full_activation_test: np.ndarray,
    full_sae_train: np.ndarray,
    full_sae_test: np.ndarray,
    save_path :Path,
    probe_C: float = 1.0,
    probe_max_iter: int = 2000,
    score_fn: Callable[[np.ndarray, np.ndarray], float] = balanced_accuracy_score,
) -> pd.DataFrame:
    """
    Metrics here do NOT depend on the individual concept.
    Compute once and reuse for all concepts.
    """
    subclass_names = list(subclass_names)
    subclass_labels_train = np.asarray(subclass_labels_train)
    subclass_labels_test = np.asarray(subclass_labels_test)

    full_activation_train = np.asarray(full_activation_train, dtype=float)
    full_activation_test = np.asarray(full_activation_test, dtype=float)
    full_sae_train = np.asarray(full_sae_train, dtype=float)
    full_sae_test = np.asarray(full_sae_test, dtype=float)
    
    rows = []
    for subclass in subclass_names:
        y_train = (subclass_labels_train == subclass).astype(int)
        y_test = (subclass_labels_test == subclass).astype(int)

        test_count = int(np.sum(subclass_labels_test == subclass))
        acc_full_act = _fit_logreg_probe(
            full_activation_train,
            y_train,
            full_activation_test,
            y_test,
            C=probe_C,
            max_iter=probe_max_iter,
            score_fn=score_fn,
        )

        acc_full_sae = _fit_logreg_probe(
            full_sae_train,
            y_train,
            full_sae_test,
            y_test,
            C=probe_C,
            max_iter=probe_max_iter,
            score_fn=score_fn,
        )
        print(acc_full_act, acc_full_sae, flush = True)
        rows.append(
            {
                "subclass": subclass,
                "test_count": test_count,
                "probe_acc_full_activation": acc_full_act,
                "probe_acc_full_sae": acc_full_sae,
            }
        )
    df = pd.DataFrame(rows).set_index("subclass")
    df.to_csv(save_path)
    return df 

def plot_subclass_probe_heatmap_cached(
    *,
    shared_df: pd.DataFrame,
    subclass_names: Sequence[str],
    subclass_labels_train: Sequence[Union[int, str]],
    subclass_labels_test: Sequence[Union[int, str]],
    sae_concept_train: np.ndarray,   # [N_train] or [N_train,1]
    sae_concept_test: np.ndarray,    # [N_test] or [N_test,1]
    save_path: Path,
    activation_threshold: float = 0.0,
    probe_C: float = 1.0,
    probe_max_iter: int = 2000,
    score_fn: Callable[[np.ndarray, np.ndarray], float] = balanced_accuracy_score,
    cmap: str = "viridis",
    title: Optional[str] = None,
) -> pd.DataFrame:
    """
    Uses precomputed shared_df and only computes the concept-specific columns:
      - activated_frac
      - probe_acc_sae_concept
    """
    subclass_names = list(subclass_names)
    subclass_labels_train = np.asarray(subclass_labels_train)
    subclass_labels_test = np.asarray(subclass_labels_test)

    sae_concept_train = np.asarray(sae_concept_train, dtype=float).reshape(len(subclass_labels_train), -1)
    sae_concept_test = np.asarray(sae_concept_test, dtype=float).reshape(len(subclass_labels_test), -1)

    if sae_concept_train.shape[1] != 1 or sae_concept_test.shape[1] != 1:
        raise ValueError("This function expects a single concept activation per image.")

    concept_rows = []
    for subclass in subclass_names:
        y_train = (subclass_labels_train == subclass).astype(int)
        y_test = (subclass_labels_test == subclass).astype(int)

        test_mask = (subclass_labels_test == subclass)
        test_count = int(np.sum(test_mask))

        if test_count > 0:
            concept_vals = sae_concept_test[test_mask, 0]
            activated_frac = float(np.mean(concept_vals > activation_threshold))
        else:
            activated_frac = np.nan

        acc_concept = _fit_logreg_probe(
            sae_concept_train,
            y_train,
            sae_concept_test,
            y_test,
            C=probe_C,
            max_iter=probe_max_iter,
            score_fn=score_fn,
        )

        concept_rows.append(
            {
                "subclass": subclass,
                "activated_frac": activated_frac,
                "probe_acc_sae_concept": acc_concept,
            }
        )

    concept_df = pd.DataFrame(concept_rows).set_index("subclass")

    # Merge cached and concept-specific metrics
    df = shared_df.loc[subclass_names].copy()
    df["activated_frac"] = concept_df["activated_frac"]
    df["probe_acc_sae_concept"] = concept_df["probe_acc_sae_concept"]

    # reorder columns to your desired layout
    df = df[
        [
            "test_count",
            "activated_frac",
            "probe_acc_sae_concept",
            "probe_acc_full_activation",
            "probe_acc_full_sae",
        ]
    ]
    #df = df[df["activated_frac"] > 0]
    
    values_raw = df.to_numpy(dtype=float)
    values_color = _columnwise_minmax_for_heatmap(values_raw)
    annot = _make_annotation_strings(values_raw, df.columns)

    fig_w = 10.5
    fig_h = max(4.0, 0.52 * len(subclass_names))

    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(
        values_color,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=[
            "test count",
            "activated frac",
            "probe acc\nSAE concept",
            "probe acc\nfull activation",
            "probe acc\nfull SAE",
        ],
        yticklabels=df.index.tolist(),
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.xlabel("Metric")
    plt.ylabel("Subclass")
    plt.title(title or "Subclass summary + logistic probes")
    plt.tight_layout()


    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    return df