from collections import Counter, deque
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError as exc:
    raise ImportError("Please install nltk first, for example: pip install nltk") from exc

try:
    wn.ensure_loaded()
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    wn.ensure_loaded()


IMAGENET_NUM_CLASSES = 1000
IMAGENET_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


def load_labels(imglist_pth: Path) :
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
    return labels


def validate_labels(labels, num_classes=IMAGENET_NUM_CLASSES):
    """Validate that labels are strict 0-based ImageNet class indices."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be a 1D array, got shape {labels.shape}")
    if labels.size == 0:
        raise ValueError("labels must contain at least one image label")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integers")

    invalid_mask = (labels < 0) | (labels >= num_classes)
    if np.any(invalid_mask):
        invalid_values = np.unique(labels[invalid_mask])[:10]
        raise ValueError(
            f"labels must be in [0, {num_classes - 1}]. "
            f"Found invalid values: {invalid_values.tolist()}"
        )


def validate_activations_and_labels(activations, labels):
    """Validate activation matrix shape and image-label alignment."""
    activations = np.asarray(activations)
    labels = np.asarray(labels)

    if activations.ndim != 2:
        raise ValueError(f"activations must be 2D with shape (M, N), got {activations.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got {labels.shape}")
    if activations.shape[1] != labels.shape[0]:
        raise ValueError(
            "Number of activation columns must match number of labels. "
            f"Got activations.shape[1]={activations.shape[1]} and len(labels)={labels.shape[0]}."
        )
    validate_labels(labels)
    return activations, labels


def load_imagenet_class_index(cache_path="imagenet_class_index.json"):
    """Load the standard ImageNet index mapping, downloading it once if needed."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        print(f"Downloading ImageNet class-index mapping to {cache_path} ...")
        urlretrieve(IMAGENET_CLASS_INDEX_URL, cache_path)

    with cache_path.open("r", encoding="utf-8") as file:
        raw_mapping = json.load(file)

    if len(raw_mapping) != IMAGENET_NUM_CLASSES:
        raise ValueError(f"Expected 1000 ImageNet classes, got {len(raw_mapping)}")

    mapping = {}
    for index_string, value in raw_mapping.items():
        class_index = int(index_string)
        synset_id, class_name = value
        mapping[class_index] = {"synset_id": synset_id, "class_name": class_name}

    missing = set(range(IMAGENET_NUM_CLASSES)) - set(mapping)
    if missing:
        raise ValueError(f"ImageNet mapping is missing class indices: {sorted(missing)[:10]}")

    return mapping


def imagenet_synset_id_to_wordnet_synset(synset_id):
    """Convert an ImageNet synset id such as n02124075 to an NLTK WordNet synset."""
    if len(synset_id) != 9 or not synset_id[1:].isdigit():
        raise ValueError(f"Invalid ImageNet synset id: {synset_id!r}")
    return wn.synset_from_pos_and_offset(synset_id[0], int(synset_id[1:]))


def build_index_to_wordnet_synset(imagenet_mapping):
    """Build a dictionary from ImageNet class index to WordNet synset."""
    return {
        index: imagenet_synset_id_to_wordnet_synset(info["synset_id"])
        for index, info in imagenet_mapping.items()
    }


def compute_class_distributions(activations, labels, activation_threshold, num_classes=IMAGENET_NUM_CLASSES):
    """Return an (M, 1000) matrix of normalized class distributions for active images."""
    activations, labels = validate_activations_and_labels(activations, labels)
    num_concepts = activations.shape[0]
    distributions = np.zeros((num_concepts, num_classes), dtype=np.float64)

    for concept_index in range(num_concepts):
        active_mask = activations[concept_index] > activation_threshold
        active_count = int(active_mask.sum())
        if active_count == 0:
            continue

        counts = np.bincount(labels[active_mask], minlength=num_classes)
        distributions[concept_index] = counts / active_count

    return distributions


def ancestors_within_depth(synset, max_depth):
    """Collect hypernym ancestors reachable within max_depth steps."""
    ancestors = set()
    queue = deque((hypernym, 1) for hypernym in synset.hypernyms())

    while queue:
        current, depth = queue.popleft()
        if depth > max_depth or current in ancestors:
            continue
        ancestors.add(current)
        queue.extend((hypernym, depth + 1) for hypernym in current.hypernyms())

    return ancestors


def share_hypernym_ancestor(class_indices, index_to_synset, ancestor_depth=4):
    """Return True if all class synsets share a hypernym ancestor within ancestor_depth."""
    if len(class_indices) < 2:
        return False

    ancestor_sets = []
    for class_index in class_indices:
        synset = index_to_synset[int(class_index)]
        ancestors = ancestors_within_depth(synset, ancestor_depth)
        if not ancestors:
            return False
        ancestor_sets.append(ancestors)

    return bool(set.intersection(*ancestor_sets))


def classify_single_distribution(
    distribution,
    index_to_synset,
    drop_ratio=0.5,
    max_superclass_size=10,
    ancestor_depth=4,
):
    """Classify one concept distribution as type 0, 1, or 2."""
    distribution = np.asarray(distribution, dtype=np.float64)
    sorted_indices = np.argsort(distribution)[::-1]
    sorted_values = distribution[sorted_indices]

    highest = sorted_values[0]
    if highest <= 0:
        return 2

    second_highest = sorted_values[1]
    if second_highest / highest < drop_ratio:
        return 0

    largest_group_size = min(max_superclass_size, len(sorted_values) - 1)
    for group_size in range(2, largest_group_size + 1):
        current_value = sorted_values[group_size - 1]
        next_value = sorted_values[group_size]
        if current_value <= 0:
            break
        if next_value / current_value < drop_ratio:
            candidate_classes = sorted_indices[:group_size]
            if share_hypernym_ancestor(candidate_classes, index_to_synset, ancestor_depth):
                return 1
            return 2

    return 2


def classify_concepts(
    class_distributions,
    index_to_synset,
    drop_ratio=0.5,
    max_superclass_size=10,
    ancestor_depth=4,
):
    """Return a length-M vector of concept types."""
    class_distributions = np.asarray(class_distributions, dtype=np.float64)
    if class_distributions.ndim != 2 or class_distributions.shape[1] != IMAGENET_NUM_CLASSES:
        raise ValueError(
            "class_distributions must have shape (M, 1000), "
            f"got {class_distributions.shape}"
        )

    return np.asarray(
        [
            classify_single_distribution(
                distribution,
                index_to_synset=index_to_synset,
                drop_ratio=drop_ratio,
                max_superclass_size=max_superclass_size,
                ancestor_depth=ancestor_depth,
            )
            for distribution in class_distributions
        ],
        dtype=np.int64,
    )


def run_concept_classification(
    activations,
    labels_txt_path,
    activation_threshold,
    drop_ratio=0.5,
    max_superclass_size=10,
    ancestor_depth=4,
    imagenet_mapping_cache_path="imagenet_class_index.json",
):
    """Compute class distributions and concept types from activations and label file."""
    labels = load_labels(labels_txt_path)
    class_distributions = compute_class_distributions(activations, labels, activation_threshold)

    imagenet_mapping = load_imagenet_class_index(imagenet_mapping_cache_path)
    index_to_synset = build_index_to_wordnet_synset(imagenet_mapping)
    concept_types = classify_concepts(
        class_distributions,
        index_to_synset=index_to_synset,
        drop_ratio=drop_ratio,
        max_superclass_size=max_superclass_size,
        ancestor_depth=ancestor_depth,
    )

    return concept_types, class_distributions


def top_classes_for_concept(distribution, imagenet_mapping=None, top_k=10):
    """Return a small dataframe with the highest-probability classes for one concept."""
    distribution = np.asarray(distribution)
    top_indices = np.argsort(distribution)[::-1][:top_k]
    rows = []
    for class_index in top_indices:
        row = {
            "class_index": int(class_index),
            "probability": float(distribution[class_index]),
        }
        if imagenet_mapping is not None:
            row.update(imagenet_mapping[int(class_index)])
        rows.append(row)
    return pd.DataFrame(rows)