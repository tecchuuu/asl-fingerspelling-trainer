"""Load extracted data and split it into train / val / test.

Both models use the same split, so their scores are comparable.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src import config
from src.normalize import mirror, normalize

VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
# Frames are numbered in recording order and neighbours are near-duplicates,
# so whole blocks of consecutive frames go to the same split.
BLOCK_SIZE = 100


def load_letters(with_crops=False):
    keys = ["landmarks", "files", "enhanced"] + (["crops"] if with_crops else [])
    parts = {k: [] for k in keys}
    labels = []
    for i, letter in enumerate(config.LETTERS):
        with np.load(config.PROCESSED_DIR / f"{letter}.npz") as data:
            for k in keys:
                parts[k].append(data[k])
            labels.append(np.full(len(data["files"]), i))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["labels"] = np.concatenate(labels)
    out["frames"] = np.array([int(f[len(config.LETTERS[y]):-4]) for f, y in zip(out["files"], out["labels"])])
    return out


def split_indices(labels, frames, mode="block", seed=config.SEED):
    """Return {"train": idx, "val": idx, "test": idx}, split separately per letter."""
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    for letter in np.unique(labels):
        idx = np.flatnonzero(labels == letter)
        if mode == "block":
            groups = frames[idx] // BLOCK_SIZE
        elif mode == "random":
            groups = np.arange(len(idx))
        else:
            raise ValueError(f"unknown split mode: {mode}")

        # Blocks hold different numbers of samples (failed detections cluster), so fill
        # test and val by sample count, not by number of blocks
        unique, counts = np.unique(groups, return_counts=True)
        order = rng.permutation(len(unique))
        share = np.cumsum(counts[order]) / len(idx)
        test_groups = unique[order[share <= TEST_FRACTION]]
        val_groups = unique[order[(share > TEST_FRACTION) & (share <= TEST_FRACTION + VAL_FRACTION)]]
        splits["test"].append(idx[np.isin(groups, test_groups)])
        splits["val"].append(idx[np.isin(groups, val_groups)])
        splits["train"].append(idx[~np.isin(groups, np.concatenate([test_groups, val_groups]))])
    return {k: np.concatenate(v) for k, v in splits.items()}


def landmark_splits(mode="block", include_enhanced=True, mirror_train=True):
    """Normalized landmark features for the MLP: {"train": (X, y), "val": ..., "test": ...}.

    include_enhanced only filters the training set, so val/test stay identical
    when comparing training with and without the brightened samples.
    """
    data = load_letters()
    X = normalize(data["landmarks"])
    idx = split_indices(data["labels"], data["frames"], mode)
    if not include_enhanced:
        idx["train"] = idx["train"][~data["enhanced"][idx["train"]]]

    out = {name: (X[i], data["labels"][i]) for name, i in idx.items()}
    if mirror_train:
        X_tr, y_tr = out["train"]
        out["train"] = (np.concatenate([X_tr, mirror(X_tr)]), np.concatenate([y_tr, y_tr]))
    return out


def crop_splits(mode="block", include_enhanced=True, mirror_train=True):
    """Hand crops for the CNN, same split as landmark_splits. Images are uint8 (N, H, W, 3)."""
    data = load_letters(with_crops=True)
    idx = split_indices(data["labels"], data["frames"], mode)
    if not include_enhanced:
        idx["train"] = idx["train"][~data["enhanced"][idx["train"]]]

    out = {name: (data["crops"][i], data["labels"][i]) for name, i in idx.items()}
    if mirror_train:
        X_tr, y_tr = out["train"]
        out["train"] = (np.concatenate([X_tr, X_tr[:, :, ::-1]]), np.concatenate([y_tr, y_tr]))
    return out


class CropDataset(Dataset):
    """Keeps crops as uint8 and converts one image at a time: all crops as float would need ~4 GB."""

    def __init__(self, crops, labels):
        # (N, H, W, 3) -> (N, 3, H, W), the layout PyTorch convolutions expect
        self.crops = torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2)
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.crops[i].float() / 255, self.labels[i]


def to_loader(X, y, batch_size, shuffle):
    if X.dtype == np.uint8:
        ds = CropDataset(X, y)
    else:
        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
