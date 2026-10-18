"""Check the train/val/test split, and measure how much a random split leaks.

Run from the project root:  python -m scripts.check_split
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src import config
from src.dataset import landmark_splits, load_letters, split_indices, to_loader


def main():
    data = load_letters()
    labels, frames = data["labels"], data["frames"]

    for mode in ["block", "random"]:
        idx = split_indices(labels, frames, mode)
        sizes = {k: len(v) for k, v in idx.items()}
        total = sum(sizes.values())
        assert total == len(labels), "split lost or duplicated samples"
        assert not set(idx["train"]) & set(idx["test"]), "train and test overlap"
        assert not set(idx["train"]) & set(idx["val"]), "train and val overlap"
        shares = "  ".join(f"{k} {n:,} ({n / total:.1%})" for k, n in sizes.items())
        print(f"{mode:>6} split: {shares}")

        if mode == "block":
            per_letter = [np.sum(labels[idx["test"]] == c) / np.sum(labels == c) for c in range(len(config.LETTERS))]
            print(f"        each letter's share in test: {min(per_letter):.1%}..{max(per_letter):.1%}")
            enhanced_test = data["enhanced"][idx["test"]].mean()
            print(f"        enhanced samples in test: {enhanced_test:.1%}")

    # 1-nearest-neighbour copies the label of the most similar training sample.
    # If test samples have near-duplicates in train, it scores suspiciously high.
    print("\n1-NN accuracy (sensitive to near-duplicates between train and test):")
    for mode in ["random", "block"]:
        s = landmark_splits(mode, mirror_train=False)
        knn = KNeighborsClassifier(n_neighbors=1).fit(*s["train"])
        print(f"  {mode:>6} split: {knn.score(*s['test']):.1%}")

    s = landmark_splits("block")
    print(f"\nWith mirror augmentation, train doubles to {len(s['train'][0]):,} samples")
    s_no_enh = landmark_splits("block", include_enhanced=False)
    assert np.array_equal(s_no_enh["test"][1], s["test"][1]), "test set must not depend on include_enhanced"
    print(f"Without enhanced samples: train {len(s_no_enh['train'][0]):,}, test unchanged ({len(s['test'][0]):,})")

    X, y = next(iter(to_loader(*s["train"], batch_size=32, shuffle=True)))
    print(f"Landmark batch: {tuple(X.shape)} {X.dtype}, labels {tuple(y.shape)}")


if __name__ == "__main__":
    main()
