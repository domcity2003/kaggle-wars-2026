"""Estimate validation weights from nearest-training-row distances."""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from train_cat import DATA, load_data


def main():
    train, _, _ = load_data()
    train_distance = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1)
    test_distance = None
    for fold in range(3):
        with np.load(DATA / f"knn15_block5000_{fold}.npz") as predictions:
            indices = predictions["val_idx"]
            if np.any(fold_ids[indices] != -1):
                raise ValueError("Grouped validation folds overlap")
            fold_ids[indices] = fold
            train_distance[indices] = predictions["val_distance"]
            if test_distance is None:
                test_distance = predictions["test_distance"]
            elif not np.array_equal(test_distance, predictions["test_distance"]):
                raise ValueError("Test distances differ across folds")

    edges = np.arange(0, 0.31, 0.005)
    if any(not np.isfinite(values).all() or values.max() >= edges[-1]
           for values in (train_distance, test_distance)):
        raise ValueError("Expected finite distances within the histogram bins")

    source_counts = np.histogram(train_distance, edges)[0]
    target_counts = np.histogram(test_distance, edges)[0]
    source_frequency = gaussian_filter1d(source_counts.astype(float), sigma=1.5, mode="nearest")
    target_frequency = gaussian_filter1d(target_counts.astype(float), sigma=1.5, mode="nearest")
    source_frequency /= source_frequency.sum()
    target_frequency /= target_frequency.sum()
    density_ratio = target_frequency / np.maximum(source_frequency, 1e-7)

    centers = (edges[:-1] + edges[1:]) / 2
    ratio = np.interp(train_distance, centers, density_ratio)
    weights = np.minimum(ratio, 5)
    weights /= weights.mean()
    conservative = np.minimum(ratio, 2)
    conservative /= conservative.mean()
    path = DATA / "distance_block5000.npz"
    np.savez(
        path, train_distance=train_distance, test_distance=test_distance,
        train_weight=weights, train_weight_conservative=conservative, fold=fold_ids,
        density_bin_edges=edges, validation_counts=source_counts, test_counts=target_counts,
    )
    print(f"Wrote {path.name}")


if __name__ == "__main__":
    main()
