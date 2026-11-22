"""Generate nearest-neighbor predictions and distances for the grouped folds."""

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

from train_cat import DATA, load_data, validation_splits


def predict_neighbors(fit, query, ranks, projected, missing, categories, patterns, labels):
    _, shortlist = cKDTree(projected[fit]).query(projected[query], k=48, eps=0.20, workers=2)
    candidates = fit[shortlist]
    selected = np.empty((len(query), 15), dtype=int)
    distances = np.empty(selected.shape, dtype=float)

    for start in range(0, len(query), 1000):
        rows = slice(start, start + 1000)
        source = query[rows]
        target = candidates[rows]
        observed = ~(missing[source, None, :] | missing[target])
        delta = np.square(ranks[source, None, :] - ranks[target])
        distance = (delta * observed).sum(axis=2) / np.maximum(observed.sum(axis=2), 1)
        distance += 0.001 * (categories[source, None] != categories[target])
        distance += 0.002 * (patterns[source, None] != patterns[target])
        nearest = np.argsort(distance, axis=1)[:, :15]
        selected[rows] = np.take_along_axis(target, nearest, axis=1)
        distances[rows] = np.take_along_axis(np.sqrt(distance), nearest, axis=1)

    probabilities = np.zeros((len(query), 11))
    weights = (distances + 0.005) ** -2
    np.add.at(
        probabilities,
        (np.repeat(np.arange(len(query)), 15), labels[selected].ravel()),
        weights.ravel(),
    )
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return distances[:, 0], probabilities.astype(np.float32)


def main():
    with threadpool_limits(2):
        train, test, labels = load_data()
        columns = [f"X{i:03d}" for i in range(60)]
        numeric = np.concatenate([
            train[columns].to_numpy(dtype=np.float32),
            test[columns].to_numpy(dtype=np.float32),
        ])
        categories = np.concatenate([train["Z"].to_numpy(), test["Z"].to_numpy()])
        missing = np.isnan(numeric)
        patterns = np.unique(np.packbits(missing, axis=1), axis=0, return_inverse=True)[1]
        ranks = QuantileTransformer(n_quantiles=1024, subsample=None).fit_transform(numeric)
        ranks = np.nan_to_num(ranks, nan=0.5)
        projected = PCA(n_components=12, random_state=7).fit_transform(ranks)
        features = (ranks, projected, missing, categories, patterns, labels)

        train_index = np.arange(len(train))
        test_index = np.arange(len(train), len(numeric))
        test_distance, test_proba = predict_neighbors(train_index, test_index, *features)

        for fold, (fit, val) in enumerate(validation_splits(labels, 5000)):
            val_distance, val_proba = predict_neighbors(fit, val, *features)
            path = DATA / f"knn15_block5000_{fold}.npz"
            np.savez(
                path, val_idx=val, val_proba=val_proba, test_proba=test_proba,
                val_distance=val_distance, test_distance=test_distance,
            )
            print(f"Wrote {path.name}", flush=True)


if __name__ == "__main__":
    main()
