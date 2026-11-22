"""Smooth predictions using a separate unlabeled graph in each holdout and test."""

import argparse

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

from train_cat import DATA, load_data


def geometry(train, test):
    columns = [f"X{i:03d}" for i in range(60)]
    numeric = np.concatenate([frame[columns].to_numpy(dtype=np.float32) for frame in (train, test)])
    category = np.concatenate([frame["Z"].to_numpy() for frame in (train, test)])
    missing = np.isnan(numeric)
    _, pattern = np.unique(np.packbits(missing, axis=1), axis=0, return_inverse=True)
    ranks = QuantileTransformer(n_quantiles=1024, subsample=None).fit_transform(numeric)
    ranks = np.nan_to_num(ranks, nan=0.5)
    projected = PCA(n_components=12, random_state=7).fit_transform(ranks)
    return projected, ranks, missing, pattern, category


def make_graph(rows, features, neighbors, category_factor):
    projected, ranks, missing, pattern, category = features
    points = projected[rows]
    _, shortlist = cKDTree(points).query(points, k=min(113, len(rows)), eps=0.28, workers=2)
    best = np.empty((len(rows), neighbors), dtype=np.int32)
    distances = np.empty(best.shape, dtype=np.float32)
    for start in range(0, len(rows), 600):
        section = slice(start, start + 600)
        row = rows[section]
        candidate = rows[shortlist[section]]
        shared = ~(missing[row, None] | missing[candidate])
        delta = ranks[row, None] - ranks[candidate]
        distance = np.sqrt((delta**2 * shared).sum(axis=2) / np.maximum(shared.sum(axis=2), 1))
        different_mask = pattern[row, None] != pattern[candidate]
        closest = np.argsort(distance, axis=1)[:, 1:3]
        gap = np.take_along_axis(distance, closest, axis=1)
        in_section = np.arange(len(row))
        first = closest[:, 0]
        likely_duplicate = (
            (gap[:, 0] < 0.035)
            & (2 * gap[:, 0] < gap[:, 1])
            & ~different_mask[in_section, first]
            & (category[row] == category[candidate[in_section, first]])
        )
        penalized = np.sqrt(distance**2 + 0.002 * different_mask)
        # Pairs in the labeled archive should not make validation propagation easier.
        penalized[distance < 0.01] = np.inf
        penalized[in_section[likely_duplicate], first[likely_duplicate]] = np.inf
        chosen = np.argsort(penalized, axis=1)[:, :neighbors]
        best[section] = np.take_along_axis(shortlist[section], chosen, axis=1)
        distances[section] = np.take_along_axis(penalized, chosen, axis=1)

    bandwidth = np.maximum(np.max(np.where(np.isfinite(distances), distances, 0), axis=1), 0.02)
    weight = np.exp(-2 * (distances / bandwidth[:, None]) ** 2)
    weight *= np.where(category[rows, None] == category[rows[best]], 1, category_factor)
    source = np.repeat(np.arange(len(rows)), neighbors)
    matrix = csr_matrix((weight.ravel(), (source, best.ravel())), shape=(len(rows), len(rows)))
    matrix = matrix.minimum(matrix.T)
    total = np.asarray(matrix.sum(axis=1)).ravel()
    matrix.setdiag(total == 0)
    total[total == 0] = 1
    return matrix.multiply((1 / total)[:, None]).tocsr()


def smooth(probabilities, graph, mix, steps, sharpness):
    result = probabilities
    for _ in range(steps):
        messages = result ** sharpness
        messages /= messages.sum(axis=1, keepdims=True)
        result = (1 - mix) * probabilities + mix * (graph @ messages)
    return result.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Names in data/, with {fold} placeholders")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--name", default="graph")
    parser.add_argument("--neighbors", type=int, default=32, choices=range(3, 65), metavar="3-64")
    parser.add_argument("--category-factor", type=float, default=0.5)
    parser.add_argument("--mix", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--sharpness", type=float, default=1.6)
    args = parser.parse_args()
    if not (0 <= args.mix <= 1 and 0 <= args.category_factor <= 1 and args.sharpness > 0 and args.steps >= 1):
        parser.error("Use valid mixing and category factors in [0,1], positive sharpness and at least one step")

    with threadpool_limits(2):
        train, test, _ = load_data()
        features = geometry(train, test)
        target_rows = np.arange(len(train), len(train) + len(test))
        target_graph = make_graph(target_rows, features, args.neighbors, args.category_factor)
        for fold in args.folds:
            paths = [DATA / model.format(fold=fold) for model in args.models]
            with np.load(paths[0]) as first:
                val_index = first["val_idx"]
            validation_graph = make_graph(val_index, features, args.neighbors, args.category_factor)
            for path in paths:
                with np.load(path) as prediction:
                    if not np.array_equal(val_index, prediction["val_idx"]):
                        raise ValueError(f"Validation indices do not match {path.name}")
                    val = smooth(prediction["val_proba"], validation_graph, args.mix, args.steps, args.sharpness)
                    target = smooth(prediction["test_proba"], target_graph, args.mix, args.steps, args.sharpness)
                dest = DATA / f"{args.name}_{path.name}"
                np.savez_compressed(dest, val_idx=val_index, val_proba=val, test_proba=target)
                print(dest.name, flush=True)


if __name__ == "__main__":
    main()
