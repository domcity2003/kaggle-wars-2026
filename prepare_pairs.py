"""Identify reciprocal near pairs in each labeled training block."""

import argparse

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from train_cat import DATA, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="paired_diagnostic.npz")
    parser.add_argument("--summary-only", action="store_true", help="Print diagnostics without saving the cache.")
    args = parser.parse_args()
    train, test, labels = load_data()
    columns = [f"X{i:03}" for i in range(60)]
    numeric = np.concatenate([
        train[columns].to_numpy(dtype=np.float32), test[columns].to_numpy(dtype=np.float32)
    ])
    rows = np.arange(len(train))
    missing = np.isnan(numeric)
    masks = np.unique(np.packbits(missing[:len(train)], axis=1), axis=0, return_inverse=True)[1]
    categories = np.unique(train["Z"], return_inverse=True)[1]
    _, groups = np.unique(
        np.column_stack([rows // 5000, labels, categories, masks]), axis=0, return_inverse=True
    )
    ranks = np.full(numeric.shape, 0.5, dtype=np.float32)
    for column in range(numeric.shape[1]):
        observed = ~missing[:, column]
        ranks[observed, column] = rankdata(numeric[observed, column]) / observed.sum()
    neighbors = np.full(len(train), -1)
    distances = np.full((len(train), 2), np.nan)
    for group in np.unique(groups):
        selected = rows[groups == group]
        if len(selected) < 2:
            continue
        observed = ~missing[selected[0]]
        values = ranks[selected][:, observed]
        distance, local = cKDTree(values).query(values, k=min(3, len(selected)), workers=1)
        neighbors[selected] = selected[local[:, 1]]
        distances[selected, :local.shape[1] - 1] = distance[:, 1:] / np.sqrt(observed.sum())
    mutual = (neighbors >= 0) & (neighbors != rows) & (neighbors[neighbors] == rows)
    print(f"Reciprocal pairs account for {mutual.mean():.2%} of training rows")
    print(f"Median reciprocal distance {np.median(distances[mutual, 0]):.6f}")
    if not args.summary_only:
        np.savez(DATA / args.output, neighbor=neighbors, dist=distances, group=groups, mutual=mutual)


if __name__ == "__main__":
    main()
