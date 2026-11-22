"""Match grouped validation to test on category, missingness, and neighbor distance."""

import numpy as np
import pandas as pd

from train_cat import DATA, load_data


def cohort_name(frame):
    numeric = frame[[f"X{i:03d}" for i in range(60)]]
    packed = np.packbits(numeric.isna().to_numpy(), axis=1)
    return np.array([f"{z}/{mask.tobytes().hex()}" for z, mask in zip(frame["Z"], packed)])


def main():
    train, test, labels = load_data()
    with np.load(DATA / "distance_block5000.npz") as data:
        train_distance = data["train_distance"]
        test_distance = data["test_distance"]
        fold = data["fold"]
    train_cohort, test_cohort = cohort_name(train), cohort_name(test)
    bins = np.arange(0.035, 0.235, 0.02)
    train_bin = np.digitize(train_distance, bins)
    test_bin = np.digitize(test_distance, bins)
    train_cells = pd.DataFrame({"cohort": train_cohort, "distance": train_bin})
    test_cells = pd.DataFrame({"cohort": test_cohort, "distance": test_bin})
    joint_train = train_cells.value_counts(sort=False)
    joint_test = test_cells.value_counts(sort=False)
    cells = joint_train.index.union(joint_test.index)
    source = joint_train.reindex(cells, fill_value=0).to_numpy()
    target = joint_test.reindex(cells, fill_value=0).to_numpy()
    raw = ((target + 12) / len(test)) / ((source + 30) / len(train))
    lookup = pd.Series(raw.clip(0.03, 8), index=cells)
    weights = lookup.reindex(pd.MultiIndex.from_frame(train_cells)).to_numpy(copy=True)
    weights /= weights.mean()
    ess = weights.sum() ** 2 / (weights @ weights)
    print(f"Effective validation size {ess:,.0f} of {len(train):,}")
    print(f"Test rows with >=30 validation neighbors in cell {target[source >= 30].sum() / len(test):.2%}")
    print(f"Unsupported cells {target[source == 0].sum() / len(test):.2%} of test")
    print("Weighted class frequencies", np.bincount(labels, weights=weights).round(0).astype(int))
    np.savez_compressed(DATA / "validation_matched.npz", weights=weights, fold=fold)


if __name__ == "__main__":
    main()
