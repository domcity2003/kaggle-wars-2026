"""Train a classifier on label prototypes from independent source blocks."""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from threadpoolctl import threadpool_limits

from challenger import LABELS, add_missing_features, load_data, report_metric
from train_cat import DATA, validation_splits


BLOCK_SIZE = 5000


def prototypes(rank, labels, cohort, block, fit, group):
    selected = fit[cohort[fit] == group]
    means, variances, known, classes, blocks = [], [], [], [], []
    for source in np.unique(block[selected]):
        source_rows = selected[block[selected] == source]
        for label in np.unique(labels[source_rows]):
            rows = source_rows[labels[source_rows] == label]
            if len(rows) < 12:
                continue
            values = rank[rows]
            finite = np.isfinite(values)
            count = finite.sum(axis=0)
            center = np.nan_to_num(values).sum(axis=0) / np.maximum(count, 1)
            delta = np.where(finite, values - center, 0)
            variance = np.square(delta).sum(axis=0) / np.maximum(count, 1)
            means.append(center)
            variances.append(variance)
            known.append(count >= max(8, len(rows) // 5))
            classes.append(label)
            blocks.append(source)
    return (np.asarray(means, dtype=np.float32), np.asarray(variances, dtype=np.float32),
            np.asarray(known, dtype=np.float32), np.asarray(classes), np.asarray(blocks))


def target_context(rank, labels, cohorts, fit):
    n_train = len(labels)
    block = np.arange(len(rank)) // BLOCK_SIZE
    block[n_train:] = -1
    fit_rows = np.zeros(len(rank), dtype=bool)
    fit_rows[fit] = True
    features = np.full((len(rank), len(LABELS), 4), np.nan, dtype=np.float32)
    for cohort in np.unique(cohorts):
        mean, variance, known, template_labels, template_groups = prototypes(
            rank, labels, cohorts, block, fit, cohort
        )
        if not len(template_labels):
            continue
        inverse_variance = known / (0.24 + 0.76 * variance)
        log_variance = np.log(0.24 + 0.76 * variance) * known
        row_indices = np.flatnonzero(cohorts == cohort)
        for start in range(0, len(row_indices), 4096):
            rows = row_indices[start : start + 4096]
            has_value = np.isfinite(rank[rows]).astype(np.float32)
            filled = np.nan_to_num(rank[rows])
            squared = np.square(filled)
            common = has_value @ known.T
            inv_common = 1 / np.maximum(common, 1)
            plain = squared @ known.T - 2 * filled @ (mean * known).T
            plain += has_value @ (np.square(mean) * known).T
            scaled = squared @ inverse_variance.T - 2 * filled @ (mean * inverse_variance).T
            scaled += has_value @ (np.square(mean) * inverse_variance).T
            incompatible = (common < 8) | (fit_rows[rows, None] & (block[rows, None] == template_groups))
            plain = np.maximum(plain * inv_common, 0)
            scaled = np.maximum(scaled * inv_common, 0)
            likelihood = scaled + (has_value @ log_variance.T) * inv_common
            plain[incompatible] = np.inf
            scaled[incompatible] = np.inf
            likelihood[incompatible] = np.inf
            for label in np.unique(template_labels):
                matches = template_labels == label
                ranked = np.sort(scaled[:, matches], axis=1)
                first = ranked[:, 0]
                second = ranked[:, min(1, ranked.shape[1] - 1)]
                features[rows, label] = np.column_stack([
                    plain[:, matches].min(axis=1), first,
                    np.where(np.isfinite(second), second, first), likelihood[:, matches].min(axis=1),
                ])
    features[~np.isfinite(features)] = np.nan
    return pd.DataFrame(features.reshape(len(rank), -1), columns=[
        f"context_{label:02d}_{metric}" for label in LABELS
        for metric in ("distance", "scaled", "second", "likelihood")
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rounds", type=int, default=90)
    parser.add_argument("--context-only", action="store_true")
    args = parser.parse_args()

    threadpool_limits(1)
    features, labels = load_data(60)
    numeric = features.select_dtypes(include="number").to_numpy(dtype=np.float32)
    cohorts = features["Z"].to_numpy()
    add_missing_features(features)
    if args.context_only:
        features = features.drop(columns=features.columns[features.columns.str.startswith("X")])
    matches = np.load(DATA / "validation_matched.npz")["weights"]
    distances = np.load(DATA / "distance_block5000.npz")["train_weight"]
    variant = "context_only" if args.context_only else "context"
    for fold, (fit, val) in enumerate(validation_splits(labels, BLOCK_SIZE)):
        if fold not in args.folds:
            continue
        transformer = QuantileTransformer(n_quantiles=512, output_distribution="normal", random_state=87)
        transformer.fit(numeric[fit])
        rank = transformer.transform(numeric).clip(-4, 4).astype(np.float32)
        evidence = target_context(rank, labels, cohorts, fit)
        augmented = pd.concat([features, evidence], axis=1)
        prior = np.bincount(labels[fit], minlength=len(LABELS)) / len(fit)
        print(f"Fold {fold} fitting {augmented.shape[1]} columns", flush=True)
        model = lgb.LGBMClassifier(
            n_estimators=args.rounds, learning_rate=0.075, num_leaves=31, max_depth=7,
            min_child_samples=65, reg_lambda=5, colsample_bytree=0.75, n_jobs=1,
            subsample=0.9, subsample_freq=1, max_bin=95, verbosity=-1, random_state=641 + fold,
        )
        model.fit(augmented.iloc[fit], labels[fit])
        v = model.predict_proba(augmented.iloc[val])
        t = model.predict_proba(augmented.iloc[len(labels) :])
        destination = DATA / f"{variant}_{args.rounds}_fold{fold}.npz"
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, val_idx=val, val_proba=v.astype(np.float32), test_proba=t.astype(np.float32))
        temporary.replace(destination)
        raw, alpha, tuned = report_metric(labels[val], v, prior)
        prediction = (v / prior**alpha).argmax(axis=1)
        matched = f1_score(labels[val], prediction, labels=LABELS, average="macro", sample_weight=matches[val])
        distance = f1_score(labels[val], prediction, labels=LABELS, average="macro", sample_weight=distances[val])
        splits = model.booster_.feature_importance(importance_type="gain")
        context_gain = splits[-evidence.shape[1] :].sum() / splits.sum()
        print(f"fold={fold} argmax={raw:.5f} tuned={tuned:.5f} alpha={alpha:.2f} "
              f"matched={matched:.5f} distance={distance:.5f} context gain={context_gain:.1%}", flush=True)


if __name__ == "__main__":
    main()
