"""Evaluate separate LightGBM models for each categorical cohort."""

import argparse

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

from challenger import add_missing_features, load_data
from train_cat import DATA, validation_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--weight-power", type=float, default=0)
    parser.add_argument("--name", default="experts")
    args = parser.parse_args()
    frame, labels = load_data(60)
    add_missing_features(frame)
    cohort = frame.pop("Z").to_numpy()
    train, test = frame.iloc[:len(labels)], frame.iloc[len(labels):]
    train_group, test_group = cohort[:len(train)], cohort[len(train):]
    all_prior = np.bincount(labels) / len(labels)
    for fold, (fit, val) in enumerate(validation_splits(labels, 5000)):
        if fold not in args.folds:
            continue
        val_proba = np.zeros((len(val), 11), dtype=np.float32)
        test_proba = np.zeros((len(test), 11), dtype=np.float32)
        for group in np.unique(cohort):
            fit_group = fit[train_group[fit] == group]
            val_where = train_group[val] == group
            test_where = test_group == group
            classes, target = np.unique(labels[fit_group], return_inverse=True)
            if not len(classes):
                val_proba[val_where] = all_prior
                test_proba[test_where] = all_prior
                continue
            model = lgb.LGBMClassifier(
                n_estimators=args.rounds, learning_rate=0.075, num_leaves=31,
                min_child_samples=38, reg_lambda=4, colsample_bytree=0.82,
                subsample=0.9, subsample_freq=1, n_jobs=1, verbosity=-1,
                max_bin=127, random_state=500 + fold,
            )
            prior = np.bincount(target) / len(target)
            row_weights = prior[target] ** -args.weight_power
            row_weights /= row_weights.mean()
            if len(classes) > 1:
                model.fit(train.iloc[fit_group], target, sample_weight=row_weights)
            if val_where.any():
                val_proba[np.ix_(val_where, classes)] = (
                    model.predict_proba(train.iloc[val[val_where]], num_threads=1)
                    if len(classes) > 1 else prior
                )
            if test_where.any():
                test_proba[np.ix_(test_where, classes)] = (
                    model.predict_proba(test.iloc[test_where], num_threads=1)
                    if len(classes) > 1 else prior
                )
            print(f"Fold {fold} cohort {group}, {len(fit_group)} rows, {len(classes)} classes", flush=True)
            val_proba[np.ix_(val_where, classes)] *= prior ** args.weight_power
            test_proba[np.ix_(test_where, classes)] *= prior ** args.weight_power
        val_proba /= val_proba.sum(axis=1, keepdims=True)
        test_proba /= test_proba.sum(axis=1, keepdims=True)
        np.savez_compressed(
            DATA / f"{args.name}_{args.rounds}_fold{fold}.npz",
            val_idx=val, val_proba=val_proba, test_proba=test_proba,
        )
        for alpha in (0, 0.25, 0.50, 0.75):
            guess = (val_proba / all_prior ** alpha).argmax(axis=1)
            result = f1_score(labels[val], guess, labels=np.arange(11), average="macro")
            print(f"Fold {fold} alpha {alpha:.2f} macro F1 {result:.5f}", flush=True)


if __name__ == "__main__":
    main()
