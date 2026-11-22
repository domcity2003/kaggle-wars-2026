"""Train a grouped LightGBM candidate on raw features and shared feature geometry."""

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer

from challenger import add_missing_features, load_data
from train_cat import DATA, validation_splits


def geometry(raw, indices):
    transformer = QuantileTransformer(
        n_quantiles=512, output_distribution="normal", random_state=42,
    )
    transformer.fit(raw[indices])
    rank = transformer.transform(raw).clip(-4.5, 4.5).astype(np.float32)
    correlation = pd.DataFrame(rank[indices]).corr(min_periods=500).to_numpy()
    residuals = {}
    pair_indices = np.triu_indices(raw.shape[1], 1)
    order = np.argsort(-np.abs(correlation[pair_indices]))
    for pair in order[:75]:
        left, right = pair_indices[0][pair], pair_indices[1][pair]
        r = correlation[left, right]
        if abs(r) >= 0.82:
            residuals[f"rank_{left}_{right}"] = rank[:, left] - np.sign(r) * rank[:, right]
    filled = np.nan_to_num(rank)
    pca = PCA(n_components=24, random_state=42, svd_solver="randomized")
    pca.fit(filled[indices])
    components = pca.transform(filled).astype(np.float32)
    return pd.concat(
        [pd.DataFrame(residuals), pd.DataFrame(components, columns=[f"pc{i}" for i in range(24)])],
        axis=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=220)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--class-weight", type=float, default=0.25)
    args = parser.parse_args()
    frame, labels = load_data(60)
    columns = [f"X{i:03d}" for i in range(60)]
    raw = frame[columns].to_numpy(dtype=np.float32)
    add_missing_features(frame)
    all_prior = np.bincount(labels) / len(labels)
    for fold, (fit, val) in enumerate(validation_splits(labels, 5000)):
        if fold not in args.folds:
            continue
        augmented = pd.concat([frame, geometry(raw, fit)], axis=1)
        train, test = augmented.iloc[:len(labels)], augmented.iloc[len(labels):]
        prior = np.bincount(labels[fit], minlength=11) / len(fit)
        row_weights = prior[labels[fit]] ** -args.class_weight
        row_weights /= row_weights.mean()
        model = lgb.LGBMClassifier(
            n_estimators=args.rounds, learning_rate=0.065, num_leaves=39,
            min_child_samples=45, reg_lambda=5, colsample_bytree=0.70,
            subsample=0.85, subsample_freq=1, n_jobs=args.threads,
            verbosity=-1, max_bin=127, random_state=900 + fold,
        )
        print(f"Fold {fold} fitting {augmented.shape[1]} features", flush=True)
        model.fit(train.iloc[fit], labels[fit], sample_weight=row_weights)
        for rounds in sorted({min(args.rounds, 100), args.rounds}):
            val_proba = model.predict_proba(train.iloc[val], num_iteration=rounds)
            test_proba = model.predict_proba(test, num_iteration=rounds)
            # Return an estimate of the original training posterior for ensembling.
            val_proba *= prior ** args.class_weight
            test_proba *= prior ** args.class_weight
            val_proba /= val_proba.sum(axis=1, keepdims=True)
            test_proba /= test_proba.sum(axis=1, keepdims=True)
            np.savez_compressed(
                DATA / f"geometry_{rounds}_fold{fold}.npz",
                val_idx=val, val_proba=val_proba, test_proba=test_proba,
            )
            f1 = [
                round(f1_score(labels[val], (val_proba / all_prior ** alpha).argmax(axis=1),
                               labels=np.arange(11), average="macro"), 5)
                for alpha in (0, 0.35, 0.70)
            ]
            print(f"Fold {fold} {rounds} macro F1 {f1}", flush=True)


if __name__ == "__main__":
    main()
