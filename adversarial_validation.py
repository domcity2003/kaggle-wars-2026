"""Estimate validation weights from cross-fitted train-versus-test probabilities."""

import argparse

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from challenger import DATA, add_missing_features, load_data


def effective_sample_size(weights):
    return weights.sum() ** 2 / np.square(weights).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocked", action="store_true")
    parser.add_argument("--rounds", type=int, default=140)
    args = parser.parse_args()
    features, y = load_data(60)
    add_missing_features(features)
    domain = (np.arange(len(features)) >= len(y)).astype(int)
    probabilities = np.empty(len(features))
    density_ratios = np.empty(len(features))
    gain = np.zeros(features.shape[1])
    if args.blocked:
        train_groups = np.arange(len(y)) // 5000
        test_groups = train_groups[-1] + 1 + np.arange(len(features) - len(y)) // 2000
        groups = np.concatenate([train_groups, test_groups])
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=85)
        splits = splitter.split(features, domain, groups)
    else:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=85)
        splits = splitter.split(features, domain)
    for fold, (fit_idx, val_idx) in enumerate(splits):
        classifier = LGBMClassifier(
            n_estimators=args.rounds,
            num_leaves=63,
            min_child_samples=75,
            learning_rate=0.055,
            reg_lambda=2,
            colsample_bytree=0.8,
            n_jobs=1,
            max_bin=127,
            random_state=711 + fold,
            verbosity=-1,
            force_col_wise=True,
        )
        classifier.fit(features.iloc[fit_idx], domain[fit_idx])
        val_probability = classifier.predict_proba(features.iloc[val_idx])[:, 1]
        probabilities[val_idx] = val_probability
        train_rows, test_rows = np.bincount(domain[fit_idx])
        density_ratios[val_idx] = val_probability / np.clip(1 - val_probability, 1e-8, None)
        density_ratios[val_idx] *= train_rows / test_rows
        gain += classifier.booster_.feature_importance(importance_type="gain")
        print(f"fold={fold} auc={roc_auc_score(domain[val_idx], probabilities[val_idx]):.5f}", flush=True)

    train_probability = probabilities[: len(y)]
    test_probability = probabilities[len(y) :]
    odds = density_ratios[: len(y)]
    weights = np.minimum(odds, 10)
    weights /= weights.mean()
    name = "validation_weights_blocked.npz" if args.blocked else "validation_weights.npz"
    np.savez_compressed(DATA / name, weights=weights, odds=odds, probabilities=probabilities)
    auc = roc_auc_score(domain, probabilities)
    brier = brier_score_loss(domain, probabilities)
    print(f"AUC {auc:.5f}; Brier {brier:.5f}; weighted ESS {effective_sample_size(weights):.0f}/{len(y)}")
    quantiles = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print("Train probabilities", np.quantile(train_probability, quantiles).round(3))
    print("Test probabilities", np.quantile(test_probability, quantiles).round(3))
    counts = np.bincount(y, weights=weights, minlength=11)
    print("Weighted class share", (counts / counts.sum()).round(4))
    for label in range(11):
        print(f"K{label:02} ESS {effective_sample_size(weights[y == label]):.0f}/{np.count_nonzero(y == label)}")
    order = np.argsort(gain)[::-1][:15]
    print("Domain gain share", list(zip(features.columns[order], (gain[order] / gain.sum()).round(4))))
    aucs = {}
    for column in features.select_dtypes(include="number"):
        present = features[column].notna().to_numpy()
        auc = roc_auc_score(domain[present], features.loc[present, column])
        aucs[column] = max(auc, 1 - auc)
    print("Best individual rank AUC", sorted(aucs.items(), key=lambda entry: -entry[1])[:15])


if __name__ == "__main__":
    main()
