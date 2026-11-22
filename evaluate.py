"""Cross-fit macro-F1 decision rules and optionally build the local submission."""

import argparse
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from train_cat import DATA, load_data


N_CLASSES = 11


def confusion_counts(labels, predictions, weights=None):
    cells = N_CLASSES * labels + predictions
    return np.bincount(cells, weights=weights, minlength=N_CLASSES ** 2).reshape(
        N_CLASSES, N_CLASSES
    )


def score(counts):
    denominator = counts.sum(axis=-1) + counts.sum(axis=-2)
    return (2 * np.diagonal(counts, axis1=-2, axis2=-1) /
            np.maximum(denominator, 1e-15)).mean(axis=-1)


def tune(logp, labels, weights, offsets):
    def objective(bias):
        prediction = (logp + bias).argmax(axis=1)
        uniform = score(confusion_counts(labels, prediction))
        shifted = score(confusion_counts(labels, prediction, weights))
        return 0.25 * uniform + 0.75 * shifted if weights is not None else uniform

    best = max(offsets, key=objective).copy()
    best_score = objective(best)
    for step in (0.45, 0.22, 0.10):
        for cls in range(N_CLASSES):
            for direction in (-1, 1):
                candidate = best.copy()
                candidate[cls] += direction * step
                candidate_score = objective(candidate)
                if candidate_score > best_score:
                    best, best_score = candidate, candidate_score
    return best


def load_predictions(models, n_train, n_test, folds):
    out_of_fold = []
    test_models = []
    assignment = np.full(n_train, -1)
    for model in models:
        probabilities = np.full((n_train, N_CLASSES), np.nan, dtype=np.float32)
        test_predictions = []
        for fold in folds:
            with np.load(DATA / model.format(fold=fold)) as archive:
                indices = archive["val_idx"]
                val_proba, test_proba = archive["val_proba"], archive["test_proba"]
            if (val_proba.shape != (len(indices), N_CLASSES) or
                    test_proba.shape != (n_test, N_CLASSES) or
                    not np.isfinite(val_proba).all() or not np.isfinite(test_proba).all() or
                    not np.allclose(val_proba.sum(axis=1), 1, atol=1e-4) or
                    not np.allclose(test_proba.sum(axis=1), 1, atol=1e-4)):
                raise ValueError(f"Invalid probabilities for {model} fold {fold}")
            if ((assignment[indices] != -1) & (assignment[indices] != fold)).any():
                raise ValueError(f"Inconsistent validation split for {model} fold {fold}")
            assignment[indices] = fold
            probabilities[indices] = val_proba
            test_predictions.append(test_proba)
        out_of_fold.append(probabilities)
        test_models.append(np.mean(test_predictions, axis=0))
    active = assignment >= 0
    if any(not np.isfinite(model[active]).all() for model in out_of_fold):
        raise ValueError("Models do not share every validation row")
    return active, assignment[active], np.stack(out_of_fold)[:, active], np.stack(test_models)


def report(name, labels, predicted, weights, indices, block_size):
    counts = confusion_counts(labels, predicted, weights)
    groups = indices // block_size
    group_counts = np.stack([
        confusion_counts(labels[groups == block], predicted[groups == block],
                         None if weights is None else weights[groups == block])
        for block in np.unique(groups)
    ])
    random = np.random.default_rng(2026)
    n_groups = len(group_counts)
    bootstrap = random.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=600)
    intervals = score((bootstrap @ group_counts.reshape(n_groups, -1)).reshape(-1, 11, 11))
    cls_f1 = 2 * counts.diagonal() / np.maximum(counts.sum(axis=0) + counts.sum(axis=1), 1e-15)
    low, high = np.quantile(intervals, [0.025, 0.975])
    print(f"{name:>18} {score(counts):.5f} [{low:.5f}, {high:.5f}] class F1 "
          f"{np.round(cls_f1, 3).tolist()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--blend", type=float, nargs="+")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output")
    parser.add_argument("--weights", default="validation_weights.npz")
    parser.add_argument("--weight-key", default="weights")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--bootstrap-block-size", type=int, default=5000)
    parser.add_argument("--geometric", action="store_true")
    args = parser.parse_args()
    if args.output and len(set(args.folds)) < 3:
        parser.error("A submission requires results from all three validation folds")

    train, test, labels = load_data()
    active, fold_ids, validation, test_probas = load_predictions(
        args.models, len(train), len(test), args.folds
    )
    blend = np.ones(len(args.models)) if args.blend is None else np.asarray(args.blend)
    if len(blend) != len(args.models) or (blend < 0).any() or blend.sum() <= 0:
        parser.error("Provide a nonnegative weight for each model and a positive total")
    if args.geometric:
        logp = np.average(np.log(np.maximum(validation, 1e-10)), weights=blend, axis=0)
        log_test = np.average(np.log(np.maximum(test_probas, 1e-10)), weights=blend, axis=0)
        logp -= logsumexp(logp, axis=1, keepdims=True)
        log_test -= logsumexp(log_test, axis=1, keepdims=True)
    else:
        logp = np.log(np.maximum(np.average(validation, weights=blend, axis=0), 1e-10))
        log_test = np.log(np.maximum(np.average(test_probas, weights=blend, axis=0), 1e-10))
    y = labels[active]
    all_prior = np.bincount(labels, minlength=N_CLASSES) / len(labels)
    alphas = (0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.95, 1.20)
    prior_offsets = [-alpha * np.log(all_prior) for alpha in alphas]
    weight_path = DATA / args.weights
    weights = None
    if weight_path.exists():
        with np.load(weight_path) as archive:
            if "fold" in archive and not np.array_equal(archive["fold"][active], fold_ids):
                raise ValueError("Validation weights use a different split")
            weights = archive[args.weight_key][active]
    if weights is not None:
        print(f"Shift-weight effective rows {(weights.sum() ** 2 / (weights @ weights)):.0f}"
              f" of {len(y)}")
    print("alpha  unweighted  weighted")
    for alpha, offset in zip(alphas, prior_offsets):
        guess = (logp + offset).argmax(axis=1)
        print(f"{alpha:5.2f}  {score(confusion_counts(y, guess)):.5f}  "
              f"{score(confusion_counts(y, guess, weights)):.5f}")
    if args.alpha is None and len(args.folds) == 1:
        return

    prediction = np.empty(len(y), dtype=np.int64)
    for fold in args.folds:
        val = fold_ids == fold
        if args.alpha is not None:
            bias = -args.alpha * np.log(all_prior)
        else:
            bias = tune(logp[~val], y[~val], None if weights is None else weights[~val],
                        prior_offsets)
        prediction[val] = (logp[val] + bias).argmax(axis=1)
        print(f"Fold {fold}: {score(confusion_counts(y[val], prediction[val])):.5f} "
              f"weighted {score(confusion_counts(y[val], prediction[val], None if weights is None else weights[val])):.5f}")
    indices = np.flatnonzero(active)
    report("Decision rule", y, prediction, None, indices, args.bootstrap_block_size)
    if weights is not None:
        report("Shift-weighted", y, prediction, weights, indices, args.bootstrap_block_size)

    if args.output:
        with ZipFile(DATA / "kagglewars.zip") as archive:
            sample = pd.read_csv(archive.open("sample_submission (4).csv"))
        if not sample["id"].equals(test["id"]) or list(sample.columns) != ["id", "target"]:
            raise ValueError("The test IDs or required columns differ from the sample")
        if args.alpha is not None:
            final_bias = -args.alpha * np.log(all_prior)
        else:
            final_bias = tune(logp, y, weights, prior_offsets)
        sample["target"] = (log_test + final_bias).argmax(axis=1)
        if sample["target"].nunique() != N_CLASSES:
            raise ValueError("Predictions omit classes")
        output = DATA / args.output
        sample.to_csv(output, index=False)
        print(f"Wrote {output}, bias {np.round(final_bias, 4)}")
        print(sample["target"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
