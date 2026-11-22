"""Adapt fold probabilities to prior shifts within observed missingness patterns."""

import argparse

import numpy as np

from train_cat import DATA, load_data, validation_splits


FEATURES = [f"X{i:03d}" for i in range(60)]
N_CLASSES = 11


def mask_priors(labels, fit, groups):
    cohort = groups[fit]
    counts = np.bincount(cohort * N_CLASSES + labels[fit], minlength=(groups.max() + 1) * N_CLASSES)
    counts = counts.reshape(-1, N_CLASSES)
    pooled = (counts.sum(axis=0) + 0.1) / (len(fit) + 0.1 * N_CLASSES)
    prior = (counts + 10 * pooled) / (counts.sum(axis=1, keepdims=True) + 10)
    return counts.sum(axis=1), prior


def normalize(probabilities):
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def adapt(probabilities, cohort, counts, prior, shrinkage, strength):
    result = probabilities.copy()
    for group in np.unique(cohort):
        rows = cohort == group
        if counts[group] < 75 or rows.sum() < 40:
            continue
        source = prior[group]
        estimate = source.copy()
        sample = probabilities[rows]
        for _ in range(12):
            ratio = np.clip(estimate / source, 1 / 3, 3)
            posterior = normalize(sample * ratio)
            estimate = shrinkage * source + (1 - shrinkage) * posterior.mean(axis=0)
        ratio = np.clip(estimate / source, 1 / 3, 3)
        result[rows] = normalize(sample * ratio**strength)
    return result.astype(np.float32)


def load_average(templates, blend, fold, val, n_test):
    validation = []
    test = []
    for template in templates:
        path = DATA / template.format(fold=fold)
        with np.load(path) as archive:
            if not np.array_equal(archive["val_idx"], val):
                raise ValueError(f"Expected the 5,000-row split in {path.name}")
            validation.append(archive["val_proba"])
            test.append(archive["test_proba"])
    for bucket, size in ((validation, len(val)), (test, n_test)):
        for p in bucket:
            if (p.shape != (size, N_CLASSES) or not np.isfinite(p).all() or
                    (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-4)):
                raise ValueError(f"Invalid fold {fold} input probabilities")
    return np.average(validation, axis=0, weights=blend), np.average(test, axis=0, weights=blend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, nargs="+", help="Names in data/, with {fold} placeholders")
    parser.add_argument("--blend", type=float, nargs="+")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--name", default="mask_em")
    parser.add_argument("--shrinkage", type=float, default=0.55)
    parser.add_argument("--strength", type=float, default=0.4)
    args = parser.parse_args()
    blend = np.ones(len(args.models)) if args.blend is None else np.asarray(args.blend)
    if (len(blend) != len(args.models) or not np.isfinite(blend).all() or
            (blend < 0).any() or blend.sum() <= 0):
        parser.error("Provide a nonnegative blend weight for each model and a positive total")
    if not (0 <= args.shrinkage <= 1 and 0 <= args.strength <= 1):
        parser.error("Shrinkage and strength must be between zero and one")

    train, test, labels = load_data()
    masks = np.concatenate([frame[FEATURES].isna().to_numpy() for frame in (train, test)])
    _, cohorts = np.unique(np.packbits(masks, axis=1), axis=0, return_inverse=True)
    test_groups = cohorts[len(train):]
    for fold, (fit, val) in enumerate(validation_splits(labels, 5000)):
        if fold not in args.folds:
            continue
        counts, prior = mask_priors(labels, fit, cohorts)
        validation, target = load_average(args.models, blend, fold, val, len(test))
        validation = adapt(validation, cohorts[val], counts, prior, args.shrinkage, args.strength)
        target = adapt(target, test_groups, counts, prior, args.shrinkage, args.strength)
        output = DATA / f"{args.name}_fold{fold}.npz"
        np.savez_compressed(output, val_idx=val, val_proba=validation, test_proba=target)
        print(output.name, flush=True)


if __name__ == "__main__":
    main()
