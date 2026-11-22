"""Choose categorical predictions from model probabilities for expected macro F1."""

import argparse
from zipfile import ZipFile

import numpy as np
import pandas as pd

from evaluate import DATA, load_predictions, report
from train_cat import load_data


def estimated_counts(probabilities, weights, decision):
    row = np.arange(len(decision))
    assigned = np.bincount(decision, weights=weights, minlength=11)
    correct = np.bincount(decision, weights=weights * probabilities[row, decision], minlength=11)
    return correct, assigned


def expected_f1_decision(probabilities, weights, temperature, iterations):
    p = probabilities ** (1 / temperature)
    p /= p.sum(axis=1, keepdims=True)
    prevalence = (weights[:, None] * p).sum(axis=0)
    current = p.argmax(axis=1)
    row = np.arange(len(p))
    for _ in range(iterations):
        correct, assigned = estimated_counts(p, weights, current)
        denominator = assigned + prevalence
        f1_half = correct / np.maximum(denominator, 1e-12)

        # Exact change to the expected confusion counts from switching one row.
        add = weights[:, None] * (p - f1_half) / (denominator + weights[:, None])
        remove = weights * (f1_half[current] - p[row, current])
        remove /= np.maximum(denominator[current] - weights, 1e-12)
        improvement = add + remove[:, None]
        improvement[row, current] = 0
        option = improvement.argmax(axis=1)
        gain = improvement[row, option]
        promising = np.flatnonzero(gain > 1e-12)
        if not len(promising):
            break
        ranked = promising[np.argsort(-gain[promising], kind="stable")]
        for fraction in (0.45, 0.20, 0.08, 0.03):
            changed = ranked[:max(1, round(len(ranked) * fraction))]
            alternative = current.copy()
            alternative[changed] = option[changed]
            correct, assigned = estimated_counts(p, weights, alternative)
            updated = np.mean(correct / np.maximum(assigned + prevalence, 1e-12))
            if updated > f1_half.mean() + 1e-7:
                current = alternative
                break
        else:
            break
    return current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--blend", type=float, nargs="+", required=True)
    parser.add_argument("--temperature", type=float, default=0.70)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compare")
    args = parser.parse_args()
    blend = np.asarray(args.blend)
    if (len(blend) != len(args.models) or not np.isfinite(blend).all() or
            (blend < 0).any() or blend.sum() <= 0):
        parser.error("Provide a nonnegative weight per model and a positive total")
    if not (np.isfinite(args.temperature) and args.temperature > 0 and args.iterations > 0):
        parser.error("Use a positive temperature and iteration count")

    train, test, labels = load_data()
    active, folds, validation, target = load_predictions(args.models, len(train), len(test), [0, 1, 2])
    if not active.all():
        raise ValueError("Expected every training row in a grouped validation fold")
    validation = np.average(validation, axis=0, weights=blend)
    target = np.average(target, axis=0, weights=blend)
    with np.load(DATA / "validation_matched.npz") as archive:
        matched = archive["weights"]
        if not np.array_equal(archive["fold"], folds):
            raise ValueError("The matched weight and prediction folds differ")
    with np.load(DATA / "distance_block5000.npz") as archive:
        distance = archive["train_weight"]
        if not np.array_equal(archive["fold"], folds):
            raise ValueError("The distance weight and prediction folds differ")

    decisions = {}
    for name, weights in (("uniform", np.ones(len(train))), ("matched", matched), ("distance", distance)):
        predictions = np.empty(len(train), dtype=int)
        for fold in (0, 1, 2):
            rows = folds == fold
            predictions[rows] = expected_f1_decision(
                validation[rows], weights[rows], args.temperature, args.iterations,
            )
        decisions[name] = predictions
    decisions["base"] = validation.argmax(axis=1)
    for name, prediction in decisions.items():
        index = np.arange(len(train))
        report(f"{name}, raw", labels, prediction, None, index, 5000)
        report(f"{name}, matched", labels, prediction, matched, index, 5000)
        report(f"{name}, distance", labels, prediction, distance, index, 5000)

    prediction = expected_f1_decision(target, np.ones(len(test)), args.temperature, args.iterations)
    with ZipFile(DATA / "kagglewars.zip") as archive:
        sample = pd.read_csv(archive.open("sample_submission (4).csv"))
    if list(sample.columns) != ["id", "target"] or not sample["id"].equals(test["id"]):
        raise ValueError("The submission template and test IDs differ")
    if not np.array_equal(np.unique(prediction), np.arange(11)):
        raise ValueError("The candidate omits a class")
    sample["target"] = prediction
    if args.compare:
        previous = pd.read_csv(DATA / args.compare)
        if not previous["id"].equals(sample["id"]):
            raise ValueError("Comparison IDs differ from the submission template")
        changed = int(np.count_nonzero(previous["target"].to_numpy() != prediction))
        print(f"Changed {changed:,} rows ({changed / len(test):.2%}) versus {args.compare}")
    output = DATA / args.output
    sample.to_csv(output, index=False)
    print(output)
    print("Class counts", np.bincount(prediction, minlength=11).tolist())


if __name__ == "__main__":
    main()
