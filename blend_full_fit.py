"""Mix test probabilities from full-data neural refits with the grouped ensemble."""

import argparse
from zipfile import ZipFile

import numpy as np
import pandas as pd

from build_alternatives import MODELS, POOLS, expected_macro_prediction
from evaluate import load_predictions
from train_cat import DATA, load_data


GROUPED_MODELS = [
    "mlp_pairs_12_fold{fold}.npz", "mlp_pairs_20_fold{fold}.npz",
    "experts_130_fold{fold}.npz", "mlp_block_16_fold{fold}.npz",
    "lgbmshift5000_fold{fold}_150.npz", "geometry_160_fold{fold}.npz",
    "residual_distill_20_fold{fold}.npz",
]
GROUPED_WEIGHTS = [1, 1, 2, 1, 1, 1, 1.5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-models", nargs="+", required=True)
    parser.add_argument("--full-weight", type=float, default=0.50)
    parser.add_argument("--alpha", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--neighborhood-macro", action="store_true")
    mode.add_argument("--hybrid-macro", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.full_weight <= 1:
        parser.error("Full-model weight must be between zero and one")
    train, test, labels = load_data()
    macro = args.neighborhood_macro or args.hybrid_macro
    if macro:
        recipe = POOLS["neighborhood"].copy()
        if args.hybrid_macro:
            for key in recipe:
                recipe[key] *= 2 / sum(POOLS["neighborhood"].values())
            reference = POOLS["reference"]
            for key, weight in reference.items():
                recipe[key] = recipe.get(key, 0) + weight / sum(reference.values())
        templates, weights = [MODELS[key] for key in recipe], list(recipe.values())
    else:
        templates, weights = GROUPED_MODELS, GROUPED_WEIGHTS
    _, _, _, grouped_test = load_predictions(templates, len(train), len(test), [0, 1, 2])
    grouped = np.average(grouped_test, weights=weights, axis=0)
    full = []
    for name in args.full_models:
        with np.load(DATA / name) as archive:
            probabilities = archive["test_proba"]
        if (probabilities.shape != (len(test), 11) or not np.isfinite(probabilities).all() or
                (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
            raise ValueError(f"Invalid test probabilities in {name}")
        full.append(probabilities)
    pooled = (1 - args.full_weight) * grouped + args.full_weight * np.mean(full, axis=0)
    prior = np.bincount(labels) / len(labels)
    if macro:
        alpha = args.alpha if args.alpha is not None else (0.10 if args.hybrid_macro else 0)
        original = grouped / prior ** alpha
        original /= original.sum(axis=1, keepdims=True)
        pooled /= prior ** alpha
        pooled /= pooled.sum(axis=1, keepdims=True)
        baseline = expected_macro_prediction(original, 0.18)
        predictions = expected_macro_prediction(pooled, 0.18)
    else:
        baseline = (grouped / prior ** 0.35).argmax(axis=1)
        predictions = (pooled / prior ** (args.alpha if args.alpha is not None else 0.35)).argmax(axis=1)
    with ZipFile(DATA / "kagglewars.zip") as archive:
        with archive.open("sample_submission (4).csv") as file:
            sample = pd.read_csv(file)
    if not sample["id"].equals(test["id"]):
        raise ValueError("Sample IDs differ from official test")
    sample["target"] = predictions
    sample.to_csv(DATA / args.output, index=False)
    print(f"Changed {(baseline != predictions).sum():,} labels versus the unrefitted baseline")
    print(sample["target"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
