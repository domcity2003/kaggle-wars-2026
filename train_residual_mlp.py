"""Fit an MLP to measured features and residuals between related measurements."""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from train_cat import DATA, load_data, validation_splits
from train_mlp import ResidualBlock, make_features


TEACHERS = (
    ("mlp_pairs_12_fold{fold}.npz", 1),
    ("mlp_pairs_20_fold{fold}.npz", 1),
    ("experts_130_fold{fold}.npz", 2),
    ("mlp_block_16_fold{fold}.npz", 1),
)


def teacher_probabilities(fold, val, n_test):
    probabilities = np.zeros((len(val) + n_test, 11), dtype=np.float32)
    for pattern, weight in TEACHERS:
        with np.load(DATA / pattern.format(fold=fold)) as archive:
            if not np.array_equal(archive["val_idx"], val):
                raise ValueError("Teacher predictions use different holdout rows")
            probabilities[:len(val)] += weight * archive["val_proba"]
            probabilities[len(val):] += weight * archive["test_proba"]
    probabilities /= sum(weight for _, weight in TEACHERS)
    confidence = probabilities.max(axis=1)
    return torch.from_numpy(probabilities), np.flatnonzero(confidence >= 0.62)


def robust_residuals(values, fit):
    correlation = pd.DataFrame(values[fit]).corr(method="spearman", min_periods=1500)
    left, right = np.triu_indices(values.shape[1], k=1)
    strength = correlation.to_numpy()[left, right]
    selected = np.where(np.abs(strength) > 0.88)[0]
    selected = selected[np.argsort(-np.abs(strength[selected]))[:120]]
    features = []
    for pair in selected:
        x, y = values[:, left[pair]], values[:, right[pair]]
        observed = np.isfinite(x[fit]) & np.isfinite(y[fit])
        x_fit, y_fit = x[fit][observed], y[fit][observed]
        x_bounds = np.quantile(x_fit, [0.10, 0.90])
        middle = (x_fit >= x_bounds[0]) & (x_fit <= x_bounds[1])
        if np.std(x_fit[middle]) < 1e-8:
            continue
        slope, offset = np.polyfit(x_fit[middle], y_fit[middle], 1)
        delta = y - (slope * x + offset)
        location = np.nanmedian(delta[fit])
        low, high = np.nanpercentile(delta[fit], [10, 90])
        delta = np.arcsinh((delta - location) / max((high - low) / 2, 1e-6))
        features.append(np.nan_to_num(delta, nan=0).clip(-10, 10))
    return torch.from_numpy(np.column_stack(features).astype(np.float32))


def prepare_training(train, test, labels):
    all_rows = pd.concat([train, test], ignore_index=True)
    values = all_rows[[f"X{i:03d}" for i in range(60)]].to_numpy().copy()
    with np.load(DATA / "paired_diagnostic.npz") as pairs:
        neighbors = pairs["neighbor"]
        left = np.flatnonzero(pairs["mutual"] & (np.arange(len(labels)) < neighbors))
    right = neighbors[left]
    if (labels[left] != labels[right]).any() or (left // 5000 != right // 5000).any():
        raise ValueError("Unexpected training pairs")
    averaged = values.copy()
    averaged[left] = (values[left] + values[right]) / 2
    keep = np.ones(len(labels), dtype=bool)
    keep[right] = False
    return all_rows, values, averaged, keep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--distill", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    train, test, labels = load_data()
    frame, values, averaged, keep = prepare_training(train, test, labels)
    columns = [f"X{i:03d}" for i in range(60)]
    targets = torch.from_numpy(labels.copy())
    for fold, (fit, val) in enumerate(validation_splits(labels, 5000)):
        if fold not in args.folds:
            continue
        torch.manual_seed(1819 + fold)
        fit = fit[keep[fit]]
        fold_values = values.copy()
        fold_values[fit] = averaged[fit]
        fold_frame = frame.copy()
        fold_frame[columns] = fold_values
        base = make_features(fold_frame, fit, 60, "asinh")
        residual = robust_residuals(fold_values, fit)
        inputs = torch.cat([base, residual], dim=1)
        model = nn.Sequential(
            nn.Linear(inputs.shape[1], 224),
            ResidualBlock(224, 0.18), ResidualBlock(224, 0.18),
            nn.BatchNorm1d(224), nn.GELU(), nn.Linear(224, 11),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0014, weight_decay=0.013)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0.00015,
        )
        prior = torch.bincount(targets[fit], minlength=11).float()
        prior /= prior.sum()
        offset = 0.45 * prior.clamp_min(1e-6).log()
        evaluation = inputs[np.r_[val, np.arange(len(train), len(frame))]]
        name = "residual_distill" if args.distill else "residual"
        if args.distill:
            soft_targets, confident = teacher_probabilities(fold, val, len(test))
        print(f"Fold {fold} with {residual.shape[1]} residual measurements", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = fit[torch.randperm(len(fit)).numpy()]
            for batch in np.array_split(order, max(1, len(order) // 750)):
                optimizer.zero_grad(set_to_none=True)
                if args.distill and epoch > 4 and len(confident):
                    pseudo = confident[torch.randint(len(confident), (len(batch) // 2,)).numpy()]
                    noisy = nn.functional.dropout(evaluation[pseudo], p=0.12, training=True)
                    logits = model(torch.cat([inputs[batch], noisy])) + offset
                    supervised, distilled = logits.split([len(batch), len(pseudo)])
                    loss = nn.functional.cross_entropy(supervised, targets[batch])
                    loss += 0.35 * nn.functional.cross_entropy(distilled, soft_targets[pseudo])
                else:
                    loss = nn.functional.cross_entropy(model(inputs[batch]) + offset, targets[batch])
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch not in (8, 14, 20, args.epochs):
                continue
            model.eval()
            with torch.inference_mode():
                probs = torch.cat([
                    (model(part) + offset).softmax(dim=1) for part in evaluation.split(2048)
                ]).numpy()
            val_proba, test_proba = np.split(probs, [len(val)])
            np.savez_compressed(
                DATA / f"{name}_{epoch}_fold{fold}.npz",
                val_idx=val, val_proba=val_proba, test_proba=test_proba,
            )
            result = f1_score(labels[val], val_proba.argmax(axis=1),
                              labels=np.arange(11), average="macro")
            print(f"Fold {fold}, epoch {epoch}, macro F1 {result:.5f}", flush=True)


if __name__ == "__main__":
    main()
