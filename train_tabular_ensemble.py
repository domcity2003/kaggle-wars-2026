"""Fit a shared-weight neural ensemble with piecewise-linear numeric features."""

import argparse
import math

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from train_cat import DATA, load_data, validation_splits
from train_mlp import make_features
from train_residual_mlp import prepare_training


class EnsembleLayer(nn.Module):
    def __init__(self, inputs, outputs, members):
        super().__init__()
        self.linear = nn.Linear(inputs, outputs, bias=False)
        self.input_scale = nn.Parameter(torch.empty(members, inputs))
        self.output_scale = nn.Parameter(torch.ones(members, outputs))
        self.bias = nn.Parameter(torch.zeros(members, outputs))
        with torch.no_grad():
            self.input_scale.bernoulli_().mul_(2).sub_(1)

    def forward(self, values):
        return self.linear(values * self.input_scale) * self.output_scale + self.bias


def piecewise_features(base, numeric):
    quantiles = base[:, -60:]
    lower = torch.tensor([-1.6, -0.9, -0.3, 0.3, 0.9, 1.6])
    upper = torch.tensor([-0.9, -0.3, 0.3, 0.9, 1.6, 2.7])
    piecewise = ((quantiles[..., None] - lower) / (upper - lower)).clamp(0, 1)
    piecewise[torch.from_numpy(np.isnan(numeric))] = 0
    return torch.cat([base, piecewise.flatten(1) - 0.5], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    train, test, labels = load_data()
    frame, values, averaged, keep = prepare_training(train, test, labels)
    columns = [f"X{i:03d}" for i in range(60)]
    targets = torch.from_numpy(labels.copy())
    members, width = 4, 144
    for fold, (fit, val) in enumerate(validation_splits(labels, 5000)):
        if fold not in args.folds:
            continue
        torch.manual_seed(2819 + fold)
        fit = fit[keep[fit]]
        fold_numeric = values.copy()
        fold_numeric[fit] = averaged[fit]
        fold_frame = frame.copy()
        fold_frame[columns] = fold_numeric
        base = make_features(fold_frame, fit, 60, "asinh")
        inputs = piecewise_features(base, fold_numeric)
        model = nn.Sequential(
            EnsembleLayer(inputs.shape[1], width, members), nn.LayerNorm(width), nn.GELU(),
            nn.Dropout(0.12), EnsembleLayer(width, width, members), nn.LayerNorm(width),
            nn.GELU(), nn.Dropout(0.15), EnsembleLayer(width, width, members),
            nn.LayerNorm(width), nn.GELU(), nn.Dropout(0.15), EnsembleLayer(width, 11, members),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.014)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0.00015,
        )
        prior = torch.bincount(targets[fit], minlength=11).float()
        offset = 0.25 * (prior / prior.sum()).clamp_min(1e-6).log()
        evaluation = inputs[np.r_[val, np.arange(len(train), len(frame))]]
        print(f"Fold {fold}, {inputs.shape[1]} input features", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = fit[torch.randperm(len(fit)).numpy()]
            for batch in np.array_split(order, math.ceil(len(order) / 512)):
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs[batch, None, :]) + offset
                target = targets[batch].repeat_interleave(members)
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), target)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch not in (7, 13, args.epochs):
                continue
            model.eval()
            with torch.inference_mode():
                values_proba = torch.cat([
                    (model(part[:, None, :]) + offset).softmax(dim=-1).mean(dim=1)
                    for part in evaluation.split(1024)
                ]).numpy()
            val_proba, test_proba = np.split(values_proba, [len(val)])
            np.savez_compressed(
                DATA / f"tabular_ensemble_{epoch}_fold{fold}.npz",
                val_idx=val, val_proba=val_proba, test_proba=test_proba,
            )
            result = f1_score(labels[val], val_proba.argmax(axis=1),
                              labels=np.arange(11), average="macro")
            print(f"Fold {fold}, epoch {epoch}, macro F1 {result:.5f}", flush=True)


if __name__ == "__main__":
    main()
