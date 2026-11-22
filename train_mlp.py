"""Train a tabular MLP using the measured features and their missingness."""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from train_cat import DATA, load_data, validation_splits


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(width), nn.Linear(width, width * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout),
        )

    def forward(self, inputs):
        return inputs + self.layers(inputs)


def make_features(frame, train_idx, num_features, transform):
    numeric = frame[[f"X{i:03d}" for i in range(num_features)]].to_numpy(dtype=np.float32)
    fit_data = numeric[train_idx]
    missing = np.isnan(numeric)
    location = np.nanmedian(fit_data, axis=0)
    filled = np.where(missing, location, numeric)
    groups = np.unique(missing.T, axis=0).T
    categorical = pd.get_dummies(frame["Z"], dummy_na=True).to_numpy()
    masks = pd.get_dummies(pd.util.hash_pandas_object(
        pd.DataFrame(groups), index=False
    )).to_numpy()
    lower, upper = np.nanpercentile(fit_data, [1, 99], axis=0)
    clipped = np.clip(filled, lower, upper)
    mean = clipped[train_idx].mean(axis=0)
    scale = clipped[train_idx].std(axis=0)
    standardized = (clipped - mean) / np.maximum(scale, 1e-7)
    inputs = [standardized, groups, categorical, masks]
    if transform in ("hybrid", "asinh"):
        rank = QuantileTransformer(
            n_quantiles=512, output_distribution="normal", random_state=42,
        )
        rank.fit(fit_data)
        quantiles = np.nan_to_num(rank.transform(numeric))
        inputs.append(quantiles)
    if transform == "asinh":
        q25, q75 = np.nanpercentile(fit_data, [25, 75], axis=0)
        spread = np.maximum(q75 - q25, np.maximum(scale / 100, 1e-7))
        inputs[0] = np.arcsinh((filled - location) / spread)
    return torch.from_numpy(np.concatenate(inputs, axis=1).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="mlp")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--features", type=int, default=60)
    parser.add_argument("--transform", choices=["raw", "hybrid", "asinh"], default="hybrid")
    parser.add_argument("--seed", type=int, default=442)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--train-weights")
    parser.add_argument("--class-weight", type=float, default=0)
    parser.add_argument("--full-fit", action="store_true")
    args = parser.parse_args()
    if args.full_fit and args.name == "mlp":
        parser.error("Name the full-data run explicitly")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    train, test, labels = load_data()
    frame = pd.concat([train, test], ignore_index=True)
    training_frame = frame.copy()
    representative = np.ones(len(train), dtype=bool)
    if args.deduplicate:
        with np.load(DATA / "paired_diagnostic.npz") as pairs:
            neighbor, mutual = pairs["neighbor"], pairs["mutual"]
        left = np.flatnonzero(mutual & (np.arange(len(train)) < neighbor))
        right = neighbor[left]
        if not np.all(labels[left] == labels[right]):
            raise ValueError("Class mismatch in paired training rows")
        if args.block_size != 5000 or (left // 5000 != right // 5000).any():
            raise ValueError("Pair averaging requires the 5,000-row grouped validation split")
        paired = [f"X{i:03d}" for i in range(args.features)]
        training_frame.loc[left, paired] = (
            frame.loc[left, paired].to_numpy() + frame.loc[right, paired].to_numpy()
        ) / 2
        representative[right] = False
    all_weights = np.ones(len(train))
    if args.train_weights:
        with np.load(DATA / args.train_weights) as archive:
            all_weights = np.sqrt(np.clip(archive["weights"], 0.04, 5))
    labels_t = torch.from_numpy(labels.astype(np.int64))
    splits = (
        [(np.arange(len(train)), np.empty(0, dtype=np.int64))]
        if args.full_fit else validation_splits(labels, args.block_size)
    )
    for fold, (train_idx, val_idx) in enumerate(splits):
        if not args.full_fit and fold not in args.folds:
            continue
        torch.manual_seed(args.seed + fold)
        train_idx = train_idx[representative[train_idx]]
        fold_frame = frame.copy()
        fold_frame.iloc[train_idx] = training_frame.iloc[train_idx]
        features = make_features(fold_frame, train_idx, args.features, args.transform)
        model = nn.Sequential(
            nn.Linear(features.shape[1], args.width),
            *[ResidualBlock(args.width, 0.16) for _ in range(args.blocks)],
            nn.BatchNorm1d(args.width), nn.GELU(), nn.Linear(args.width, 11),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0013, weight_decay=0.008)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=0.00015,
        )
        prior = np.bincount(labels[train_idx]) / len(train_idx)
        loss = nn.CrossEntropyLoss(reduction="none")
        fit_weights = all_weights * prior[labels] ** -args.class_weight
        fit_weights /= fit_weights[train_idx].mean()
        weights = torch.from_numpy(fit_weights.astype(np.float32))
        eval_x = features[np.r_[val_idx, np.arange(len(train), len(frame))]]
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = train_idx[torch.randperm(len(train_idx)).numpy()]
            for batch in np.array_split(order, max(1, len(order) // 740)):
                optimizer.zero_grad(set_to_none=True)
                batch_loss = loss(model(features[batch]), labels_t[batch])
                (batch_loss * weights[batch]).mean().backward()
                optimizer.step()
            scheduler.step()
            if epoch % 4 == 0 or epoch == args.epochs:
                model.eval()
                with torch.inference_mode():
                    prediction = torch.cat([
                        model(batch).softmax(dim=1) for batch in eval_x.split(2048)
                    ]).numpy()
                prediction *= prior ** args.class_weight
                prediction /= prediction.sum(axis=1, keepdims=True)
                val_proba, test_proba = np.split(prediction, [len(val_idx)])
                np.savez_compressed(
                    DATA / f"{args.name}_{epoch}_fold{fold}.npz",
                    val_idx=val_idx, val_proba=val_proba, test_proba=test_proba,
                )
                if args.full_fit:
                    print(f"{args.name} full-data epoch {epoch}", flush=True)
                    continue
                notes = []
                for alpha in (0, 0.25, 0.5):
                    predicted = (val_proba / prior ** alpha).argmax(axis=1)
                    notes.append(round(f1_score(labels[val_idx], predicted, average="macro"), 5))
                print(f"{args.name} fold {fold} epoch {epoch}: {notes}", flush=True)


if __name__ == "__main__":
    main()
