"""Train CatBoost candidates for Kaggle Wars and save validation/test probabilities."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


DATA = Path(__file__).resolve().parent / "data"


def load_data():
    with ZipFile(DATA / "kagglewars.zip") as archive:
        train = pd.read_csv(archive.open("Train.csv"))
        test = pd.read_csv(archive.open("test (1).csv"))
    labels = train.pop("target").str[1:].astype(int).to_numpy()
    return train, test, labels


def features(frame, add_missing):
    result = frame.drop(columns="id")
    numeric = sorted(column for column in result if column.startswith("X"))
    result = result[numeric + ["Z"]].copy()
    result["Z"] = result["Z"].fillna("__missing__").astype(str)
    if add_missing:
        missing = result[numeric].isna()
        result["missing_total"] = missing.sum(axis=1)
        patterns = {}
        for column in numeric:
            pattern = np.packbits(missing[column].to_numpy()).tobytes()
            patterns.setdefault(pattern, []).append(column)
        for columns in patterns.values():
            if len(columns) > 1 and missing[columns[0]].any():
                result["missing_" + columns[0]] = missing[columns[0]].astype("int8")
    return result


def validation_splits(labels, block_size):
    if block_size:
        groups = np.arange(len(labels)) // block_size
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        return splitter.split(labels, labels, groups)
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(labels, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=850)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--missing", action="store_true")
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--features", type=int, default=76)
    parser.add_argument("--borders", type=int, default=254)
    parser.add_argument("--feature-fraction", type=float, default=1)
    parser.add_argument("--name", default="cat6")
    args = parser.parse_args()

    train, test, labels = load_data()
    excluded = [f"X{i:03d}" for i in range(args.features, 76)]
    all_features = features(
        pd.concat([train, test], ignore_index=True).drop(columns=excluded), args.missing
    )
    train_x, test_x = all_features.iloc[:len(train)], all_features.iloc[len(train):]
    for fold, (train_idx, val_idx) in enumerate(validation_splits(labels, args.block_size)):
        if fold not in args.folds:
            continue
        model = CatBoostClassifier(
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=0.09,
            loss_function="MultiClass",
            cat_features=["Z"],
            one_hot_max_size=8,
            border_count=args.borders,
            rsm=args.feature_fraction,
            random_seed=42 + fold,
            thread_count=args.threads,
            allow_writing_files=False,
        )
        model.fit(
            train_x.iloc[train_idx], labels[train_idx],
            eval_set=(train_x.iloc[val_idx], labels[val_idx]),
            use_best_model=False,
            verbose=100,
        )
        prior = np.bincount(labels[train_idx]) / len(train_idx)
        stages = sorted({min(stage, args.iterations) for stage in [300, 550, args.iterations]})
        for stage in stages:
            val_proba = model.predict_proba(train_x.iloc[val_idx], ntree_end=stage)
            test_proba = model.predict_proba(test_x, ntree_end=stage)
            np.savez_compressed(
                DATA / f"{args.name}_{stage}_fold{fold}.npz",
                val_idx=val_idx, val_proba=val_proba, test_proba=test_proba,
            )
            scores = {
                alpha: round(f1_score(
                    labels[val_idx], (val_proba / prior ** alpha).argmax(axis=1),
                    average="macro",
                ), 5)
                for alpha in (0, 0.15, 0.30, 0.45, 0.60)
            }
            print(f"{args.name} fold {fold} trees {stage}: {scores}", flush=True)


if __name__ == "__main__":
    main()
