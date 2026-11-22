"""Measure the label signal in the sixteen Gaussian-looking features."""

from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, log_loss

from train_cat import DATA, validation_splits


def main():
    columns = [f"X{i:03d}" for i in range(60, 76)]
    with ZipFile(DATA / "kagglewars.zip") as archive:
        train = pd.read_csv(archive.open("Train.csv"), usecols=["target", *columns])
        test = pd.read_csv(archive.open("test (1).csv"), usecols=columns)
    labels = train.pop("target").str[1:].astype(int).to_numpy()
    train, test = train[columns].to_numpy(), test[columns].to_numpy()
    prior = np.bincount(labels) / len(labels)
    print("Conditional mean amplitude", np.round(
        np.linalg.norm(np.stack([train[labels == i].mean(axis=0) for i in range(11)]), axis=1),
        3
    ))
    print("Test and train Gaussian means", test.mean(axis=0).round(3), train.mean(axis=0).round(3))
    print("Prior-only loss", log_loss(labels, np.tile(prior, (len(labels), 1))))
    for block_size in (0, 5000):
        for fold, (train_idx, val_idx) in enumerate(validation_splits(labels, block_size)):
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            model.fit(train[train_idx], labels[train_idx])
            val_proba = model.predict_proba(train[val_idx])
            test_proba = model.predict_proba(test)
            print(f"Block {block_size} fold {fold} loss {log_loss(labels[val_idx], val_proba):.5f}"
                  f" uniform-prior F1 {f1_score(labels[val_idx], (val_proba / prior).argmax(axis=1), average='macro'):.5f}")
            np.savez_compressed(
                DATA / f"gaussian{block_size}_fold{fold}.npz",
                val_idx=val_idx, val_proba=val_proba, test_proba=test_proba,
            )


if __name__ == "__main__":
    main()
