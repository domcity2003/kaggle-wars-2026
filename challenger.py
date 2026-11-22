"""Train and validate a LightGBM model on the Kaggle Wars archive."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DATA = Path(__file__).resolve().parent / "data"
LABELS = np.arange(11)


def load_data(feature_count):
    columns = [f"X{index:03}" for index in range(feature_count)]
    with ZipFile(DATA / "kagglewars.zip") as archive:
        train = pd.read_csv(archive.open("Train.csv"))
        test = pd.read_csv(archive.open("test (1).csv"))
    features = pd.concat([train[columns + ["Z"]], test[columns + ["Z"]]], ignore_index=True)
    features[columns] = features[columns].astype(np.float32)
    features["Z"] = features["Z"].astype("category")
    return features, train["target"].str[1:].astype(int).to_numpy()


def add_missing_features(features):
    missing = features.select_dtypes(include="number").isna()
    features["missing_count"] = missing.sum(axis=1).astype(np.int16)
    masks = np.packbits(missing.to_numpy().T, axis=1)
    _, first_columns, groups = np.unique(masks, axis=0, return_index=True, return_inverse=True)
    for group_index, first_column in enumerate(first_columns):
        if np.count_nonzero(groups == group_index) > 1:
            column = missing.columns[first_column]
            features[f"missing_group_{column}"] = missing[column].astype(np.int8)
    patterns = [row.tobytes() for row in np.packbits(missing.to_numpy(), axis=1)]
    counts = pd.Series(patterns).value_counts()
    retained = {pattern for pattern, count in counts.items() if count >= 100}
    features["missing_pattern"] = pd.Categorical(
        [pattern.hex() if pattern in retained else "other" for pattern in patterns]
    )


def report_metric(y, probabilities, priors):
    scores = {
        alpha: f1_score(y, np.argmax(probabilities / priors**alpha, axis=1), labels=LABELS, average="macro")
        for alpha in np.arange(0.0, 1.01, 0.05).round(2)
    }
    best_alpha = max(scores, key=scores.get)
    return scores[0], best_alpha, scores[best_alpha]


def average_fit_pairs(train, y, fit_idx, weights, pairs):
    is_fit = np.isin(np.arange(len(y)), fit_idx)
    partner = pairs["neighbor"][fit_idx]
    matched = pairs["mutual"][fit_idx] & (partner >= 0) & is_fit[partner]
    keep = ~matched | (fit_idx < partner)
    retained = fit_idx[keep]
    left = fit_idx[matched & keep]
    right = partner[matched & keep]
    averaged = train.iloc[retained].copy()
    numeric = train.columns[train.columns.str.startswith("X")]
    averaged.loc[left, numeric] = (train.loc[left, numeric].to_numpy() + train.loc[right, numeric].to_numpy()) * 0.5
    combined = weights[keep].copy()
    paired = matched[keep]
    combined[paired] = 0.5 * (combined[paired] + weights[np.searchsorted(fit_idx, right)])
    combined /= combined.mean()
    return averaged, y[retained], combined


def conditional_ranks(frame, keep_raw):
    numeric = frame.columns[frame.columns.str.startswith("X")]
    ranks = frame.groupby("Z", observed=True)[numeric].rank(method="average", pct=True)
    ranks = ranks.astype(np.float32).add_prefix("rank_")
    if keep_raw:
        return pd.concat([frame, ranks], axis=1)
    return pd.concat([frame.drop(columns=numeric), ranks], axis=1)


def main():
    from train_cat import validation_splits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--leaves", type=int, default=63)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--min-leaf", type=int, default=45)
    parser.add_argument("--weight-power", type=float, default=0)
    parser.add_argument("--domain-weight-power", type=float, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.065)
    parser.add_argument("--feature-count", type=int, default=76)
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--name", default="lgbm")
    parser.add_argument("--engine", choices=("lightgbm", "xgboost"), default="lightgbm")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--pair-average", action="store_true")
    parser.add_argument("--conditional-rank", choices=("only", "add"))
    args = parser.parse_args()

    features, y = load_data(args.feature_count)
    if not args.raw:
        add_missing_features(features)
    train = features.iloc[: len(y)]
    test = features.iloc[len(y) :]
    domain_weights = None
    if args.domain_weight_power:
        domain_weights = np.load(DATA / "validation_weights.npz")["weights"]
    pairs = None
    if args.pair_average:
        if args.block_size != 5000:
            parser.error("--pair-average needs leakage-safe --block-size 5000 folds")
        with np.load(DATA / "paired_diagnostic.npz") as pair_data:
            pairs = {name: pair_data[name] for name in ("mutual", "neighbor")}
    print(f"Loaded {len(train)} training rows with {features.shape[1]} features", flush=True)
    for fold, (fit_idx, val_idx) in enumerate(validation_splits(y, args.block_size)):
        if fold not in args.folds:
            continue
        priors = np.bincount(y[fit_idx], minlength=len(LABELS)) / len(fit_idx)
        weights = priors[y[fit_idx]] ** -args.weight_power
        if domain_weights is not None:
            weights *= domain_weights[fit_idx] ** args.domain_weight_power
        weights /= weights.mean()
        fit, fit_y = train.iloc[fit_idx], y[fit_idx]
        if pairs is not None:
            fit, fit_y, weights = average_fit_pairs(train, y, fit_idx, weights, pairs)
        val = train.iloc[val_idx]
        fit_test = test
        if args.conditional_rank is not None:
            keep_raw = args.conditional_rank == "add"
            fit, val, fit_test = [conditional_ranks(frame, keep_raw) for frame in (fit, val, fit_test)]
        print(f"fold={fold} fit rows={len(fit_y)}", flush=True)
        def should_save(iteration):
            if iteration in (1, 10):
                print(f"fold={fold} reached {iteration} rounds", flush=True)
            return not iteration % args.save_every or iteration == args.rounds

        def save_checkpoint(iteration, val_proba, test_proba):
            destination = DATA / f"{args.name}_fold{fold}_{iteration}.npz"
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                val_idx=val_idx,
                val_proba=val_proba.astype(np.float32),
                test_proba=test_proba.astype(np.float32),
            )
            temporary.replace(destination)
            raw, alpha, tuned = report_metric(y[val_idx], val_proba, priors)
            print(f"fold={fold} rounds={iteration} raw={raw:.5f} tuned={tuned:.5f} alpha={alpha:.2f}", flush=True)

        if args.engine == "xgboost":
            fit_xgboost(args, fold, fit, fit_y, weights, val, fit_test,
                        should_save, save_checkpoint)
        else:
            fit_lightgbm(args, fold, fit, fit_y, weights, val, fit_test,
                         should_save, save_checkpoint)


def fit_lightgbm(args, fold, train, y, weights, val, test, should_save, save_checkpoint):
    import lightgbm as lgb

    dataset = lgb.Dataset(train, label=y, weight=weights)
    params = {
        "objective": "multiclass",
        "num_class": len(LABELS),
        "metric": "None",
        "learning_rate": args.learning_rate,
        "num_leaves": args.leaves,
        "min_data_in_leaf": args.min_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 2.5,
        "verbosity": -1,
        "num_threads": args.threads,
        "seed": 601 + fold,
        "force_col_wise": True,
        "max_bin": args.max_bin,
    }

    def callback(env):
        iteration = env.iteration + 1
        if should_save(iteration):
            save_checkpoint(iteration, env.model.predict(val, num_threads=args.threads),
                            env.model.predict(test, num_threads=args.threads))

    lgb.train(params, dataset, num_boost_round=args.rounds, callbacks=[callback])


def fit_xgboost(args, fold, train, y, weights, val, test, should_save, save_checkpoint):
    import xgboost as xgb

    params = {
        "objective": "multi:softprob",
        "num_class": len(LABELS),
        "tree_method": "hist",
        "learning_rate": args.learning_rate,
        "max_depth": 7,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "reg_lambda": 2.5,
        "nthread": args.threads,
        "seed": 601 + fold,
        "max_bin": args.max_bin,
    }
    fit = xgb.QuantileDMatrix(train, label=y, weight=weights, max_bin=args.max_bin,
                             nthread=args.threads, enable_categorical=True)
    val_matrix = xgb.QuantileDMatrix(val, ref=fit, max_bin=args.max_bin,
                                    nthread=args.threads, enable_categorical=True)
    test_matrix = xgb.QuantileDMatrix(test, ref=fit, max_bin=args.max_bin,
                                     nthread=args.threads, enable_categorical=True)

    class Checkpoint(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            iteration = epoch + 1
            if should_save(iteration):
                save_checkpoint(iteration, model.predict(val_matrix), model.predict(test_matrix))
            return False

    xgb.train(params, fit, num_boost_round=args.rounds, callbacks=[Checkpoint()])


if __name__ == "__main__":
    main()
