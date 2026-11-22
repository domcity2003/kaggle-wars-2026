"""Build distance-weighted alternatives from grouped out-of-fold predictions."""

import argparse
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from evaluate import DATA, confusion_counts, load_predictions, score


MODELS = {
    "expert": "experts_130_fold{fold}.npz",
    "weighted_expert": "experts_weighted_100_fold{fold}.npz",
    "pair12": "mlp_pairs_12_fold{fold}.npz",
    "pair20": "mlp_pairs_20_fold{fold}.npz",
    "mlp16": "mlp_block_16_fold{fold}.npz",
    "shift": "lgbmshift5000_fold{fold}_150.npz",
    "geometry": "geometry_160_fold{fold}.npz",
    "context": "context_90_fold{fold}.npz",
    "distill": "residual_distill_20_fold{fold}.npz",
    "residual": "residual_14_fold{fold}.npz",
    "tabm": "tabular_ensemble_13_fold{fold}.npz",
    "graph": "graph_fixed_experts_weighted_100_fold{fold}.npz",
    "knn": "knn15_block5000_{fold}.npz",
}


POOLS = {
    "reference": {"pair12": 1, "pair20": 1, "expert": 2, "mlp16": 1, "shift": 1, "geometry": 1, "distill": 1.5},
    "neural": {"pair12": 1, "distill": 2, "residual": 1, "tabm": 2, "mlp16": 1},
    "mixed": {"pair12": 1, "distill": 2, "tabm": 2, "context": 1, "weighted_expert": 1, "graph": 1, "shift": 1},
    "neighborhood": {"graph": 2, "knn": 1, "context": 1, "distill": 2, "tabm": 1, "pair12": 1},
    "trees": {"expert": 2, "weighted_expert": 2, "geometry": 1, "shift": 1, "context": 1, "distill": 1},
}


def expected_macro_prediction(probability, change_cost):
    prediction = probability.argmax(1)
    original = prediction.copy()
    expected_count = probability.sum(axis=0)
    output_count = np.bincount(prediction, minlength=11).astype(float)
    true_positive = np.bincount(prediction, weights=probability[np.arange(len(prediction)), prediction], minlength=11)
    random = np.random.default_rng(819)
    for _ in range(3):
        for row in random.permutation(len(prediction)):
            current = prediction[row]
            output_count[current] -= 1
            true_positive[current] -= probability[row, current]
            denominator = np.maximum(expected_count + output_count, 1e-8)
            increase = (probability[row] * denominator - true_positive) / (denominator * (denominator + 1))
            increase[original[row]] += change_cost / len(prediction)
            best = increase.argmax()
            prediction[row] = best
            output_count[best] += 1
            true_positive[best] += probability[row, best]
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pools-only", action="store_true")
    mode.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    threadpool_limits(1)
    with ZipFile(DATA / "kagglewars.zip") as archive:
        train = pd.read_csv(archive.open("Train.csv"), usecols=["target", "Z"])
        test = pd.read_csv(archive.open("test (1).csv"), usecols=["id", "Z"])
        sample = pd.read_csv(archive.open("sample_submission (4).csv"))
    if list(sample.columns) != ["id", "target"] or not sample["id"].equals(test["id"]):
        raise ValueError("Unexpected sample schema or ID order")
    labels = train["target"].str[1:].astype(int).to_numpy()
    active, fold_ids, validation, test_models = load_predictions(list(MODELS.values()), len(train), len(test), [0, 1, 2])
    if not active.all():
        raise ValueError("Expected full grouped validation coverage")
    matched = np.load(DATA / "validation_matched.npz")
    distance = np.load(DATA / "distance_block5000.npz")
    if not np.array_equal(matched["fold"], fold_ids) or not np.array_equal(distance["fold"], fold_ids):
        raise ValueError("Mismatched grouped validation weights")
    matched_weight, distance_weight = matched["weights"], distance["train_weight"]

    def pooled(recipe, geometry):
        weights = np.asarray([recipe.get(key, 0) for key in MODELS], dtype=float)
        if geometry:
            return [softmax(np.average(np.log(np.maximum(values, 1e-5)), weights=weights, axis=0), axis=1)
                    for values in (validation, test_models)]
        return [np.average(values, weights=weights, axis=0) for values in (validation, test_models)]

    def metrics(prediction):
        return tuple(float(score(confusion_counts(labels, prediction, weight)))
                     for weight in (None, matched_weight, distance_weight))

    base_val, base_test = pooled(POOLS["reference"], False)
    reference, reference_test = base_val.argmax(1), base_test.argmax(1)

    def print_report(name, pred, test_prediction):
        scores = metrics(pred)
        moved = np.mean(test_prediction != reference_test)
        print(name, "grouped/matched/distance", np.round(scores, 5), "test disagreement", round(moved, 4), flush=True)
        return scores

    def write(filename, prediction):
        if set(prediction) != set(range(11)):
            raise ValueError("The candidate does not predict all classes")
        sample.assign(target=prediction).to_csv(DATA / filename, index=False)
        print("wrote", filename, "class counts", np.bincount(prediction).tolist(), flush=True)

    if args.hybrid:
        neighborhood_val, neighborhood_test = pooled(POOLS["neighborhood"], False)
        prior = np.bincount(labels) / len(labels)
        hybrid_val = (base_val + 2 * neighborhood_val) / 3 / prior**0.10
        hybrid_test = (base_test + 2 * neighborhood_test) / 3 / prior**0.10
        hybrid_val /= hybrid_val.sum(axis=1, keepdims=True)
        hybrid_test /= hybrid_test.sum(axis=1, keepdims=True)
        prediction = np.empty(len(train), dtype=int)
        for fold in range(3):
            take = fold_ids == fold
            prediction[take] = expected_macro_prediction(hybrid_val[take], 0.18)
        test_prediction = expected_macro_prediction(hybrid_test, 0.18)
        print_report("hybrid macro decision", prediction, test_prediction)
        write("candidate_hybrid_macro.csv", test_prediction)
        return

    pool_runs = []
    for name, recipe in POOLS.items():
        for geometric in (False, True):
            v, t = pooled(recipe, geometric)
            val_pred, test_pred = v.argmax(1), t.argmax(1)
            tag = f"{name}-{'geo' if geometric else 'mean'}"
            result = print_report(tag, val_pred, test_pred)
            if np.mean(test_pred != reference_test) >= 0.05:
                pool_runs.append((result[2] + 0.2 * result[1], tag, v, t))
    _, selected_pool, chosen_val, chosen_test = max(pool_runs, key=lambda result: result[0])

    if args.pools_only:
        mixed_val, mixed_test = pooled(POOLS["mixed"], True)
        write("candidate_mixed_geometric.csv", mixed_test.argmax(1))
        priors = np.bincount(labels) / len(labels)
        for name, v, t in (("mixed-geo", mixed_val, mixed_test), (selected_pool, chosen_val, chosen_test)):
            for alpha in (0.15, 0.35, 0.5, 0.65):
                divisor = priors ** alpha
                print_report(f"{name}-prior{alpha:.2f}", (v / divisor).argmax(1), (t / divisor).argmax(1))
        write("candidate_mixed_geometric_prior35.csv", (mixed_test / priors**0.35).argmax(1))
        return

    write("candidate_distance_pool.csv", chosen_test.argmax(1))
    print("distance pool recipe", selected_pool, flush=True)

    for penalty in (0.01, 0.07, 0.18):
        pred = np.empty(len(train), dtype=int)
        for fold in range(3):
            take = fold_ids == fold
            pred[take] = expected_macro_prediction(chosen_val[take], penalty)
        tpred = expected_macro_prediction(chosen_test, penalty)
        result = print_report(f"expected-macro-{penalty}", pred, tpred)
        if np.mean(tpred != chosen_test.argmax(1)) >= 0.05:
            pool_runs.append((result[2] + 0.2 * result[1], f"expected-macro-{penalty}", pred, tpred))

    def stack_input(probability, cohort):
        logp = np.log(np.clip(probability, 1e-4, 1))
        logits = logp - logp.mean(axis=2, keepdims=True)
        z = pd.get_dummies(pd.Categorical(cohort, categories=sorted(train["Z"].unique()))).to_numpy()
        return np.column_stack([logits.transpose(1, 0, 2).reshape(len(cohort), -1), z]).astype(np.float64)

    x_val, x_test = stack_input(validation, train["Z"]), stack_input(test_models, test["Z"])
    stack_val = np.zeros_like(base_val)
    stack_test = []
    for fold in range(3):
        take = fold_ids == fold
        estimator = LogisticRegression(C=0.01, max_iter=120, tol=2e-4)
        estimator.fit(x_val[~take], labels[~take], sample_weight=distance_weight[~take])
        stack_val[take] = estimator.predict_proba(x_val[take])
        stack_test.append(estimator.predict_proba(x_test))
    stack_test = np.mean(stack_test, axis=0)
    for stack_weight in (0.45, 0.7, 1.0):
        pred = ((1 - stack_weight) * chosen_val + stack_weight * stack_val).argmax(1)
        tpred = ((1 - stack_weight) * chosen_test + stack_weight * stack_test).argmax(1)
        result = print_report(f"stack-{stack_weight}", pred, tpred)
        if np.mean(tpred != chosen_test.argmax(1)) >= 0.05:
            pool_runs.append((result[2] + 0.2 * result[1], f"stack-{stack_weight}", pred, tpred))
    _, name, _, test_pred = max((result for result in pool_runs if result[2].ndim == 1), key=lambda result: result[0])
    write("candidate_distance_decision.csv", test_pred)
    print("distance decision recipe", name, flush=True)


if __name__ == "__main__":
    main()
