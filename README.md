# Kaggle Wars 2026

**0.53597 public macro F1 · 12th on the public leaderboard after the deadline · 7 submissions**

A one-day tabular classification challenge run by the OWASP Student Chapter during Saturnalia. The task was to predict one of 11 classes from 76 anonymized numeric features and one categorical feature. Macro F1 gives each class equal weight. Kaggle scored 30% of the test set publicly and reserved 70% for the final ranking. The rank above refers to the public leaderboard. [Competition](https://www.kaggle.com/competitions/kagglewars) · [Leaderboard](https://www.kaggle.com/competitions/kagglewars/leaderboard)

## Why the validation split mattered

Reciprocal near-neighbor pairs accounted for **98.26%** of training rows under our pair diagnostic. A random split could put one member of a pair in training and the other in validation. A 1-nearest-neighbor model scored about **0.93** on random folds, so we switched to holding out whole 5,000-row blocks to keep those pairs together.

Grouped validation still looked different from the test set. Complete rows made up **54.8%** of the training set and **70.7%** of the test set. A train-versus-test classifier reached **0.945 AUC** under grouped validation. We compared unweighted results with results weighted by missingness, category, and distance to the training set. The first submission scored **0.55569** on that matched validation set and **0.51179** on the public leaderboard.

## The best entry

1. Train LightGBM specialists and neural networks on the grouped folds, including a neural network trained on averaged pairs and a model distilled from fold-specific teacher predictions.
2. Combine their predictions with a 15-nearest-neighbor model and LightGBM probabilities smoothed over a graph of similar rows.
3. Apply a small class-prior adjustment, then choose labels by optimizing estimated macro F1 across the batch.

The best entry gained **0.02418** macro F1 over the first. The full-data neural refit in round seven scored below round six.

| Round | Variant | Public macro F1 |
| ---: | --- | ---: |
| 1 | Reference ensemble | 0.51179 |
| 2 | Reference with moderate class-prior adjustment | 0.52524 |
| 3 | Reference with stronger class-prior adjustment | 0.51564 |
| 4 | Geometric ensemble with class-prior adjustment | 0.51552 |
| 5 | Neighborhood ensemble with macro F1 label assignment | 0.53100 |
| 6 | Hybrid ensemble with macro F1 label assignment | **0.53597** |
| 7 | Hybrid with full-data neural refits | 0.53298 |

## Reproduce the best entry

Use Python 3.12 and install the dependencies in an isolated environment. The original run used the CPU build of PyTorch. If needed, install that build from the [PyTorch wheel index](https://download.pytorch.org/whl/cpu).

```bash
python -m pip install -r requirements.txt
```

Download the official competition ZIP according to Kaggle's access rules and save it as `data/kagglewars.zip`. Keep the original filenames inside the ZIP. The repository shares source code and aggregate results. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/kagglewars/data).

Run the following commands from the repository root. Training runs on CPU. The scripts save fold predictions and intermediate files under `data/`.

```bash
python prepare_pairs.py
python neighbors.py
python distance_weighting.py
python match_validation.py
python adversarial_validation.py
python train_experts.py --rounds 130
python train_experts.py --rounds 100 --weight-power 0.65 --name experts_weighted
python train_mlp.py --name mlp_block --block-size 5000 --epochs 20 --width 192 --threads 3 --folds 0 1 2
python train_mlp.py --name mlp_pairs --block-size 5000 --epochs 36 --width 192 --blocks 2 --transform asinh --deduplicate --threads 3 --folds 0 1 2
python challenger.py --folds 0 1 2 --rounds 150 --save-every 50 --feature-count 60 --block-size 5000 --threads 1 --max-bin 127 --pair-average --domain-weight-power .5 --min-leaf 30 --name lgbmshift5000
python train_geometry.py --rounds 160
python train_context.py --rounds 90
python train_residual_mlp.py --threads 4
python train_residual_mlp.py --threads 3 --distill
python train_tabular_ensemble.py --threads 4
python graph_smooth.py --models 'experts_weighted_100_fold{fold}.npz' --name graph_fixed --neighbors 16 --category-factor 0.5 --mix 0.35 --steps 3 --sharpness 1
python build_alternatives.py --hybrid
python verify_submission.py data/candidate_hybrid_macro.csv
```

The verifier checks all **28,397** predictions, exact ID order, the `id,target` header, and integer labels from 0 to 10. The original round-six CSV had SHA-256 `95843313fa22e7a8572f41c33154c58b7286658d962553e2f59cb520214b70c7`. CPU libraries and platform differences can affect retraining outputs.
