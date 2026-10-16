# Strict Benchmark and Pipeline for Smart Ponzi Scheme Detection

Anonymized review material for manuscript JIAAS-D-26-00059.

## Contents

| File | Description |
|---|---|
| `PonziMamba_R1.ipynb` | Complete pipeline: data preparation and MinHash-LSH audit, all models (XGBoost, TextCNN, Transformer, SSM, frozen GraphCodeBERT probe), LNF fusion, statistical analysis (Part A, Section 12), and sensitivity/leakage experiments (Part B, Section 13). |
| `results.jsonl` | Every recorded run: config, per-fold metrics, decision thresholds, and per-fold test + validation predictions. Part A of the notebook reproduces all statistics in the paper from this file alone (deterministic, seed 42). |
| `data/ponzi_clean.jsonl` | The strict benchmark: 4,021 contracts (158 Ponzi) after exact + near-duplicate (Jaccard >= 0.9) deduplication. |
| `data/near_duplicate_pairs.csv` | Full near-duplicate audit trail: 14,648 pairs with labels and Jaccard similarity, including the 113 conflicting-label pairs (rows where `label_a != label_b`). |

## Reproducing the paper's numbers

1. Create the environment: Python 3.12, `torch` (CUDA build), `transformers`,
   `xgboost`, `scikit-learn`, `scipy`, `numpy`.
2. Open `PonziMamba_R1.ipynb`, run the definition cells (Sections 0-4).
3. Run Section 12 (Part A, CPU, ~2 min): reproduces Tables 2, 3, and 5 of the
   paper from `results.jsonl`.
4. Optional: Section 13 (Part B, GPU) rebuilds the benchmark variants and
   reproduces Tables 6-8.

The underlying contract corpus is the public XBlock/SourceP release; the
notebook downloads it automatically in Section 1.
