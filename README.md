# Multi-Seed_Re-Evaluation_of_RF_Jamming_Intrusion_Detection_in_Vehicular_Networks
Official repository (Code and per-seed outputs) for the paper "Protocol-Controlled Multi-Seed Re-Evaluation of RF Jamming Intrusion Detection in Vehicular Networks"
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)


The study proposes and evaluates a hybrid ensemble intrusion detection system for RF jamming attacks in VANETs, combining BiLSTM, XGBoost, and BERT WordPiece within a protocol-controlled, multi-seed statistical evaluation framework across two mobility regimes (15 m/s and 25 m/s).

---
## Repository Structure

```
.
├── README.md                            ← This file
├── common.py                            ← Shared pipeline (verbatim from notebooks)
├── requirements.txt                     ← Pinned pip dependencies
├── environment.yml                      ← Conda environment (full pinned versions)
│
├── notebooks/
│   ├── Detector_25ms_FINAL.ipynb        ← Main detector, 25 m/s regime
│   ├── Detector_15ms_FINAL.ipynb        ← Main detector, 15 m/s regime
│   ├── Detector_BERT_FINAL.ipynb        ← BERT WordPiece baseline
│   ├── Extra1_Cohens_D_Analysis.ipynb   ← Cohen's d effect sizes
│   ├── Extra2_Ablation_Paired_Test.ipynb← Ablation paired t-tests
│   ├── T1_Leakage_GapSplit.ipynb        ← Leakage audit + gap-split protocol
│   ├── T2_Ablation_Features.ipynb       ← Feature ablation with paired tests
│   └── T3_LSTM_Sensitivity.ipynb        ← LSTM activation/callback sensitivity
│
├── scripts/
│   ├── reproduce_all.sh                 ← Master end-to-end script (Steps 1–2 automated)
│   ├── export_split_indices.py          ← Generate per-seed split JSONs (requires dataset)
│   ├── generate_splits.py              ← Alternative split generator with SHA-256 manifest
│   ├── preprocess_dataset.py           ← Raw CSV → windows → tabular features
│   ├── reproduce_tables_figures.py     ← Regenerate all Tables + Figures from stored outputs
│   └── benchmark_latency.py            ← Table VIII latency benchmark
│
├── split_indices/                       ← Per-seed episode split JSONs (run export_split_indices.py)
│   ├── split_seed42_25ms.json
│   ├── split_seed42_15ms.json
│   └── ...  (10 files total: 5 seeds × 2 speeds)
│
└── outputs/
    ├── seed_runs_25ms/                  ← Per-seed JSON: accuracy, F1, CM — 25 m/s
    │   ├── seed_42_result.json
    │   ├── seed_123_result.json  ...    (seeds 42 123 456 789 2024)
    │   ├── summary.json                 ← Mean ± std over 5 seeds
    │   ├── ablation_summary.json
    │   └── knn_sensitivity_summary.json
    ├── seed_runs_15ms/                  ← Same structure for 15 m/s
    ├── baselines_25ms/                  ← RF-Stat, LSTM-ReLU, BERT baselines — 25 m/s
    ├── baselines_15ms/                  ← Same for 15 m/s
    ├── models_25ms/                     ← Saved XGBoost meta-learner .pkl (5 seeds)
    ├── extra_analysis/                  ← Cohen's d, ablation paired t-tests
    ├── tables/                          ← Pre-generated CSV tables (Table IV–VIII)
    ├── figures/                         ← Pre-generated PNG figures
    ├── complexity_benchmark_v3_25ms.json← Table VIII latency data, 25 m/s
    ├── complexity_benchmark_v3_15ms.json← Table VIII latency data, 15 m/s
    ├── tabel_VII_final.json             ← Cross-speed per-method results
    └── bert_mitigation_summary.json
```

---

## Dataset

This work uses the public **RF Jamming Dataset for Vehicular Wireless Networks** by Kosmanos et al.:

| File | Speed Regime | Used Rows (filter) |
|------|-------------|-------------------|
| `Dataset_1.csv` | 25 m/s | Speed ∈ [23, 27] m/s |
| `Dataset_2.csv` | 15 m/s | Speed ∈ [13, 17] m/s |

**Source:** [IEEE DataPort](https://ieee-dataport.org/documents/rf-jamming-dataset-vehicular-wireless-networks)
**DOI:** `10.21227/jjcg-bk62` *(please verify before submission)*

Columns used: `SNR`, `RSSI`, `PDR`, `Speed`, `Relative_Speed`, `Scenario`, `Time`

---

## Environment Setup

### Option A — Conda (recommended, fully pinned)

```bash
conda env create -f environment.yml
conda activate detector-v2x
```

### Option B — pip

```bash
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

**Key library versions** (full list in `environment.yml` / `requirements.txt`):

| Library | Version | Role |
|---------|---------|------|
| TensorFlow | 2.20.0 | BiLSTM training & inference |
| PyTorch | 2.11.0+cu128 | BERT-WP baseline |
| XGBoost | 2.0.x | Meta-classifier |
| scikit-learn | ≥1.3 | RF-Stat baseline, k-NN, metrics |
| NumPy | ≥1.24, <2.0 | Array operations |
| transformers | 4.44.2 | BERT tokeniser/model |
| tokenizers | 0.19.1 | BERT WordPiece tokenization |

**Hardware reference:** Google Colab T4 GPU, CUDA 12.8, cuDNN 9.x
(see `outputs/complexity_benchmark_v3_25ms.json` → `hardware`)

---

## Reproduction: Step-by-Step

### Path A — Verify stored outputs *(no dataset required, ~1 minute)*

Regenerate all Tables and Figures from the pre-computed per-seed JSON files:

```bash
python scripts/reproduce_tables_figures.py \
    --outputs outputs \
    --tables-dir outputs/tables \
    --fig-dir outputs/figures
```

This produces:

| File | Content |
|------|---------|
| `outputs/tables/table_IV_main_results.csv` | Accuracy / Precision / Recall / F1 (Table IV) |
| `outputs/tables/table_V_ablation.csv` | Ablation study (Table V) |
| `outputs/tables/table_VI_knn_sensitivity.csv` | k-NN sensitivity (Table VI) |
| `outputs/tables/table_VII_cross_speed.csv` | Per-method cross-speed (Table VII) |
| `outputs/tables/table_VIII_latency.csv` | Latency & complexity (Table VIII) |
| `outputs/figures/fig_accuracy_distribution.png` | Per-seed accuracy box-plots |
| `outputs/figures/fig_knn_sensitivity.png` | k-NN sensitivity chart |
| `outputs/figures/fig_confusion_matrix_15ms.png` | Confusion matrix, 15 m/s |
| `outputs/figures/fig_confusion_matrix_25ms.png` | Confusion matrix, 25 m/s |

---

### Path B — Full end-to-end reproduction *(requires datasets + GPU)*

#### Step 1 — Export fixed split indices

```bash
python scripts/export_split_indices.py \
    --dataset-25 /path/to/Dataset_1.csv \
    --dataset-15 /path/to/Dataset_2.csv \
    --outdir split_indices/
```

Generates `split_indices/split_seed{S}_{R}ms.json` for each seed S ∈ {42, 123, 456, 789, 2024}
and regime R ∈ {15, 25}, containing the exact `pseudo_run_id` lists for train/val/test.

#### Step 2 — Verify preprocessing

```bash
python scripts/preprocess_dataset.py \
    --dataset-25 /path/to/Dataset_1.csv \
    --dataset-15 /path/to/Dataset_2.csv \
    --all-seeds --save-arrays
```

Runs: speed-band filtering → pseudo-episode construction → stratified episode split →
15-second windowing → padding → StandardScaler normalisation → 35-dim tabular feature extraction.

> **Or run Steps 1–2 together:**
> ```bash
> bash scripts/reproduce_all.sh /path/to/Dataset_1.csv /path/to/Dataset_2.csv
> ```

#### Step 3 — Run main notebooks (Google Colab, T4 GPU)

Set dataset paths in cell 1 of each notebook, then run all cells:

| Notebook | Produces | Manuscript |
|----------|----------|------------|
| `Detector_25ms_FINAL.ipynb` | Tables IV–VI, VIII (25 m/s) | §4.2–4.6 |
| `Detector_15ms_FINAL.ipynb` | Tables IV–VI (15 m/s) | §4.2–4.5 |
| `Detector_BERT_FINAL.ipynb` | Table VII (BERT baseline) | §4.5 |
| `Extra1_Cohens_D_Analysis.ipynb` | Cohen's d effect sizes | §4.4 |
| `Extra2_Ablation_Paired_Test.ipynb` | Ablation paired t-tests | §4.3 |
| `T1_Leakage_GapSplit.ipynb` | Leakage audit, gap-split | §4.7 |
| `T2_Ablation_Features.ipynb` | Feature ablation (Table XI) | §4.3 |
| `T3_LSTM_Sensitivity.ipynb` | LSTM sensitivity (Table XII) | §4.6 |

#### Step 4 — Latency benchmark (Table VIII)

```bash
python scripts/benchmark_latency.py \
    --outputs outputs \
    --out outputs/latency
```

> Requires trained model weights in `outputs/models_25ms/`.
> Reference numbers in the paper were measured on Colab T4, N=1000 timed runs after 100 warm-up passes.

#### Step 5 — Regenerate tables and figures

```bash
python scripts/reproduce_tables_figures.py \
    --outputs outputs \
    --tables-dir outputs/tables \
    --fig-dir outputs/figures
```

---

## Expected Results

### Table IV — Main Results (mean ± std, 5 seeds)

| Speed | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|----------|-------------------|----------------|------------|
| 15 m/s | **91.28 ± 3.87 %** | 79.37 ± 12.31 % | 76.82 ± 17.88 % | 77.29 ± 15.09 % |
| 25 m/s | **98.17 ± 1.34 %** | 97.13 ± 2.67 % | 98.42 ± 1.26 % | 97.66 ± 1.89 % |

### Per-Seed Accuracy (Detector)

| Seed | 15 m/s | 25 m/s |
|------|--------|--------|
| 42 | 95.24 % | 97.13 % |
| 123 | 94.07 % | 99.20 % |
| 456 | 85.79 % | 97.09 % |
| 789 | 89.02 % | 97.44 % |
| 2024 | 92.31 % | 100.00 % |

---

## Reproducibility Protocol Details

### Seeds
All experiments use 5 fixed random seeds: **42, 123, 456, 789, 2024**.
Seeds control: NumPy, Python `random`, `PYTHONHASHSEED`, TensorFlow (`tf.random.set_seed`).
See `common.set_all_seeds()`.

### Preprocessing Steps
1. **Label mapping:** Scenario 1 → Interference (0), Scenario 2 → Reactive (1), Scenario 3 → Constant (2)
2. **Speed filtering:** ±2 m/s band — 25 m/s → [23, 27]; 15 m/s → [13, 17]
3. **Pseudo-episodes:** `floor(Time / 60)` groups rows into 60-second episodes
4. **Split:** Stratified episode-based (60% train / 20% val / 20% test), per-class shuffle with `np.random.RandomState(seed)`
5. **Windowing:** 15-second non-overlapping windows within each episode
6. **Padding:** Zero-pad to `max_len` computed from **training set only** (leakage-free)
7. **Normalisation:** `StandardScaler` fit on training set, applied to val/test
8. **Tabular features:** 7 statistics × 5 features = 35-dim vector per window

### Latency Benchmark Protocol
- Hardware: Google Colab T4 GPU, CUDA 12.8
- 100 warmup iterations + 1000 timed iterations, batch size = 1
- GPU sync: `tf.constant(0.0).numpy()` (TF), `torch.cuda.synchronize()` (PyTorch)
- Memory: `tracemalloc` (CPU), `tf.config.experimental.get_memory_info` / `torch.cuda.max_memory_allocated` (GPU)

### Known Limitations (disclosed in manuscript)
- **Wilcoxon p-floor:** At n=5 seeds, minimum achievable p-value is 0.0625 (§5.4)
- **Relative speed coupling:** MI at 15 m/s (55.2%) exceeds 25 m/s (47.8%)
- **BiLSTM dominance:** BiLSTM-15s branch carries ~91% decision weight in the XGBoost meta-learner

---

## Citation

```bibtex
@article{fathiyana2026protocol,
  title   = {Protocol-Controlled Multi-Seed Re-Evaluation of RF Jamming IDS
             in Vehicular Networks},
  author  = {Fathiyana, Rana Zaini and Ramli, Kalamullah and Harwahyu, Ruki
             and Gunawan, Teddy Surya},
  journal = {Future Internet},
  year    = {2026},
  publisher = {MDPI}
}
```

---

## License

Code: MIT. Dataset: subject to original Kosmanos et al. license terms.
