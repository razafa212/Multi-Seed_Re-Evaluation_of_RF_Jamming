# Multi-Seed_Re-Evaluation_of_RF_Jamming_Intrusion_Detection_in_Vehicular_Networks
Official repository (Code and per-seed outputs) for the paper "Protocol-Controlled Multi-Seed Re-Evaluation of RF Jamming Intrusion Detection in Vehicular Networks"

The study proposes and evaluates a hybrid ensemble intrusion detection system for RF jamming attacks in VANETs, combining BiLSTM, XGBoost, and BERT WordPiece within a protocol-controlled, multi-seed statistical evaluation framework across two mobility regimes (15 m/s and 25 m/s).

---

## Repository Structure

```
├── notebooks/
│   ├── Detector_15ms_FINAL.ipynb       # Experiments at 15 m/s
│   ├── Detector_25ms_FINAL.ipynb       # Experiments at 25 m/s
│   └── Detector_BERT_FINAL.ipynb       # BERT WordPiece baseline & mitigation
├── results/
│   ├── seed_runs_15ms/                 # Per-seed JSON results at 15 m/s
│   ├── seed_runs_25ms/                 # Per-seed JSON results at 25 m/s
│   ├── baselines_15ms/                 # Baseline model outputs at 15 m/s
│   ├── models_25ms/                    # Saved XGBoost meta-learner (.pkl) per seed
│   ├── complexity_benchmark_v3_15m/    # CPU/GPU latency benchmarks at 15 m/s
│   ├── complexity_benchmark_v3_25m/    # CPU/GPU latency benchmarks at 25 m/s
│   ├── extra_analysis/
│   ├── bert_mitigation_summary.json
│   └── tabel_VII_final.json
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- GPU recommended (notebooks were developed on Google Colab with CUDA)
- See `requirements.txt` for full dependency list

Key libraries:

```
tensorflow
torch
transformers==4.44.2
tokenizers==0.19.1
huggingface-hub==0.23.4
xgboost
scikit-learn
numpy
pandas
scipy
matplotlib
seaborn
```

---

## How to Run

### Option A — Google Colab (recommended)

The notebooks are designed to run on Google Colab with Google Drive mounted.

1. Upload `Dataset_1.csv` to your Google Drive at the path:
   ```
   MyDrive/Dataset RF Jamming/Dataset_1.csv
   ```
2. Open the relevant notebook in Colab.
3. Run Cell 1–2 to mount Drive and install dependencies.
4. Run all remaining cells in order.

Output JSON files will be saved automatically to `MyDrive/Detector_Results/`.

### Option B — Local / non-Colab environment

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset file at the path expected by the notebook, or update the `DATASET_PATH` variable in Cell 3 of each notebook to point to your local copy of `Dataset_1.csv`.

4. Run the notebooks:
   ```
   notebooks/Detector_15ms_FINAL.ipynb    ← start here for 15 m/s regime
   notebooks/Detector_25ms_FINAL.ipynb    ← 25 m/s regime
   notebooks/Detector_BERT_FINAL.ipynb    ← BERT baseline and mitigation variants
   ```

> **Note on Colab-specific cells:** A few cells use `from google.colab import drive` and `drive.mount(...)`. If running locally, skip those cells and set `DATASET_PATH` and output directory paths manually.

---

## Dataset

The experiments use a simulated RF jamming dataset for vehicular networks (VANETs), generated using NS-2/SUMO-based simulation.

**File:** `Dataset_1.csv`

**Access:** The dataset is not included in this repository due to size constraints.

> ⚠️ *Dataset access instructions will be updated here after the paper is accepted. If you need access before then, please contact the corresponding author.*

Interim options:
- Request the dataset directly via the contact email on the paper.
- If a public repository link (e.g., Mendeley Data, Zenodo) is established, it will be listed here.

---

## Seeds Used

All experiments follow a protocol-controlled multi-seed design using 5 fixed seeds to ensure reproducibility:

```python
SEEDS = [42, 123, 456, 789, 2024]
```

Statistical comparisons use Wilcoxon signed-rank tests across these 5 seeds. Note: at n=5, the structural minimum achievable p-value is 0.0625, making p < 0.05 mathematically unattainable — this limitation is explicitly reported in the paper.

---

## Contact

For questions regarding the code or dataset, please refer to the contact information in the published paper.
