# Multimodal Cognitive Workload Classification

Classifies working-memory load (N-back task, 4 difficulty levels) from three
physiological modalities — **EEG**, **ECG/HRV**, and **pupillometry** — using
[OpenNeuro ds007169](https://openneuro.org/datasets/ds007169).

Applications: neuroergonomics, brain-computer interfaces, driver fatigue
detection, ICU cognitive-load monitoring.

## Pipeline

1. **Events** — load `*_events.tsv`, filter to real N-back trials (levels 1–4)
2. **EEG** — notch filter → bandpass (1–40 Hz) → average reference → ICA (EOG removal) → epoching → spectral features (band power, theta/alpha ratio, spectral entropy, etc.)
3. **ECG** — R-peak detection → HRV features (time-domain, frequency-domain via Lomb-Scargle, nonlinear/Poincaré)
4. **Pupil** — dilation statistics, blink rate, fixation/saccade ratio
5. **Fusion** — per-modality z-scoring → early concatenation → ANOVA top-50 feature selection
6. **Models** — SVM, Random Forest, XGBoost, MLP, Logistic Regression (stratified 5-fold CV)
7. **Explainability** — SHAP values on the XGBoost multimodal model
8. **Dashboard** — 10-panel research figure summarizing everything above

## Setup

```bash
git clone https://github.com/<your-username>/cognitive-workload-classification.git
cd cognitive-workload-classification
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Getting the dataset

The dataset (~18 subjects, EEG + ECG + eye-tracking) is **not included** in
this repo — it's too large for git. Download it with the
[OpenNeuro CLI](https://docs.openneuro.org/packages/cli-tools):

```bash
npm install -g @openneuro/cli
openneuro download --snapshot 1.0.0 ds007169 data/ds007169
```

Or browse and download manually from the
[ds007169 dataset page](https://openneuro.org/datasets/ds007169).

Expected layout after download:

```
data/ds007169/
├── sub-001/
│   ├── eeg/    *_eeg.vhdr, *_eeg.vmrk, *_eeg.eeg, *_events.tsv
│   ├── ecg/    *_physio.tsv, *_physio.json
│   └── pupil/  *_pupil.tsv, *_eyetrack.json
├── sub-002/
│   └── ...
└── ...
```

## Running the pipeline

```bash
python src/workload_pipeline.py --data-root data/ds007169 --output-dir results
```

Both flags are optional:
- `--data-root` defaults to `data/ds007169` (or the `DATA_ROOT` env var)
- `--output-dir` defaults to `results` (or the `OUTPUT_DIR` env var)

```bash
# Equivalent using environment variables
export DATA_ROOT=/path/to/ds007169
export OUTPUT_DIR=/path/to/results
python src/workload_pipeline.py
```

## Outputs

Written to `results/` (or your `--output-dir`):

| File | Description |
|---|---|
| `features_eeg.csv` | Extracted EEG features, all subjects pooled |
| `features_ecg.csv` | Extracted HRV features |
| `features_pupil.csv` | Extracted pupillometry features |
| `results_summary.csv` | Best model + accuracy/F1 per modality |
| `shap_importance.csv` | Top SHAP feature importances |
| `workload_dashboard.png` | 10-panel results dashboard |

## Notes

- EEG rejection thresholds are computed adaptively per subject (95th
  percentile peak-to-peak amplitude), since fixed thresholds don't generalize
  across recording setups.
- The script auto-detects and corrects BrainVision unit-scaling issues
  (raw ADC units vs. Volts) before filtering.
- Modalities with insufficient epochs (<10) for a given subject are skipped
  gracefully; the pipeline still runs on whichever modalities are available.

## License

MIT — see [LICENSE](LICENSE).
