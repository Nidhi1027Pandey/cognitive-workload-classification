"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MULTIMODAL COGNITIVE WORKLOAD CLASSIFICATION                              ║
║   Dataset  : OpenNeuro ds007169  ·  N-back Task  ·  4 Difficulty Levels    ║
║   Modalities: EEG (BrainVision)  ·  ECG/HRV  ·  Pupillometry              ║
║   Subjects : 18  (sub-001 … sub-020, gaps at 013, 017)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Applications : Neuroergonomics · BCI · Driver Fatigue · ICU Monitoring   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   PIPELINE                                                                  ║
║    1  Load events TSV → filter real n-back trials (levels 1–4)             ║
║    2  EEG  : notch → bandpass → avg-ref → ICA → epochs → spectral feats   ║
║    3  ECG  : R-peak detect → HRV time/freq/nonlinear per epoch window      ║
║    4  Pupil: dilation mean/slope/kurtosis · blink rate · fixation ratio    ║
║    5  Fuse : early concat · z-score per modality · ANOVA top-50 select    ║
║    6  Train: SVM · RF · XGBoost · MLP · Logistic  (stratified 5-fold CV)  ║
║    7  SHAP : XGBoost explainability · per-modality importance ranking      ║
║    8  Plot : 10-panel research dashboard → PNG                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  HOW TO RUN
  ──────────
  pip install -r requirements.txt
  python src/workload_pipeline.py --data-root /path/to/ds007169

  Or set the DATA_ROOT environment variable / edit data_root in config.yaml.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 · CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os, warnings, glob, json, argparse
warnings.filterwarnings("ignore")

# ── ▶  Dataset path resolution ───────────────────────────────────────────────
# Priority: --data-root CLI arg  >  DATA_ROOT env var  >  ./data/ds007169 default
def _resolve_data_root():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    root = (
        args.data_root
        or os.environ.get("DATA_ROOT")
        or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "ds007169")
    )
    out = (
        args.output_dir
        or os.environ.get("OUTPUT_DIR")
        or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results")
    )
    return root, out

DATA_ROOT, OUTPUT_DIR = _resolve_data_root()
# ─────────────────────────────────────────────────────────────────────────────

TASK         = "nback"
N_BACK_LEVELS = [1, 2, 3, 4]       # working memory load labels
EPOCH_TMIN   = -0.2                 # s before stimulus onset
EPOCH_TMAX   =  2.0                 # s after  stimulus onset
BASELINE     = (-0.2, 0.0)
REJECT_UV    = None                 # None = data-adaptive via autoreject logic
#  ↑ Fixed thresholds fail on datasets with higher baseline amplitude.
#    We compute a per-subject peak-to-peak threshold automatically instead.

BANDS = {                           # EEG frequency bands
    "delta": ( 1,  4),
    "theta": ( 4,  8),
    "alpha": ( 8, 13),
    "beta" : (13, 30),
    "gamma": (30, 40),
}

CV_FOLDS     = 5
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

import mne
from mne.preprocessing import ICA

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

print("✅  Imports OK")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · EVENTS LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_events(subject_dir: str) -> pd.DataFrame | None:
    """
    Load *_events.tsv from the eeg/ subfolder.
    Keep only real (non-tutorial, non-dropped) trials with a valid nback_level.

    Returns DataFrame with columns: onset (float, seconds), label (int 1-4)
    Returns None if file not found or no valid trials.
    """
    pattern = os.path.join(subject_dir, "eeg", f"*{TASK}*_events.tsv")
    files   = glob.glob(pattern)
    if not files:
        return None

    df = pd.read_csv(files[0], sep="\t")

    # Drop tutorial rows
    if "istutorial" in df.columns:
        df = df[df["istutorial"].isna() | (df["istutorial"] == False)]

    # Drop non-stimulus rows (dropped_samples, started_tutorial, etc.)
    if "trial_type" in df.columns:
        df = df[~df["trial_type"].str.contains(
            "dropped|tutorial|started|ended|rest|break", case=False, na=False)]

    # Keep rows with a valid nback_level integer
    df["nback_level"] = pd.to_numeric(df["nback_level"], errors="coerce")
    df = df[df["nback_level"].isin(N_BACK_LEVELS)].copy()

    if len(df) == 0:
        return None

    df["label"] = df["nback_level"].astype(int)
    df["onset"] = pd.to_numeric(df["onset"], errors="coerce")
    df = df.dropna(subset=["onset", "label"])
    return df[["onset", "label"]].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 · EEG PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class EEGProcessor:
    """
    BrainVision .vhdr → preprocess → epochs → spectral features.

    Preprocessing chain:
      notch @50 Hz → bandpass 1–40 Hz → average reference → ICA (EOG removal)

    Features per epoch (28 total):
      · Band power mean/std/frontal/posterior  ×5 bands  (20 features)
      · Theta/alpha ratio, theta/beta ratio, engagement index  (3)
      · Spectral entropy mean + std  (2)
      · RMS, kurtosis, skewness  (3)
    """

    def load(self, subject_dir: str):
        path = glob.glob(
            os.path.join(subject_dir, "eeg", f"*{TASK}*_eeg.vhdr"))[0]
        raw = mne.io.read_raw_brainvision(path, preload=True, verbose=False)

        # ── Drop non-EEG channels (D1-D5 digital triggers, misc markers) ──
        ch_types = raw.get_channel_types()
        keep = [ch for ch, t in zip(raw.ch_names, ch_types)
                if t in ("eeg", "eog")]
        if keep:
            raw.pick(keep)

        # ── Fix unit scaling ───────────────────────────────────────────────
        # BrainVision stores data as raw integers. MNE reads unit as Volts
        # (FIFF_UNIT_V, code 107) but the cal=0.1 and range=1e-6 factors
        # are not always auto-applied, leaving values in the ±millions range
        # instead of the expected ±100 µV range.
        #
        # Diagnosis: if mean |amplitude| > 1.0 the data is NOT in Volts —
        # it's in raw integer units. Apply cal × range = 0.1 × 1e-6 = 1e-7.
        sample_data = raw.get_data(start=0, stop=min(1000, raw.n_times))
        mean_abs    = np.abs(sample_data).mean()

        if mean_abs > 1.0:
            # Data is in raw ADC units — scale to proper Volts
            # Read cal and range from channel info
            cal   = raw.info["chs"][0].get("cal",   0.1)
            rng   = raw.info["chs"][0].get("range", 1e-6)
            scale = cal * rng          # typically 0.1 × 1e-6 = 1e-7
            if scale == 0 or not np.isfinite(scale):
                scale = 1e-7           # safe default

            # Apply scaling in-place
            raw._data *= scale
            scaled_mean = np.abs(raw.get_data(
                start=0, stop=min(1000, raw.n_times))).mean()
            print(f"  │  EEG    : scaled by {scale:.2e}  "
                  f"(mean |amp| {mean_abs:.0f} → {scaled_mean*1e6:.1f} µV)")

        return raw

    def _adaptive_reject_threshold(self, raw) -> dict:
        """
        Compute a data-adaptive peak-to-peak rejection threshold.
        Works in whatever unit the data is currently in (Volts after scaling).
        Strategy: 95th percentile of per-channel PtP over a 60 s window × 3.
        Clamped to [100 µV, 800 µV] in Volts.
        """
        try:
            sfreq    = raw.info["sfreq"]
            ch_types = raw.get_channel_types()
            eeg_idx  = [i for i, t in enumerate(ch_types) if t == "eeg"]
            if not eeg_idx:
                return dict(eeg=800e-6)

            n_total = raw.n_times
            mid     = n_total // 2
            n_win   = min(int(60 * sfreq), n_total)
            start   = max(0, mid - n_win // 2)
            segment = raw.get_data(picks=eeg_idx,
                                   start=start, stop=start + n_win)
            ptp_per_ch = np.ptp(segment, axis=1)
            threshold  = float(np.percentile(ptp_per_ch, 95) * 3)
            # Clamp to sensible Volt range (100 µV … 800 µV)
            threshold  = float(np.clip(threshold, 100e-6, 800e-6))
            print(f"  │  EEG    : reject threshold = {threshold*1e6:.0f} µV")
            return dict(eeg=threshold)
        except Exception:
            return dict(eeg=800e-6)

    def preprocess(self, raw):
        raw.notch_filter(freqs=50.0, verbose=False)
        raw.filter(l_freq=1.0, h_freq=40.0,
                   method="fir", fir_window="hamming", verbose=False)

        ch_types = raw.get_channel_types()
        eeg_ch   = [ch for ch, t in zip(raw.ch_names, ch_types) if t == "eeg"]
        eog_ch   = [ch for ch, t in zip(raw.ch_names, ch_types) if t == "eog"]

        if len(eeg_ch) > 1:
            raw.set_eeg_reference("average", projection=True, verbose=False)
            raw.apply_proj(verbose=False)

        n_comp = min(20, max(2, len(eeg_ch) - 1))
        ica = ICA(n_components=n_comp, random_state=RANDOM_STATE,
                  max_iter="auto")
        ica.fit(raw, verbose=False)

        if eog_ch:
            try:
                idx, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = idx[:3]
            except Exception:
                pass
        ica.apply(raw, verbose=False)
        return raw

    def make_epochs(self, raw, events_df: pd.DataFrame):
        sfreq     = raw.info["sfreq"]
        reject    = self._adaptive_reject_threshold(raw)
        ev = np.array([
            [int(r["onset"] * sfreq), 0, int(r["label"])]
            for _, r in events_df.iterrows()
        ])
        event_id = {f"{n}-back": n for n in N_BACK_LEVELS}
        epochs = mne.Epochs(
            raw, ev, event_id=event_id,
            tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
            baseline=BASELINE, preload=True,
            reject=reject, verbose=False,
        )
        return epochs

    def extract_features(self, epochs) -> pd.DataFrame:
        sfreq   = epochs.info["sfreq"]
        data    = epochs.get_data()          # (n_ep, n_ch, n_t)
        n_ch    = data.shape[1]
        f_sl    = slice(0, max(1, n_ch // 4))        # frontal proxy
        p_sl    = slice(max(1, 3*n_ch//4), n_ch)     # posterior proxy
        records = []

        for ep, label in zip(data, epochs.events[:, -1]):
            psd, freqs = mne.time_frequency.psd_array_welch(
                ep, sfreq=sfreq, fmin=1, fmax=40,
                n_fft=256, verbose=False)            # (n_ch, n_freq)
            r = {}

            for band, (lo, hi) in BANDS.items():
                mask = (freqs >= lo) & (freqs <= hi)
                bp   = psd[:, mask].mean(axis=1)
                r[f"eeg_{band}_mean"]      = bp.mean()
                r[f"eeg_{band}_std"]       = bp.std()
                r[f"eeg_{band}_frontal"]   = bp[f_sl].mean()
                r[f"eeg_{band}_posterior"] = bp[p_sl].mean()

            th = psd[:, (freqs >= 4)  & (freqs <= 8) ].mean()
            al = psd[:, (freqs >= 8)  & (freqs <= 13)].mean()
            be = psd[:, (freqs >= 13) & (freqs <= 30)].mean()

            r["eeg_theta_alpha_ratio"] = th / (al + 1e-10)
            r["eeg_theta_beta_ratio"]  = th / (be + 1e-10)
            r["eeg_engagement_index"]  = be / (al + th + 1e-10)

            pn  = psd / (psd.sum(axis=1, keepdims=True) + 1e-10)
            ent = -(pn * np.log2(pn + 1e-10)).sum(axis=1)
            r["eeg_spectral_entropy_mean"] = ent.mean()
            r["eeg_spectral_entropy_std"]  = ent.std()

            r["eeg_rms"]      = np.sqrt((ep**2).mean())
            r["eeg_kurtosis"] = kurtosis(ep.flatten())
            r["eeg_skew"]     = skew(ep.flatten())
            r["label"]        = int(label)
            records.append(r)

        return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · ECG / HRV PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class ECGProcessor:
    """
    File: ecg/*_recording-ecg_physio.tsv  (plain TSV, ~6000 KB)
    Sidecar: ecg/*_recording-ecg_physio.json  → column names + sampling rate

    HRV features per epoch window:
      Time domain  : mean RR, SDNN, RMSSD, pNN50
      Frequency    : LF power, HF power, LF/HF  (Lomb-Scargle)
      Nonlinear    : SD1, SD2  (Poincaré plot)
    """

    def _load_json(self, json_path: str):
        with open(json_path) as f:
            meta = json.load(f)
        cols  = meta.get("Columns", [])
        sfreq = float(meta.get("SamplingFrequency", 1000.0))
        return cols, sfreq

    def load(self, subject_dir: str):
        tsv  = glob.glob(os.path.join(
            subject_dir, "ecg", f"*{TASK}*_physio.tsv"))[0]
        jsn  = tsv.replace("_physio.tsv", "_physio.json")

        cols, sfreq = self._load_json(jsn) if os.path.exists(jsn) else ([], 1000.0)

        df = pd.read_csv(tsv, sep="\t", header=None if not cols else 0)
        if cols and df.shape[1] == len(cols):
            df.columns = cols
        elif df.shape[1] >= 1:
            # Guess: name first column "ecg"
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
            df.rename(columns={"col_0": "cardiac"}, inplace=True)

        return df, sfreq

    def _find_ecg_col(self, df: pd.DataFrame) -> str | None:
        for c in ["cardiac", "ECG", "ecg", "heart", "ecg_mV"]:
            if c in df.columns:
                return c
        # Fall back to first column
        return df.columns[0] if len(df.columns) > 0 else None

    def _r_peaks(self, ecg: np.ndarray, sfreq: float) -> np.ndarray:
        nyq = sfreq / 2
        hi  = min(0.49, 15 / nyq)
        b, a = sp_signal.butter(2, [5/nyq, hi], btype="band")
        filt  = sp_signal.filtfilt(b, a, ecg)
        sq    = np.diff(filt) ** 2
        win   = max(1, int(0.15 * sfreq))
        ma    = np.convolve(sq, np.ones(win)/win, mode="same")
        thr   = 0.5 * ma.max()
        peaks, _ = sp_signal.find_peaks(
            ma, height=thr, distance=int(0.25*sfreq))
        return peaks

    def _hrv(self, ecg: np.ndarray, sfreq: float) -> dict:
        zeros = {k: 0.0 for k in [
            "hrv_mean_rr","hrv_sdnn","hrv_rmssd","hrv_pnn50",
            "hrv_lf","hrv_hf","hrv_lf_hf","hrv_sd1","hrv_sd2"]}
        peaks = self._r_peaks(ecg, sfreq)
        if len(peaks) < 4:
            return zeros

        rr = np.diff(peaks) / sfreq * 1000      # ms
        f  = {}
        f["hrv_mean_rr"] = rr.mean()
        f["hrv_sdnn"]    = rr.std()
        f["hrv_rmssd"]   = np.sqrt(np.mean(np.diff(rr)**2))
        f["hrv_pnn50"]   = np.mean(np.abs(np.diff(rr)) > 50) * 100

        try:
            t  = np.cumsum(rr) / 1000.0
            fs = np.linspace(0.01, 0.5, 512)
            pg = sp_signal.lombscargle(t, rr - rr.mean(), 2*np.pi*fs)
            f["hrv_lf"]    = pg[(fs>=0.04)&(fs<=0.15)].mean()
            f["hrv_hf"]    = pg[(fs>=0.15)&(fs<=0.40)].mean()
            f["hrv_lf_hf"] = f["hrv_lf"] / (f["hrv_hf"] + 1e-10)
        except Exception:
            f["hrv_lf"] = f["hrv_hf"] = f["hrv_lf_hf"] = 0.0

        f["hrv_sd1"] = np.std((rr[1:]-rr[:-1]) / np.sqrt(2)) if len(rr)>1 else 0.
        f["hrv_sd2"] = np.std((rr[1:]+rr[:-1]) / np.sqrt(2)) if len(rr)>1 else 0.
        return f

    def extract_features(self, subject_dir: str,
                         events_df: pd.DataFrame) -> pd.DataFrame | None:
        try:
            df, sfreq = self.load(subject_dir)
        except (IndexError, FileNotFoundError):
            return None

        col = self._find_ecg_col(df)
        if col is None:
            return None
        ecg_full = df[col].values.astype(float)

        epoch_len = int((EPOCH_TMAX - EPOCH_TMIN) * sfreq)
        records   = []
        for _, row in events_df.iterrows():
            s = int(float(row["onset"]) * sfreq)
            e = min(s + epoch_len, len(ecg_full))
            if (e - s) < int(0.5 * sfreq):
                continue
            feat = self._hrv(ecg_full[s:e], sfreq)
            feat["label"] = int(row["label"])
            records.append(feat)

        return pd.DataFrame(records) if records else None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 · PUPILLOMETRY PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────

class PupilProcessor:
    """
    File: pupil/*_pupil.tsv  (plain TSV, ~22 000 KB)
    Sidecar: pupil/*_eyetrack.json  → column names + sampling rate

    Features per epoch:
      Pupil    : mean, std, range, slope (linear trend), kurtosis
      Blinks   : rate (blinks/s), mean duration (s)
                 detected as contiguous runs of pupil ≤ 0 or NaN
      Oculomotor: fixation ratio, saccade ratio, mean saccade amplitude
                  (velocity threshold: 30 deg/s proxy)
    """

    PUPIL_CANDIDATES = ["pupil_diameter", "pupil", "Pupil",
                        "diameter", "size", "PupilDiameter"]
    X_CANDIDATES     = ["x_coordinate", "gaze_x", "GazeX", "x", "X",
                        "x_pos", "gaze_position_x"]
    Y_CANDIDATES     = ["y_coordinate", "gaze_y", "GazeY", "y", "Y",
                        "y_pos", "gaze_position_y"]

    def _load_json(self, json_path: str):
        with open(json_path) as f:
            meta = json.load(f)
        cols  = meta.get("Columns", [])
        sfreq = float(meta.get("SamplingFrequency", 120.0))
        return cols, sfreq

    def load(self, subject_dir: str):
        tsv = glob.glob(
            os.path.join(subject_dir, "pupil", f"*{TASK}*_pupil.tsv"))[0]
        jsn = glob.glob(
            os.path.join(subject_dir, "pupil", f"*{TASK}*_eyetrack.json"))
        jsn = jsn[0] if jsn else None

        cols, sfreq = self._load_json(jsn) if jsn else ([], 120.0)

        df = pd.read_csv(tsv, sep="\t",
                         header=0 if cols else None,
                         low_memory=False)
        if cols and df.shape[1] == len(cols):
            df.columns = cols

        return df, sfreq

    def _col(self, df, candidates):
        for c in candidates:
            if c in df.columns:
                return df[c].values.astype(float)
        return None

    def _blinks(self, pupil):
        mask = (pupil <= 0) | ~np.isfinite(pupil)
        segs, in_b, st = [], False, 0
        for i, b in enumerate(mask):
            if b and not in_b:   in_b, st = True, i
            elif not b and in_b: segs.append((st, i)); in_b = False
        return segs

    def _epoch_feats(self, pupil, gx, gy, sfreq):
        f   = {}
        dur = len(pupil) / sfreq

        valid = pupil[np.isfinite(pupil) & (pupil > 0)]
        if len(valid) > 5:
            f["pup_mean"]     = valid.mean()
            f["pup_std"]      = valid.std()
            f["pup_range"]    = float(np.ptp(valid))
            f["pup_slope"]    = np.polyfit(np.arange(len(valid)), valid, 1)[0]
            f["pup_kurtosis"] = kurtosis(valid)
        else:
            for k in ["pup_mean","pup_std","pup_range","pup_slope","pup_kurtosis"]:
                f[k] = 0.0

        blinks = self._blinks(pupil)
        f["pup_blink_rate"]     = len(blinks) / dur if dur > 0 else 0.0
        f["pup_blink_dur_mean"] = float(np.mean(
            [(e-s)/sfreq for s,e in blinks])) if blinks else 0.0

        if gx is not None and gy is not None and len(gx) > 1:
            vel = np.sqrt(np.diff(gx)**2 + np.diff(gy)**2) * sfreq
            vel = np.nan_to_num(vel)
            fix = vel < 30.0          # deg/s threshold
            f["pup_fix_ratio"] = float(fix.mean())
            f["pup_sac_ratio"] = float(1 - fix.mean())
            sac_amp = vel[~fix]
            f["pup_sac_amp"]   = float(sac_amp.mean()) if len(sac_amp) > 0 else 0.0
        else:
            f["pup_fix_ratio"] = f["pup_sac_ratio"] = f["pup_sac_amp"] = 0.0

        return f

    def extract_features(self, subject_dir: str,
                         events_df: pd.DataFrame) -> pd.DataFrame | None:
        try:
            df, sfreq = self.load(subject_dir)
        except (IndexError, FileNotFoundError):
            return None

        pupil = self._col(df, self.PUPIL_CANDIDATES)
        gx    = self._col(df, self.X_CANDIDATES)
        gy    = self._col(df, self.Y_CANDIDATES)

        if pupil is None:
            # Try column index 3 (common BIDS eye layout: t, x, y, pupil)
            if df.shape[1] >= 4:
                pupil = df.iloc[:, 3].values.astype(float)
            else:
                return None

        epoch_len = int((EPOCH_TMAX - EPOCH_TMIN) * sfreq)
        records   = []
        for _, row in events_df.iterrows():
            s = int(float(row["onset"]) * sfreq)
            e = min(s + epoch_len, len(pupil))
            if (e - s) < int(0.3 * sfreq):
                continue
            p   = pupil[s:e]
            gx_ = gx[s:e] if gx is not None else None
            gy_ = gy[s:e] if gy is not None else None
            feat = self._epoch_feats(p, gx_, gy_, sfreq)
            feat["label"] = int(row["label"])
            records.append(feat)

        return pd.DataFrame(records) if records else None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 · FEATURE FUSION
# ─────────────────────────────────────────────────────────────────────────────

def align_and_fuse(dfs: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    dfs: {"eeg": df, "ecg": df, "pup": df}  — each has a "label" column.
    Truncates all to the shortest, z-scores per modality, concatenates.
    Returns X, y, feature_names.
    """
    min_n = min(len(d) for d in dfs.values())
    blocks, names = [], []
    y = None

    for mod, df in dfs.items():
        df  = df.iloc[:min_n].reset_index(drop=True)
        cols = [c for c in df.columns if c != "label"]
        X_m  = df[cols].fillna(0).values.astype(float)
        X_m  = StandardScaler().fit_transform(X_m)
        blocks.append(X_m)
        names += [f"{mod}__{c}" for c in cols]
        if y is None:
            y = df["label"].values

    X = np.hstack(blocks)
    return X, y, names


def select_features(X, y, names, k=50):
    k  = min(k, X.shape[1])
    sel = SelectKBest(f_classif, k=k)
    Xs  = sel.fit_transform(X, y)
    sel_names = [names[i] for i in sel.get_support(indices=True)]
    return Xs, sel_names


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 · MODELS
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "SVM":          SVC(kernel="rbf", C=5.0, gamma="scale",
                        probability=True, random_state=RANDOM_STATE),
    "Random Forest":RandomForestClassifier(n_estimators=300, max_depth=12,
                        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost":      xgb.XGBClassifier(n_estimators=300, max_depth=6,
                        learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, eval_metric="mlogloss",
                        random_state=RANDOM_STATE, verbosity=0),
    "MLP":          MLPClassifier(hidden_layer_sizes=(256, 128, 64),
                        activation="relu", alpha=1e-3, max_iter=500,
                        early_stopping=True, random_state=RANDOM_STATE),
    "Logistic":     LogisticRegression(C=1.0, max_iter=1000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE),
}


def train_evaluate(X, y, label="Multimodal") -> dict:
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    cv    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    results = {}
    print(f"\n  ── {label}  ({X.shape[1]} features · {len(y)} epochs) ──")

    for name, model in MODELS.items():
        pipe = Pipeline([("sc", StandardScaler()), ("clf", model)])
        acc  = cross_val_score(pipe, X, y_enc, cv=cv,
                               scoring="accuracy", n_jobs=-1)
        f1   = cross_val_score(pipe, X, y_enc, cv=cv,
                               scoring="f1_weighted", n_jobs=-1)
        results[name] = dict(
            acc_mean=acc.mean(), acc_std=acc.std(),
            f1_mean=f1.mean(),   f1_std=f1.std(),
        )
        print(f"    {name:<16}  Acc {acc.mean():.3f} ± {acc.std():.3f}"
              f"   F1 {f1.mean():.3f} ± {f1.std():.3f}")

    return results


def compute_shap(X, y, names, top_n=20) -> pd.DataFrame:
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    sc    = StandardScaler()
    Xs    = sc.fit_transform(X)
    model = xgb.XGBClassifier(n_estimators=200, max_depth=6,
                               learning_rate=0.05, eval_metric="mlogloss",
                               random_state=RANDOM_STATE, verbosity=0)
    model.fit(Xs, y_enc)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    # sv shape varies by SHAP version and n_classes:
    #   list  → old API: list of (n_samples, n_features) per class
    #   3-D   → new API: (n_samples, n_features, n_classes)  OR
    #                    (n_classes, n_samples, n_features)
    #   2-D   → binary: (n_samples, n_features)
    sv_arr = np.array(sv)
    if sv_arr.ndim == 3:
        if sv_arr.shape[0] == Xs.shape[0]:
            # (n_samples, n_features, n_classes)
            mean_imp = np.abs(sv_arr).mean(axis=0).mean(axis=1)
        else:
            # (n_classes, n_samples, n_features)
            mean_imp = np.abs(sv_arr).mean(axis=0).mean(axis=0)
    elif sv_arr.ndim == 2:
        mean_imp = np.abs(sv_arr).mean(axis=0)
    else:
        mean_imp = model.feature_importances_

    mean_imp = np.asarray(mean_imp).flatten()
    n = min(len(names), len(mean_imp))
    return (pd.DataFrame({"feature": list(names[:n]),
                          "importance": list(mean_imp[:n])})
              .sort_values("importance", ascending=False)
              .head(top_n)
              .reset_index(drop=True))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 · DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

BG    = "#0b0d12";  PANEL = "#12151e"
C1    = "#00e5b0";  C2    = "#7b68ee";  C3 = "#f5a623";  C4 = "#ef4444"
TEXT  = "#dde1f0";  DIM   = "#5a607a";  GRID = "#1c2030"
LC    = [C2, C1, C3, C4]   # one colour per n-back level

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,  "axes.labelcolor": TEXT,
    "xtick.color": DIM,      "ytick.color": DIM,
    "text.color": TEXT,      "grid.color": GRID,
    "grid.linewidth": 0.5,   "font.family": "monospace",
})


def dashboard(all_res, shap_df, df_eeg, df_ecg, df_pup, out_path):
    mods  = [m for m in ["EEG","ECG","Pupil","Multimodal"] if m in all_res]
    mnames = list(MODELS.keys())

    # ── Align all dataframes to shortest length so plots never mismatch ──
    n_align = len(df_eeg)
    if df_ecg is not None: n_align = min(n_align, len(df_ecg))
    if df_pup is not None: n_align = min(n_align, len(df_pup))
    df_eeg = df_eeg.iloc[:n_align].reset_index(drop=True)
    if df_ecg is not None:
        df_ecg = df_ecg.iloc[:n_align].reset_index(drop=True)
    if df_pup is not None:
        df_pup = df_pup.iloc[:n_align].reset_index(drop=True)

    fig = plt.figure(figsize=(24, 28), facecolor=BG)
    fig.suptitle(
        "MULTIMODAL COGNITIVE WORKLOAD CLASSIFICATION  ·  OpenNeuro ds007169\n"
        "EEG Spectral  ·  ECG / HRV  ·  Pupillometry          "
        "Neuroergonomics  ·  BCI  ·  Driver Fatigue  ·  ICU",
        fontsize=13, fontweight="bold", color=TEXT, y=0.987,
        fontfamily="monospace")
    gs = gridspec.GridSpec(4, 4, figure=fig,
                           hspace=0.50, wspace=0.40,
                           top=0.955, bottom=0.04, left=0.06, right=0.97)

    # ── A  Accuracy grouped bar ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    bw = 0.17;  x = np.arange(len(mnames))
    for i, (mod, col) in enumerate(zip(mods, LC)):
        vals = [all_res[mod][m]["acc_mean"] for m in mnames]
        errs = [all_res[mod][m]["acc_std"]  for m in mnames]
        ax.bar(x + i*bw, vals, bw, yerr=errs, label=mod,
               color=col, alpha=0.85, capsize=3, zorder=3)
    ax.axhline(0.25, color=DIM, ls="--", lw=0.9, label="Chance (25 %)")
    ax.set_xticks(x + bw*1.5)
    ax.set_xticklabels([m.replace(" ","\n") for m in mnames], fontsize=8)
    ax.set_ylim(0, 1.05);  ax.set_ylabel("Accuracy")
    ax.set_title("A · Classification Accuracy — Model × Modality",
                 color=C1, fontweight="bold", pad=8)
    ax.legend(fontsize=7, framealpha=0.25, loc="upper left")
    ax.grid(axis="y", zorder=0)

    # ── B  F1 heatmap ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2:])
    mat = [[all_res.get(mod,{}).get(m,{}).get("f1_mean",0)
            for m in mnames] for mod in mods]
    im  = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                    vmin=0.2, vmax=1.0, interpolation="nearest")
    ax.set_xticks(range(len(mnames)))
    ax.set_xticklabels([m.replace(" ","\n") for m in mnames], fontsize=8)
    ax.set_yticks(range(len(mods)));  ax.set_yticklabels(mods, fontsize=9)
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if v > 0.6 else "white", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="F1")
    ax.set_title("B · Weighted F1 Score — Modality × Model",
                 color=C1, fontweight="bold", pad=8)

    # ── C  EEG band power vs load ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    bands_plot = [("eeg_theta_mean","Theta\n4–8 Hz"),
                  ("eeg_alpha_mean","Alpha\n8–13 Hz"),
                  ("eeg_beta_mean", "Beta\n13–30 Hz")]
    xb = np.arange(len(bands_plot))
    for li, lbl in enumerate(N_BACK_LEVELS):
        sub = df_eeg[df_eeg["label"] == lbl]
        if not len(sub): continue
        means = [sub[b].mean() for b, _ in bands_plot]
        stds  = [sub[b].std()  for b, _ in bands_plot]
        ax.errorbar(xb, means, yerr=stds, fmt="o-", color=LC[li],
                    label=f"{lbl}-back", lw=2, ms=7,
                    capsize=4, elinewidth=1, zorder=3)
    ax.set_xticks(xb);  ax.set_xticklabels([l for _, l in bands_plot])
    ax.set_ylabel("Mean Band Power (µV²/Hz)")
    ax.set_title("C · EEG Spectral Power vs N-back Level",
                 color=C1, fontweight="bold", pad=8)
    ax.legend(fontsize=8, framealpha=0.25);  ax.grid(zorder=0)

    # ── D  HRV profile vs load ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2:])
    hrv_plot = [("hrv_sdnn","SDNN"),("hrv_rmssd","RMSSD"),
                ("hrv_lf_hf","LF/HF"),("hrv_sd1","SD1")]
    xh = np.arange(len(hrv_plot))
    if df_ecg is not None:
        for li, lbl in enumerate(N_BACK_LEVELS):
            sub  = df_ecg[df_ecg["label"] == lbl]
            if not len(sub): continue
            vals = []
            for feat, _ in hrv_plot:
                col_all = df_ecg[feat]
                vmin, vmax = col_all.min(), col_all.max()
                vals.append((sub[feat].mean()-vmin) / (vmax-vmin+1e-10))
            ax.plot(xh, vals, "s-", color=LC[li],
                    label=f"{lbl}-back", lw=2, ms=7, zorder=3)
    ax.set_xticks(xh);  ax.set_xticklabels([l for _, l in hrv_plot])
    ax.set_ylabel("Normalised Value (0–1)")
    ax.set_title("D · HRV Feature Profile vs N-back Level",
                 color=C1, fontweight="bold", pad=8)
    ax.legend(fontsize=8, framealpha=0.25);  ax.grid(zorder=0)

    # ── E  Pupil violin ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    if df_pup is not None and "pup_mean" in df_pup.columns:
        parts = [df_pup[df_pup["label"]==l]["pup_mean"].values
                 for l in N_BACK_LEVELS]
        parts = [p for p in parts if len(p) > 1]
        if parts:
            vp = ax.violinplot(parts,
                               positions=list(range(1, len(parts)+1)),
                               showmedians=True, showextrema=False)
            for pc, c in zip(vp["bodies"], LC):
                pc.set_facecolor(c);  pc.set_alpha(0.72)
            vp["cmedians"].set_color(TEXT)
    ax.set_xticks(range(1, len(N_BACK_LEVELS)+1))
    ax.set_xticklabels([f"{l}-back" for l in N_BACK_LEVELS], fontsize=8)
    ax.set_ylabel("Pupil Diameter (a.u.)");  ax.grid(axis="y", zorder=0)
    ax.set_title("E · Pupil Dilation\nvs Workload Level",
                 color=C1, fontweight="bold", pad=8)

    # ── F  SHAP importance ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1:3])
    if shap_df is not None and len(shap_df):
        top  = shap_df.head(15).sort_values("importance")
        cols_f = [C4 if "eeg" in f else (C1 if "ecg" in f else C3)
                  for f in top["feature"]]
        ax.barh(range(len(top)), top["importance"],
                color=cols_f, alpha=0.85, zorder=3)
        ax.set_yticks(range(len(top)))
        clean = [f.replace("eeg__eeg_","eeg:").replace("ecg__hrv_","hrv:")
                  .replace("pup__pup_","pup:").replace("_"," ")
                 for f in top["feature"]]
        ax.set_yticklabels(clean, fontsize=7)
        ax.set_xlabel("Mean |SHAP| value")
        ax.set_title("F · SHAP Feature Importance (XGBoost · Multimodal)",
                     color=C1, fontweight="bold", pad=8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=C4, label="EEG"),
                            Patch(facecolor=C1, label="ECG/HRV"),
                            Patch(facecolor=C3, label="Pupil")],
                  fontsize=7, framealpha=0.25, loc="lower right")
        ax.grid(axis="x", zorder=0)

    # ── G  Feature correlation heatmap ──────────────────────────────────
    ax = fig.add_subplot(gs[2, 3])
    # Align all arrays to the shortest available dataframe
    n_corr = len(df_eeg)
    if df_ecg is not None: n_corr = min(n_corr, len(df_ecg))
    if df_pup is not None: n_corr = min(n_corr, len(df_pup))

    kf = {
        "θ/α ratio":  df_eeg["eeg_theta_alpha_ratio"].values[:n_corr],
        "F-theta":    df_eeg["eeg_theta_frontal"].values[:n_corr],
        "P-alpha":    df_eeg["eeg_alpha_posterior"].values[:n_corr],
        "Engage":     df_eeg["eeg_engagement_index"].values[:n_corr],
        "N-back":     df_eeg["label"].values[:n_corr].astype(float),
    }
    if df_ecg is not None and "hrv_rmssd" in df_ecg.columns:
        kf["RMSSD"] = df_ecg["hrv_rmssd"].values[:n_corr]
        kf["LF/HF"] = df_ecg["hrv_lf_hf"].values[:n_corr]
    if df_pup is not None and "pup_mean" in df_pup.columns:
        kf["Pupil"] = df_pup["pup_mean"].values[:n_corr]

    # Final safety: drop any key whose array length doesn't match
    n_final = min(len(v) for v in kf.values())
    kf = {k: v[:n_final] for k, v in kf.items()}

    corr = pd.DataFrame(kf).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, linewidths=0.3, linecolor=BG,
                annot=True, fmt=".2f", annot_kws={"size": 6},
                cbar_kws={"shrink": 0.8})
    ax.set_title("G · Key Feature\nCorrelations",
                 color=C1, fontweight="bold", pad=8)
    ax.tick_params(labelsize=6.5, rotation=45)

    # ── H  Epoch pie ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    cnt = df_eeg["label"].value_counts().sort_index()
    ax.pie(cnt.values,
           labels=[f"{l}-back" for l in cnt.index],
           colors=LC[:len(cnt)], autopct="%1.0f%%",
           textprops={"color": TEXT, "fontsize": 8},
           wedgeprops={"linewidth": 2, "edgecolor": BG},
           pctdistance=0.75, startangle=90)
    ax.set_title("H · Epoch Distribution\nby Load Level",
                 color=C1, fontweight="bold", pad=8)

    # ── I  Theta/alpha scatter + trend ───────────────────────────────────
    ax = fig.add_subplot(gs[3, 1:3])
    rng = np.random.RandomState(0)
    for li, lbl in enumerate(N_BACK_LEVELS):
        sub = df_eeg[df_eeg["label"] == lbl]["eeg_theta_alpha_ratio"]
        if not len(sub): continue
        jx  = lbl + (rng.rand(len(sub)) - 0.5) * 0.3
        ax.scatter(jx, sub, color=LC[li], alpha=0.25, s=10, zorder=2)
        ax.plot(lbl, sub.mean(), "D", color=LC[li], ms=12, zorder=4,
                markeredgecolor="white", markeredgewidth=0.9)
    means_ta = [df_eeg[df_eeg["label"]==l]["eeg_theta_alpha_ratio"].mean()
                for l in N_BACK_LEVELS
                if len(df_eeg[df_eeg["label"]==l]) > 0]
    ax.plot(N_BACK_LEVELS[:len(means_ta)], means_ta,
            "--", color=TEXT, lw=1.5, alpha=0.6)
    ax.set_xlabel("N-back Level");  ax.set_ylabel("Theta / Alpha Ratio")
    ax.set_title("I · Frontal Theta/Alpha Ratio vs Workload\n"
                 "(validated biomarker of working-memory load)",
                 color=C1, fontweight="bold", pad=8)
    ax.grid(zorder=0)

    # ── J  Multimodal accuracy gain ──────────────────────────────────────
    ax = fig.add_subplot(gs[3, 3])
    if "Multimodal" in all_res:
        gains, snames = [], []
        for mn in mnames:
            mm = all_res["Multimodal"].get(mn, {}).get("acc_mean", 0)
            best_s = max(
                all_res.get(mod, {}).get(mn, {}).get("acc_mean", 0)
                for mod in ["EEG","ECG","Pupil"])
            gains.append((mm - best_s) * 100)
            snames.append(mn.split(" ")[0])
        cols_g = [C1 if g >= 0 else C4 for g in gains]
        ax.barh(snames, gains, color=cols_g, alpha=0.85, zorder=3)
        ax.axvline(0, color=TEXT, lw=1)
        ax.set_xlabel("Δ Accuracy (percentage points)")
        ax.set_title("J · Multimodal Gain\nvs Best Single Modality",
                     color=C1, fontweight="bold", pad=8)
        ax.grid(axis="x", zorder=0)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\n  Dashboard → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 · MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "═"*68)
    print("  COGNITIVE WORKLOAD PIPELINE  ·  ds007169  ·  18 subjects")
    print("═"*68)

    if not os.path.exists(DATA_ROOT):
        print(f"\n  ✗  DATA_ROOT not found:\n     {DATA_ROOT}")
        print(f"     Pass --data-root /path/to/ds007169 or set the DATA_ROOT env var.")
        print(f"     See README.md for download instructions.")
        return

    subjects = sorted([
        d for d in os.listdir(DATA_ROOT) if d.startswith("sub-")
        and os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    print(f"\n  Subjects  : {len(subjects)}  →  {subjects}")

    eeg_proc = EEGProcessor()
    ecg_proc = ECGProcessor()
    pup_proc = PupilProcessor()

    pool_eeg, pool_ecg, pool_pup = [], [], []

    # ── Per-subject loop ─────────────────────────────────────────────────
    for sub in subjects:
        sub_dir = os.path.join(DATA_ROOT, sub)
        print(f"\n  ┌─ {sub} " + "─"*46)

        events_df = load_events(sub_dir)
        if events_df is None or len(events_df) < 10:
            print(f"  │  No valid events — skip");  continue
        print(f"  │  Events : {len(events_df)} trials  "
              f"{dict(events_df['label'].value_counts().sort_index())}")

        # EEG ──────────────────────────────────────────────────────────
        try:
            raw    = eeg_proc.load(sub_dir)
            raw    = eeg_proc.preprocess(raw)
            epochs = eeg_proc.make_epochs(raw, events_df)
            if len(epochs) >= 10:
                df_e = eeg_proc.extract_features(epochs)
                pool_eeg.append(df_e)
                print(f"  │  EEG    : {len(df_e)} epochs · "
                      f"{len(df_e.columns)-1} features")
            else:
                print(f"  │  EEG    : only {len(epochs)} epochs after rejection")
        except Exception as exc:
            print(f"  │  EEG  ✗ : {exc}")

        # ECG ──────────────────────────────────────────────────────────
        try:
            df_c = ecg_proc.extract_features(sub_dir, events_df)
            if df_c is not None and len(df_c) >= 10:
                pool_ecg.append(df_c)
                print(f"  │  ECG    : {len(df_c)} epochs · "
                      f"{len(df_c.columns)-1} features")
            else:
                print(f"  │  ECG    : insufficient epochs")
        except Exception as exc:
            print(f"  │  ECG  ✗ : {exc}")

        # Pupil ────────────────────────────────────────────────────────
        try:
            df_p = pup_proc.extract_features(sub_dir, events_df)
            if df_p is not None and len(df_p) >= 10:
                pool_pup.append(df_p)
                print(f"  │  Pupil  : {len(df_p)} epochs · "
                      f"{len(df_p.columns)-1} features")
            else:
                print(f"  │  Pupil  : insufficient epochs")
        except Exception as exc:
            print(f"  │  Pupil✗ : {exc}")

        print(f"  └" + "─"*52)

    # ── Pool all subjects ────────────────────────────────────────────────
    if not pool_eeg:
        print("\n  ✗  No usable EEG data — check DATA_ROOT and task name.")
        return

    print("\n" + "─"*68)
    print("[POOLING] Concatenating across subjects ...")
    df_eeg = pd.concat(pool_eeg, ignore_index=True)
    df_ecg = pd.concat(pool_ecg, ignore_index=True) if pool_ecg else None
    df_pup = pd.concat(pool_pup, ignore_index=True) if pool_pup else None

    print(f"  EEG   : {len(df_eeg)} epochs")
    if df_ecg is not None: print(f"  ECG   : {len(df_ecg)} epochs")
    if df_pup is not None: print(f"  Pupil : {len(df_pup)} epochs")
    print(f"  Label distribution:\n"
          f"{df_eeg['label'].value_counts().sort_index()}")

    # Save feature CSVs
    df_eeg.to_csv(os.path.join(OUTPUT_DIR, "features_eeg.csv"), index=False)
    if df_ecg is not None:
        df_ecg.to_csv(os.path.join(OUTPUT_DIR, "features_ecg.csv"), index=False)
    if df_pup is not None:
        df_pup.to_csv(os.path.join(OUTPUT_DIR, "features_pupil.csv"), index=False)

    # ── Prepare single-modality arrays ──────────────────────────────────
    def prep(df):
        cols = [c for c in df.columns if c != "label"]
        X    = StandardScaler().fit_transform(df[cols].fillna(0).values)
        y    = df["label"].values
        return X, y

    X_eeg, y_eeg = prep(df_eeg)
    X_ecg, y_ecg = prep(df_ecg) if df_ecg is not None else (None, None)
    X_pup, y_pup = prep(df_pup) if df_pup is not None else (None, None)

    # ── Train single modalities ──────────────────────────────────────────
    print("\n[TRAINING] Single-modality baselines")
    all_res = {}
    all_res["EEG"] = train_evaluate(X_eeg, y_eeg, "EEG")
    if X_ecg is not None:
        all_res["ECG"] = train_evaluate(X_ecg, y_ecg, "ECG")
    if X_pup is not None:
        all_res["Pupil"] = train_evaluate(X_pup, y_pup, "Pupil")

    # ── Multimodal fusion ────────────────────────────────────────────────
    print("\n[TRAINING] Multimodal fusion")
    mod_dfs = {"eeg": df_eeg}
    if df_ecg is not None:
        mod_dfs["ecg"] = df_ecg.iloc[:len(df_eeg)].reset_index(drop=True)
    if df_pup is not None:
        mod_dfs["pup"] = df_pup.iloc[:len(df_eeg)].reset_index(drop=True)
    # Trim to shortest
    min_n = min(len(d) for d in mod_dfs.values())
    mod_dfs = {k: v.iloc[:min_n].reset_index(drop=True)
               for k, v in mod_dfs.items()}

    X_multi, y_multi, feat_names = align_and_fuse(mod_dfs)
    X_sel, sel_names = select_features(X_multi, y_multi, feat_names, k=50)

    all_res["Multimodal"] = train_evaluate(X_sel, y_multi, "Multimodal")

    # ── SHAP ────────────────────────────────────────────────────────────
    print("\n[SHAP] Computing feature importances ...")
    shap_df = compute_shap(X_sel, y_multi, sel_names)

    # ── Dashboard ────────────────────────────────────────────────────────
    print("\n[DASHBOARD] Rendering 10-panel figure ...")
    dash_path = os.path.join(OUTPUT_DIR, "workload_dashboard.png")
    dashboard(all_res, shap_df, df_eeg, df_ecg, df_pup, dash_path)

    # ── Results summary ──────────────────────────────────────────────────
    print("\n" + "═"*68)
    print("  FINAL RESULTS")
    print("═"*68)
    rows = []
    for mod, res in all_res.items():
        best  = max(res, key=lambda m: res[m]["acc_mean"])
        r     = res[best]
        rows.append({"Modality": mod, "Best Model": best,
                     "Accuracy": f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f}",
                     "F1 (wtd)": f"{r['f1_mean']:.3f} ± {r['f1_std']:.3f}"})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)
    shap_df.to_csv(os.path.join(OUTPUT_DIR, "shap_importance.csv"), index=False)

    print(f"\n  All outputs saved to:\n  {OUTPUT_DIR}")
    print("═"*68)


if __name__ == "__main__":
    run()
