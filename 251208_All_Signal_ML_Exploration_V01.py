# app.py
import io
import os
import zipfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from scipy.stats import skew, kurtosis, entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION (EDIT THESE IF NEEDED)
# ============================================================

# CSV reading options
CSV_HAS_HEADER = True   # ✅ we DO have a header row (NIR, VIS, L/O, BR)
NIR_COL = "NIR"         # use the 'NIR' column
VIS_COL = "VIS"         # use the 'VIS' column

# Bead segmentation
SEGMENT_COLUMN = "L/O"  # third column used for thresholding
DEFAULT_THRESHOLD = 0.0

# Sampling frequency for FFT-based features
DEFAULT_FS = 1.0   # If unknown, 1.0 means "normalized frequency units"
MAX_FFT_LEN = 4096  # Truncate signal for FFT to this length to keep it light

# ============================================================
# BEAD SEGMENTATION FUNCTION (YOUR ORIGINAL)
# ============================================================

def segment_beads(df: pd.DataFrame, column: str, threshold: float) -> List[Tuple[int, int]]:
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def parse_defocus_from_filename(filename: str) -> str:
    """Extract defocus label from string between last '_' and '.csv'."""
    base = os.path.basename(filename)
    if base.lower().endswith(".csv"):
        stem = base[:-4]  # remove .csv
    else:
        stem = base
    parts = stem.split("_")
    if len(parts) == 0:
        return "unknown"
    return parts[-1]


def safe_array(x: np.ndarray) -> np.ndarray:
    """Ensure 1D numpy array without NaN."""
    x = np.asarray(x).astype(float)
    if x.ndim > 1:
        x = x.ravel()
    x = x[~np.isnan(x)]
    return x


# ============================================================
# FEATURE COMPUTATION
# ============================================================

def compute_time_features(signal: np.ndarray, prefix: str) -> Dict[str, float]:
    x = safe_array(signal)
    if len(x) == 0:
        return {f"{prefix}_{name}": np.nan for name in [
            "mean", "std", "min", "max", "peak_to_peak", "rms",
            "skew", "kurtosis", "energy", "abs_mean"
        ]}
    feats = {}
    feats[f"{prefix}_mean"] = float(np.mean(x))
    feats[f"{prefix}_std"] = float(np.std(x))
    feats[f"{prefix}_min"] = float(np.min(x))
    feats[f"{prefix}_max"] = float(np.max(x))
    feats[f"{prefix}_peak_to_peak"] = float(np.max(x) - np.min(x))
    feats[f"{prefix}_rms"] = float(np.sqrt(np.mean(x ** 2)))
    feats[f"{prefix}_skew"] = float(skew(x)) if len(x) > 2 else np.nan
    feats[f"{prefix}_kurtosis"] = float(kurtosis(x)) if len(x) > 3 else np.nan
    feats[f"{prefix}_energy"] = float(np.sum(x ** 2))
    feats[f"{prefix}_abs_mean"] = float(np.mean(np.abs(x)))
    return feats


def compute_freq_features(
    signal: np.ndarray,
    prefix: str,
    fs: float = DEFAULT_FS,
    max_len: int = MAX_FFT_LEN,
) -> Dict[str, float]:
    x = safe_array(signal)
    if len(x) == 0:
        return {f"{prefix}_{name}": np.nan for name in [
            "dom_freq", "spec_centroid", "spec_bandwidth",
            "spec_energy", "low_band_ratio", "high_band_ratio"
        ]}

    # Truncate for FFT efficiency
    if len(x) > max_len:
        x = x[:max_len]

    # Remove DC mean
    x = x - np.mean(x)

    # Real FFT
    X = np.fft.rfft(x)
    P = np.abs(X) ** 2  # power spectrum
    if fs <= 0:
        fs = 1.0
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    P_sum = np.sum(P)
    if P_sum == 0:
        return {f"{prefix}_{name}": np.nan for name in [
            "dom_freq", "spec_centroid", "spec_bandwidth",
            "spec_energy", "low_band_ratio", "high_band_ratio"
        ]}

    # Dominant frequency
    dom_freq = freqs[np.argmax(P)]

    # Spectral centroid
    spec_centroid = np.sum(freqs * P) / P_sum

    # Spectral bandwidth (std around centroid)
    spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * P) / P_sum)

    # Total energy
    spec_energy = P_sum

    # Band ratios
    nyquist = fs / 2.0
    f_low = 0.05 * nyquist
    f_high = 0.5 * nyquist

    low_mask = freqs < f_low
    high_mask = freqs > f_high

    low_band_energy = np.sum(P[low_mask]) if np.any(low_mask) else 0.0
    high_band_energy = np.sum(P[high_mask]) if np.any(high_mask) else 0.0

    low_band_ratio = low_band_energy / P_sum
    high_band_ratio = high_band_energy / P_sum

    feats = {
        f"{prefix}_dom_freq": float(dom_freq),
        f"{prefix}_spec_centroid": float(spec_centroid),
        f"{prefix}_spec_bandwidth": float(spec_bandwidth),
        f"{prefix}_spec_energy": float(spec_energy),
        f"{prefix}_low_band_ratio": float(low_band_ratio),
        f"{prefix}_high_band_ratio": float(high_band_ratio),
    }
    return feats


def compute_shape_features(signal: np.ndarray, prefix: str) -> Dict[str, float]:
    x = safe_array(signal)
    if len(x) == 0:
        return {f"{prefix}_{name}": np.nan for name in [
            "q10", "q25", "q50", "q75", "q90", "iqr", "amp_entropy"
        ]}

    q10 = np.percentile(x, 10)
    q25 = np.percentile(x, 25)
    q50 = np.percentile(x, 50)
    q75 = np.percentile(x, 75)
    q90 = np.percentile(x, 90)
    iqr = q75 - q25

    # Coarse histogram entropy
    hist, bins = np.histogram(x, bins=16, density=True)
    hist = hist + 1e-12
    amp_entropy = float(entropy(hist))

    feats = {
        f"{prefix}_q10": float(q10),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q50": float(q50),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q90": float(q90),
        f"{prefix}_iqr": float(iqr),
        f"{prefix}_amp_entropy": amp_entropy,
    }
    return feats


def compute_cross_channel_features(
    nir: np.ndarray,
    vis: np.ndarray,
    prefix: str = "cross",
) -> Dict[str, float]:
    nir = safe_array(nir)
    vis = safe_array(vis)
    min_len = min(len(nir), len(vis))
    if min_len == 0:
        return {
            f"{prefix}_corr_coef": np.nan,
            f"{prefix}_nir_vis_mean_ratio": np.nan,
            f"{prefix}_nir_vis_energy_ratio": np.nan,
            f"{prefix}_nir_minus_vis_mean": np.nan,
            f"{prefix}_nir_minus_vis_rms": np.nan,
        }
    nir = nir[:min_len]
    vis = vis[:min_len]

    # Correlation
    if np.std(nir) == 0 or np.std(vis) == 0:
        corr_coef = np.nan
    else:
        corr_coef = float(np.corrcoef(nir, vis)[0, 1])

    nir_mean = float(np.mean(nir))
    vis_mean = float(np.mean(vis))
    nir_energy = float(np.sum(nir ** 2))
    vis_energy = float(np.sum(vis ** 2))

    eps = 1e-12
    nir_vis_mean_ratio = nir_mean / (vis_mean + eps)
    nir_vis_energy_ratio = nir_energy / (vis_energy + eps)

    diff = nir - vis
    nir_minus_vis_mean = float(np.mean(diff))
    nir_minus_vis_rms = float(np.sqrt(np.mean(diff ** 2)))

    feats = {
        f"{prefix}_corr_coef": corr_coef,
        f"{prefix}_nir_vis_mean_ratio": nir_vis_mean_ratio,
        f"{prefix}_nir_vis_energy_ratio": nir_vis_energy_ratio,
        f"{prefix}_nir_minus_vis_mean": nir_minus_vis_mean,
        f"{prefix}_nir_minus_vis_rms": nir_minus_vis_rms,
    }
    return feats


def compute_features_for_bead(
    bead_id: str,
    file_id: int,
    file_name: str,
    defocus_label: str,
    nir: np.ndarray,
    vis: np.ndarray,
    fs: float,
) -> Dict[str, float]:
    features = {
        "bead_id": bead_id,
        "file_id": file_id,
        "file_name": file_name,
        "defocus_label": defocus_label,
    }

    # NIR features
    features.update(compute_time_features(nir, "NIR"))
    features.update(compute_freq_features(nir, "NIR", fs=fs))
    features.update(compute_shape_features(nir, "NIR"))

    # VIS features
    features.update(compute_time_features(vis, "VIS"))
    features.update(compute_freq_features(vis, "VIS", fs=fs))
    features.update(compute_shape_features(vis, "VIS"))

    # Cross-channel
    features.update(compute_cross_channel_features(nir, vis, "CROSS"))
    return features


# ============================================================
# PROCESS ZIP → FILES, BEADS, SIGNALS, FEATURES
# ============================================================

@st.cache_data(show_spinner=True)
def process_zip_and_compute_features(
    zip_bytes: bytes,
    fs: float,
    threshold: float,
    segment_column,  # allow int or str
    nir_col,
    vis_col,
):
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))

    files_records = []
    bead_records = []
    signals_dict = {}
    feature_rows = []

    file_id_counter = 0

    for name in zf.namelist():
        if not name.lower().endswith(".csv"):
            continue

        # Read CSV into DataFrame
        with zf.open(name) as f:
            if CSV_HAS_HEADER:
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, header=None)
                df.columns = list(range(df.shape[1]))

        # 👇 IMPORTANT: check that all three columns exist
        if segment_column not in df.columns or nir_col not in df.columns or vis_col not in df.columns:
            st.warning(
                f"Skipping file '{name}': expected columns {nir_col}, {vis_col}, {segment_column} "
                f"but got {list(df.columns)}"
            )
            continue

        defocus_label = parse_defocus_from_filename(name)

        files_records.append(
            {
                "file_id": file_id_counter,
                "file_name": os.path.basename(name),
                "defocus_label": defocus_label,
                "n_rows": len(df),
            }
        )

        # ✅ segmentation uses 3rd column (SEGMENT_COLUMN = 2)
        segments = segment_beads(df, segment_column, threshold)

        if not segments:
            segments = [(0, len(df) - 1)]

        for bead_index, (start_idx, end_idx) in enumerate(segments, start=1):
            bead_id = f"file{file_id_counter}_bead{bead_index}"

            # ✅ NIR & VIS use 1st and 2nd columns
            nir_signal = df[nir_col].to_numpy()[start_idx : end_idx + 1]
            vis_signal = df[vis_col].to_numpy()[start_idx : end_idx + 1]

            signals_dict[bead_id] = {
                "NIR": nir_signal,
                "VIS": vis_signal,
            }

            bead_records.append(
                {
                    "bead_id": bead_id,
                    "file_id": file_id_counter,
                    "file_name": os.path.basename(name),
                    "defocus_label": defocus_label,
                    "bead_index": bead_index,
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "length": int(end_idx - start_idx + 1),
                }
            )

            feats = compute_features_for_bead(
                bead_id=bead_id,
                file_id=file_id_counter,
                file_name=os.path.basename(name),
                defocus_label=defocus_label,
                nir=nir_signal,
                vis=vis_signal,
                fs=fs,
            )
            feature_rows.append(feats)

        file_id_counter += 1

    files_df = pd.DataFrame(files_records)
    beads_df = pd.DataFrame(bead_records)
    features_df = pd.DataFrame(feature_rows)

    return files_df, beads_df, signals_dict, features_df



# ============================================================
# FEATURE FAMILY DEFINITIONS (COLUMN NAMES)
# ============================================================

TIME_FEATURES = [
    # NIR
    "NIR_mean", "NIR_std", "NIR_min", "NIR_max", "NIR_peak_to_peak",
    "NIR_rms", "NIR_skew", "NIR_kurtosis", "NIR_energy", "NIR_abs_mean",
    # VIS
    "VIS_mean", "VIS_std", "VIS_min", "VIS_max", "VIS_peak_to_peak",
    "VIS_rms", "VIS_skew", "VIS_kurtosis", "VIS_energy", "VIS_abs_mean",
]

FREQ_FEATURES = [
    # NIR
    "NIR_dom_freq", "NIR_spec_centroid", "NIR_spec_bandwidth",
    "NIR_spec_energy", "NIR_low_band_ratio", "NIR_high_band_ratio",
    # VIS
    "VIS_dom_freq", "VIS_spec_centroid", "VIS_spec_bandwidth",
    "VIS_spec_energy", "VIS_low_band_ratio", "VIS_high_band_ratio",
]

SHAPE_FEATURES = [
    # NIR
    "NIR_q10", "NIR_q25", "NIR_q50", "NIR_q75", "NIR_q90",
    "NIR_iqr", "NIR_amp_entropy",
    # VIS
    "VIS_q10", "VIS_q25", "VIS_q50", "VIS_q75", "VIS_q90",
    "VIS_iqr", "VIS_amp_entropy",
]

CROSS_FEATURES = [
    "CROSS_corr_coef",
    "CROSS_nir_vis_mean_ratio",
    "CROSS_nir_vis_energy_ratio",
    "CROSS_nir_minus_vis_mean",
    "CROSS_nir_minus_vis_rms",
]


# ============================================================
# MODELING HELPERS
# ============================================================

def get_scaled_features(features_df: pd.DataFrame, feature_cols: List[str], scaling: str):
    X = features_df[feature_cols].to_numpy(dtype=float)
    mask = np.any(~np.isnan(X), axis=1)
    X = X[mask]
    meta = features_df.loc[mask, ["bead_id", "file_name", "defocus_label"]].reset_index(drop=True)

    if scaling == "Standard":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling == "MinMax":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    return X_scaled, meta


def pca_2d_plot(X: np.ndarray, meta: pd.DataFrame, color_col: str, title: str):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_plot = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "color": meta[color_col].astype(str).values,
        }
    )

    fig, ax = plt.subplots()
    for label in df_plot["color"].unique():
        mask = df_plot["color"] == label
        ax.scatter(df_plot.loc[mask, "PC1"], df_plot.loc[mask, "PC2"], label=label, alpha=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    st.pyplot(fig)


# ============================================================
# STREAMLIT APP
# ============================================================

st.set_page_config(page_title="Welding Defect – Unsupervised Exploration", layout="wide")
st.title("Welding Defect – Unsupervised Exploration App")

# ---------------- Sidebar: Upload & Segmentation ----------------

# Initialize session_state containers once
if "files_df" not in st.session_state:
    st.session_state["files_df"] = None
if "beads_df" not in st.session_state:
    st.session_state["beads_df"] = None
if "signals_dict" not in st.session_state:
    st.session_state["signals_dict"] = {}
if "features_df" not in st.session_state:
    st.session_state["features_df"] = None

st.sidebar.header("1. Data & Segmentation")

uploaded_zip = st.sidebar.file_uploader("Upload ZIP of CSV files", type=["zip"])

fs = st.sidebar.number_input(
    "Sampling frequency (for FFT features)",
    min_value=0.0001,
    value=float(DEFAULT_FS),
    step=0.1,
)

threshold = st.sidebar.number_input(
    f"Segmentation threshold on column {SEGMENT_COLUMN}",
    value=float(DEFAULT_THRESHOLD),
    step=0.1,
)

run_segmentation = st.sidebar.button(
    "Segment Bead & Compute Features",
    disabled=(uploaded_zip is None),
)

# When button is clicked and a zip is present -> process and store in session_state
if run_segmentation and uploaded_zip is not None:
    with st.spinner("Processing ZIP, segmenting beads, and computing features..."):
        files_df, beads_df, signals_dict, features_df = process_zip_and_compute_features(
            zip_bytes=uploaded_zip.getvalue(),  # get raw bytes from uploaded file
            fs=fs,
            threshold=threshold,
            segment_column=SEGMENT_COLUMN,      # 3rd column (index 2) used for segmentation
            nir_col=NIR_COL,                    # 1st column (index 0) used as NIR
            vis_col=VIS_COL,                    # 2nd column (index 1) used as VIS
        )
        st.session_state["files_df"] = files_df
        st.session_state["beads_df"] = beads_df
        st.session_state["signals_dict"] = signals_dict
        st.session_state["features_df"] = features_df

        st.success(
            f"Processed {len(files_df)} file(s), {len(beads_df)} bead(s). "
            f"Features extracted: {features_df.shape[1] - 4} per bead."
        )

# 🔴 IMPORTANT: these lines must NOT be indented under the if-block above.
# They run every time the script reruns, so all tabs can see the stored data.

files_df = st.session_state["files_df"]
beads_df = st.session_state["beads_df"]
signals_dict = st.session_state["signals_dict"]
features_df = st.session_state["features_df"]

# ---------------- Main Tabs ----------------

overview_tab, clustering_tab, anomaly_tab = st.tabs(["Overview", "Clustering", "Anomaly Detection"])

# ============================================================
# OVERVIEW TAB
# ============================================================
with overview_tab:
    st.header("Overview")

    if features_df is None or features_df.empty:
        st.info("After uploading the ZIP, click 'Segment Bead & Compute Features' in the sidebar.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of files", int(files_df["file_id"].nunique()))
        with col2:
            st.metric("Number of beads", int(features_df["bead_id"].nunique()))
        with col3:
            st.metric("Features per bead", features_df.shape[1] - 4)  # exclude metadata cols

        st.subheader("Defect distribution")
        df_counts = features_df.groupby("defocus_label")["bead_id"].nunique().reset_index()
        df_counts = df_counts.rename(columns={"bead_id": "num_beads"})
        st.bar_chart(df_counts.set_index("defocus_label"))

        st.subheader("Example signals")
        unique_defocus = sorted(features_df["defocus_label"].unique())
        selected_defocus = st.selectbox("Select defect level", unique_defocus)

        beads_in_defocus = features_df[features_df["defocus_label"] == selected_defocus]["bead_id"].tolist()
        selected_bead = st.selectbox("Select bead", beads_in_defocus)

        if selected_bead in signals_dict:
            sig = signals_dict[selected_bead]
            nir = sig["NIR"]
            vis = sig["VIS"]
            t = np.arange(len(nir))

            fig, ax = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
            ax[0].plot(t, nir)
            ax[0].set_ylabel("NIR")
            ax[0].grid(True)

            ax[1].plot(t, vis)
            ax[1].set_ylabel("VIS")
            ax[1].set_xlabel("Sample index")
            ax[1].grid(True)

            fig.suptitle(f"Bead: {selected_bead} | Defect: {selected_defocus}")
            st.pyplot(fig)
        else:
            st.warning("Signal data not found for this bead (unexpected).")


# ============================================================
# CLUSTERING TAB
# ============================================================
with clustering_tab:
    st.header("Clustering")

    if features_df is None or features_df.empty:
        st.info("Please upload data and run segmentation first.")
    else:
        st.subheader("Feature selection")

        selected_features = []

        with st.expander("Time Domain Features", expanded=True):
            time_feats_selected = st.multiselect(
                "Select time-domain features",
                options=[f for f in TIME_FEATURES if f in features_df.columns],
                default=[f for f in TIME_FEATURES if f in features_df.columns][:4],
                key="time_feats_cluster",
            )
            selected_features.extend(time_feats_selected)

        with st.expander("Frequency Domain Features", expanded=False):
            freq_feats_selected = st.multiselect(
                "Select frequency-domain features",
                options=[f for f in FREQ_FEATURES if f in features_df.columns],
                default=[],
                key="freq_feats_cluster",
            )
            selected_features.extend(freq_feats_selected)

        with st.expander("Shape & Distribution Features", expanded=False):
            shape_feats_selected = st.multiselect(
                "Select shape/distribution features",
                options=[f for f in SHAPE_FEATURES if f in features_df.columns],
                default=[],
                key="shape_feats_cluster",
            )
            selected_features.extend(shape_feats_selected)

        with st.expander("Cross-Channel Features", expanded=False):
            cross_feats_selected = st.multiselect(
                "Select cross-channel features",
                options=[f for f in CROSS_FEATURES if f in features_df.columns],
                default=[],
                key="cross_feats_cluster",
            )
            selected_features.extend(cross_feats_selected)

        with st.expander("Wavelet Features (future)", expanded=False):
            st.write("Wavelet-domain features are not implemented in this lightweight v1.")

        selected_features = list(dict.fromkeys(selected_features))  # unique while preserving order
        st.write(f"Selected {len(selected_features)} feature(s):")
        st.code(", ".join(selected_features) if selected_features else "None selected")

        if len(selected_features) < 2:
            st.warning("Select at least 2 features for clustering.")
        else:
            st.subheader("Clustering models")

            model_tabs = st.tabs(["K-Means", "Gaussian Mixture", "DBSCAN"])

            # --------------------- K-Means ---------------------
            with model_tabs[0]:
                st.markdown("### K-Means Clustering")

                k_kmeans = st.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, step=1)
                scaling_kmeans = st.radio(
                    "Scaling",
                    options=["Standard", "MinMax", "None"],
                    index=0,
                    horizontal=True,
                    key="kmeans_scaling",
                )

                if st.button("Run K-Means"):
                    X_scaled, meta = get_scaled_features(features_df, selected_features, scaling_kmeans)
                    kmeans = KMeans(n_clusters=k_kmeans, random_state=42, n_init="auto")
                    cluster_labels = kmeans.fit_predict(X_scaled)

                    # Metrics
                    sil = silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else np.nan
                    ari = adjusted_rand_score(meta["defocus_label"], cluster_labels)
                    nmi = normalized_mutual_info_score(meta["defocus_label"], cluster_labels)

                    st.write("**Metrics:**")
                    st.write(f"- Silhouette score: {sil:.3f}" if not np.isnan(sil) else "- Silhouette score: N/A")
                    st.write(f"- Adjusted Rand Index (vs defocus): {ari:.3f}")
                    st.write(f"- Normalized Mutual Information (vs defocus): {nmi:.3f}")

                    # Contingency table
                    df_res = meta.copy()
                    df_res["cluster"] = cluster_labels
                    contingency = pd.crosstab(df_res["cluster"], df_res["defocus_label"])
                    st.write("**Cluster vs Defect contingency table:**")
                    st.dataframe(contingency)

                    # PCA plots
                    st.write("**PCA Projection (color = cluster)**")
                    pca_2d_plot(X_scaled, df_res, color_col="cluster", title="PCA – K-Means clusters")

                    st.write("**PCA Projection (color = defocus label)**")
                    pca_2d_plot(X_scaled, df_res, color_col="defocus_label", title="PCA – Defect labels")

            # --------------------- GMM ---------------------
            with model_tabs[1]:
                st.markdown("### Gaussian Mixture Model")

                k_gmm = st.slider("Number of components (k)", min_value=2, max_value=10, value=4, step=1)
                cov_type = st.selectbox(
                    "Covariance type",
                    options=["full", "tied", "diag", "spherical"],
                    index=0,
                )
                scaling_gmm = st.radio(
                    "Scaling",
                    options=["Standard", "MinMax", "None"],
                    index=0,
                    horizontal=True,
                    key="gmm_scaling",
                )

                if st.button("Run GMM"):
                    X_scaled, meta = get_scaled_features(features_df, selected_features, scaling_gmm)
                    gmm = GaussianMixture(n_components=k_gmm, covariance_type=cov_type, random_state=42)
                    cluster_labels = gmm.fit_predict(X_scaled)

                    sil = silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else np.nan
                    ari = adjusted_rand_score(meta["defocus_label"], cluster_labels)
                    nmi = normalized_mutual_info_score(meta["defocus_label"], cluster_labels)

                    st.write("**Metrics:**")
                    st.write(f"- Silhouette score: {sil:.3f}" if not np.isnan(sil) else "- Silhouette score: N/A")
                    st.write(f"- Adjusted Rand Index (vs defocus): {ari:.3f}")
                    st.write(f"- Normalized Mutual Information (vs defocus): {nmi:.3f}")

                    df_res = meta.copy()
                    df_res["cluster"] = cluster_labels
                    contingency = pd.crosstab(df_res["cluster"], df_res["defocus_label"])
                    st.write("**Cluster vs Defect contingency table:**")
                    st.dataframe(contingency)

                    st.write("**PCA Projection (color = cluster)**")
                    pca_2d_plot(X_scaled, df_res, color_col="cluster", title="PCA – GMM clusters")

                    st.write("**PCA Projection (color = defocus label)**")
                    pca_2d_plot(X_scaled, df_res, color_col="defocus_label", title="PCA – Defect labels")

            # --------------------- DBSCAN ---------------------
            with model_tabs[2]:
                st.markdown("### DBSCAN Clustering")

                eps = st.slider("eps", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                min_samples = st.slider("min_samples", min_value=2, max_value=20, value=5, step=1)
                scaling_dbscan = st.radio(
                    "Scaling",
                    options=["Standard", "MinMax", "None"],
                    index=0,
                    horizontal=True,
                    key="dbscan_scaling",
                )

                if st.button("Run DBSCAN"):
                    X_scaled, meta = get_scaled_features(features_df, selected_features, scaling_dbscan)
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    cluster_labels = dbscan.fit_predict(X_scaled)

                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    st.write(f"Detected clusters (excluding noise): {n_clusters}")

                    sil = silhouette_score(X_scaled, cluster_labels) if n_clusters > 1 else np.nan
                    ari = adjusted_rand_score(meta["defocus_label"], cluster_labels)
                    nmi = normalized_mutual_info_score(meta["defocus_label"], cluster_labels)

                    st.write("**Metrics:**")
                    st.write(f"- Silhouette score: {sil:.3f}" if not np.isnan(sil) else "- Silhouette score: N/A")
                    st.write(f"- Adjusted Rand Index (vs defocus): {ari:.3f}")
                    st.write(f"- Normalized Mutual Information (vs defocus): {nmi:.3f}")

                    df_res = meta.copy()
                    df_res["cluster"] = cluster_labels
                    contingency = pd.crosstab(df_res["cluster"], df_res["defocus_label"])
                    st.write("**Cluster vs Defect contingency table:**")
                    st.dataframe(contingency)

                    st.write("**PCA Projection (color = cluster)**")
                    pca_2d_plot(X_scaled, df_res, color_col="cluster", title="PCA – DBSCAN clusters")

                    st.write("**PCA Projection (color = defocus label)**")
                    pca_2d_plot(X_scaled, df_res, color_col="defocus_label", title="PCA – Defect labels")


# ============================================================
# ANOMALY DETECTION TAB
# ============================================================
with anomaly_tab:
    st.header("Anomaly Detection")

    if features_df is None or features_df.empty:
        st.info("Please upload data and run segmentation first.")
    else:
        st.subheader("Feature selection")

        selected_features_anom = []

        with st.expander("Time Domain Features", expanded=True):
            time_feats_selected = st.multiselect(
                "Select time-domain features",
                options=[f for f in TIME_FEATURES if f in features_df.columns],
                default=[f for f in TIME_FEATURES if f in features_df.columns][:4],
                key="time_feats_anom",
            )
            selected_features_anom.extend(time_feats_selected)

        with st.expander("Frequency Domain Features", expanded=False):
            freq_feats_selected = st.multiselect(
                "Select frequency-domain features",
                options=[f for f in FREQ_FEATURES if f in features_df.columns],
                default=[],
                key="freq_feats_anom",
            )
            selected_features_anom.extend(freq_feats_selected)

        with st.expander("Shape & Distribution Features", expanded=False):
            shape_feats_selected = st.multiselect(
                "Select shape/distribution features",
                options=[f for f in SHAPE_FEATURES if f in features_df.columns],
                default=[],
                key="shape_feats_anom",
            )
            selected_features_anom.extend(shape_feats_selected)

        with st.expander("Cross-Channel Features", expanded=False):
            cross_feats_selected = st.multiselect(
                "Select cross-channel features",
                options=[f for f in CROSS_FEATURES if f in features_df.columns],
                default=[],
                key="cross_feats_anom",
            )
            selected_features_anom.extend(cross_feats_selected)

        with st.expander("Wavelet Features (future)", expanded=False):
            st.write("Wavelet-domain features are not implemented in this lightweight v1.")

        selected_features_anom = list(dict.fromkeys(selected_features_anom))
        st.write(f"Selected {len(selected_features_anom)} feature(s):")
        st.code(", ".join(selected_features_anom) if selected_features_anom else "None selected")

        if len(selected_features_anom) < 2:
            st.warning("Select at least 2 features for anomaly detection.")
        else:
            model_tabs_anom = st.tabs(["Isolation Forest", "Local Outlier Factor"])

            # --------------- Isolation Forest ---------------
            with model_tabs_anom[0]:
                st.markdown("### Isolation Forest")

                contamination = st.slider(
                    "Contamination (expected fraction of anomalies)",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                )
                scaling_if = st.radio(
                    "Scaling",
                    options=["Standard", "MinMax", "None"],
                    index=0,
                    horizontal=True,
                    key="if_scaling",
                )

                if st.button("Run Isolation Forest"):
                    X_scaled, meta = get_scaled_features(features_df, selected_features_anom, scaling_if)
                    iso = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                    )
                    iso.fit(X_scaled)
                    scores = -iso.score_samples(X_scaled)  # higher -> more anomalous
                    preds = iso.predict(X_scaled)          # -1 = outlier, 1 = inlier
                    is_outlier = preds == -1

                    df_res = meta.copy()
                    df_res["anomaly_score"] = scores
                    df_res["is_outlier"] = is_outlier

                    st.write("**Anomaly score distribution**")
                    fig, ax = plt.subplots()
                    ax.hist(scores, bins=20)
                    ax.set_xlabel("Anomaly score (higher = more anomalous)")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

                    st.write("**Anomaly rate by defocus label**")
                    stats = (
                        df_res.groupby("defocus_label")["is_outlier"]
                        .agg(["count", "sum"])
                        .rename(columns={"count": "num_samples", "sum": "num_outliers"})
                    )
                    stats["outlier_rate"] = stats["num_outliers"] / stats["num_samples"]
                    st.dataframe(stats)

                    st.write("**PCA Projection (color = inlier/outlier)**")
                    X_scaled_if, meta_if = X_scaled, df_res
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled_if)
                    df_plot = pd.DataFrame(
                        {
                            "PC1": X_pca[:, 0],
                            "PC2": X_pca[:, 1],
                            "status": np.where(meta_if["is_outlier"], "outlier", "inlier"),
                        }
                    )
                    fig2, ax2 = plt.subplots()
                    for status in ["inlier", "outlier"]:
                        mask = df_plot["status"] == status
                        ax2.scatter(df_plot.loc[mask, "PC1"], df_plot.loc[mask, "PC2"], label=status, alpha=0.7)
                    ax2.set_xlabel("PC1")
                    ax2.set_ylabel("PC2")
                    ax2.set_title("PCA – Isolation Forest")
                    ax2.legend()
                    st.pyplot(fig2)

            # --------------- Local Outlier Factor ---------------
            with model_tabs_anom[1]:
                st.markdown("### Local Outlier Factor")

                contamination_lof = st.slider(
                    "Contamination (expected fraction of anomalies)",
                    min_value=0.01,
                    max_value=0.3,
                    value=0.1,
                    step=0.01,
                    key="lof_contamination",
                )
                n_neighbors = st.slider(
                    "Number of neighbors",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=1,
                )
                scaling_lof = st.radio(
                    "Scaling",
                    options=["Standard", "MinMax", "None"],
                    index=0,
                    horizontal=True,
                    key="lof_scaling",
                )

                if st.button("Run LOF"):
                    X_scaled, meta = get_scaled_features(features_df, selected_features_anom, scaling_lof)
                    lof = LocalOutlierFactor(
                        n_neighbors=n_neighbors,
                        contamination=contamination_lof,
                        novelty=False,
                    )
                    preds = lof.fit_predict(X_scaled)
                    scores = -lof.negative_outlier_factor_
                    is_outlier = preds == -1

                    df_res = meta.copy()
                    df_res["anomaly_score"] = scores
                    df_res["is_outlier"] = is_outlier

                    st.write("**Anomaly score distribution**")
                    fig, ax = plt.subplots()
                    ax.hist(scores, bins=20)
                    ax.set_xlabel("LOF anomaly score (higher = more anomalous)")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)

                    st.write("**Anomaly rate by defocus label**")
                    stats = (
                        df_res.groupby("defocus_label")["is_outlier"]
                        .agg(["count", "sum"])
                        .rename(columns={"count": "num_samples", "sum": "num_outliers"})
                    )
                    stats["outlier_rate"] = stats["num_outliers"] / stats["num_samples"]
                    st.dataframe(stats)

                    st.write("**PCA Projection (color = inlier/outlier)**")
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    df_plot = pd.DataFrame(
                        {
                            "PC1": X_pca[:, 0],
                            "PC2": X_pca[:, 1],
                            "status": np.where(df_res["is_outlier"], "outlier", "inlier"),
                        }
                    )
                    fig2, ax2 = plt.subplots()
                    for status in ["inlier", "outlier"]:
                        mask = df_plot["status"] == status
                        ax2.scatter(df_plot.loc[mask, "PC1"], df_plot.loc[mask, "PC2"], label=status, alpha=0.7)
                    ax2.set_xlabel("PC1")
                    ax2.set_ylabel("PC2")
                    ax2.set_title("PCA – LOF")
                    ax2.legend()
                    st.pyplot(fig2)
