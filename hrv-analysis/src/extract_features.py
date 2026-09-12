import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

import EntropyHub as EH

from hrvanalysis import (
    get_time_domain_features,
    get_frequency_domain_features,
    get_csi_cvi_features,
    get_poincare_plot_features,
)

from pyhrv.hrv import hrv
import pyhrv.nonlinear as nl
import pyhrv.time_domain as td


def hrv_time_domain(rri, threshold=50):
    """
    Compute common time-domain HRV metrics.

    Parameters
    ----------
    rri : array-like
        Sequence of NN intervals (ms).
    threshold : int, optional (default=50)
        Threshold in milliseconds for NNx / pNNx calculation.

    Returns
    -------
    dict
        Dictionary with HRV metrics.
    """
    rri = np.array(rri, dtype=float)
    diff_rri = np.diff(rri)

    nnx = np.sum(np.abs(diff_rri) > threshold)

    return {
        "AVNN": np.mean(rri),
        "SDNN": np.std(rri, ddof=1),
        "SDSD": np.std(diff_rri),
        "RMSSD": np.sqrt(np.mean(diff_rri**2)),
        f"NN{threshold}": nnx,
        f"pNN{threshold}": nnx / len(diff_rri) * 100 if len(diff_rri) > 0 else np.nan,
    }


def hrv_time_domain_lib(rri):

    time_domain_features = get_time_domain_features(rri)

    return {
        "mean_nni": time_domain_features["mean_nni"],
        "sdnn": time_domain_features["sdnn"],
        "sdsd": time_domain_features["sdsd"],
        "rmssd": time_domain_features["rmssd"],
        "nni_50": time_domain_features["nni_50"],
        "pnni_50": time_domain_features["pnni_50"],
        "nni_20": time_domain_features["nni_20"],
        "pnni_20": time_domain_features["pnni_20"],
        "cvsd": time_domain_features["cvsd"],
        "cvnni": time_domain_features["cvnni"],
        "mean_hr": time_domain_features["mean_hr"],
        "max_hr": time_domain_features["max_hr"],
        "min_hr": time_domain_features["min_hr"],
        "std_hr": time_domain_features["std_hr"],
    }


def hrv_frequency_domain(rri, fs=4.0):
    """
    Compute frequency-domain HRV metrics using hrv-analysis.

    Parameters
    ----------
    rri : array-like
        Sequence of NN intervals (ms).
    fs : float, optional (default=4.0)
        Resampling frequency in Hz.

    Returns
    -------
    dict
        Dictionary with frequency-domain HRV metrics.
    """
    try:
        features = get_frequency_domain_features(rri, sampling_frequency=fs)
    except Exception as e:
        return {
            "TotalPower": np.nan,
            "VLF": np.nan,
            "LF": np.nan,
            "HF": np.nan,
            "LF/HF": np.nan,
        }

    # Substituir divisões inválidas por NaN
    for k in ["total_power", "vlf", "lf", "hf", "lf_hf_ratio"]:
        if k not in features or np.isnan(features[k]) or np.isinf(features[k]):
            features[k] = np.nan

    return {
        "TotalPower": features["total_power"],
        "VLF": features["vlf"],
        "LF": features["lf"],
        "HF": features["hf"],
        "LF/HF": features["lf_hf_ratio"],
    }


def hrv_frequency_domain_nolib(rri, fs=4.0):
    time_rri = np.cumsum(rri) / 1000.0  # seconds
    t_interp = np.arange(time_rri[0], time_rri[-1], 1 / fs)
    rri_interp = np.interp(t_interp, time_rri, rri)

    # Welch PSD
    f, pxx = welch(rri_interp, fs=fs, nperseg=256)

    vlf = trapezoid(pxx[(f >= 0.003) & (f < 0.04)], f[(f >= 0.003) & (f < 0.04)])
    lf = trapezoid(pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = trapezoid(pxx[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)])
    total_power = vlf + lf + hf

    return {"TotalPower": total_power, "VLF": vlf, "LF": lf, "HF": hf}


def get_entropies(rri):

    # Approximate Entropy
    apen = EH.ApEn(rri)

    # Sample Entropy
    sampen = EH.SampEn(rri)

    # Fuzzy Entropy
    fuzzen = EH.FuzzEn(rri)

    # Kolmogorov Entropy
    k2en = EH.K2En(rri)

    # Permutation Entropy
    permen = EH.PermEn(rri)

    # Conditional Entropy
    conden = EH.CondEn(rri)

    # Distribution Entropy
    disten = EH.DistEn(rri)

    # Spectral Entropy
    specen = EH.SpecEn(rri)

    # Dispersion Entropy
    dispen = EH.DispEn(rri)

    # Symbolic Dynamic Entropy
    sydyen = EH.SyDyEn(rri)

    # Increment Entropy
    incren = EH.IncrEn(rri)

    # Cosine Similarity Entropy
    cosien = EH.CoSiEn(rri)

    # Phase Entropy
    phasen = EH.PhasEn(rri)

    # Slope Entropy
    slopen = EH.SlopEn(rri)

    # Bubble Entropy
    bubben = EH.BubbEn(rri)

    # Gridded Distribution Entropy
    griden = EH.GridEn(rri)

    # Entropy of Entropy
    enofen = EH.EnofEn(rri)

    # Attention Entropy
    attnen = EH.AttnEn(rri)

    # Range Entropy
    rangen = EH.RangEn(rri)

    return {
        "Approximate_Entropy": apen[0][-1],
        "Sample_Entropy": sampen[0][-1],
        "Fuzzy_Entropy": fuzzen[0][-1],
        "Kolmogorov_Entropy": k2en[0][-1],
        "Permutation_Entropy": permen[0][-1],
        "Conditional_Entropy": conden[0][-1],
        "Distribution_Entropy": disten[0],
        "Range_Entropy": rangen[0],
        "Spectral_Entropy": specen[0],
        "Dispersion_Entropy": dispen[0],
        "Symbolic_Dynamic_Entropy": sydyen[0],
        "Increment_Entropy": incren,
        "Cosine_Similarity_Entropy": cosien[0],
        "Phase_Entropy": phasen,
        "Slope_Entropy": slopen[0],
        "Bubble_Entropy": bubben[0][-1],
        "Gridded_Distribution_Entropy": griden[0],
        "Entropy_of_Entropy": enofen[0],
        "Attention_Entropy": attnen[0],
    }


def hrv_nonlinear(rri):

    features_pc = get_poincare_plot_features(rri)
    features_csi_cvi = get_csi_cvi_features(rri)
    entropies = get_entropies(rri)

    dfa = nl.dfa(rri, mode="dev")

    return {
        "sd1": features_pc["sd1"],
        "sd2": features_pc["sd2"],
        "ratio_sd2_sd1": features_pc["ratio_sd2_sd1"],
        "csi": features_csi_cvi["csi"],
        "cvi": features_csi_cvi["cvi"],
        "Modified_csi": features_csi_cvi["Modified_csi"],
        "dfa_alpha1": dfa["dfa_alpha1"],
        "dfa_alpha2": dfa["dfa_alpha2"],
        **entropies,
    }


def extract_features(rri, patient_id, group_name):
    # Convert to numpy array and ensure in milliseconds
    rri = np.array(rri, dtype=float)
    if np.median(rri) < 100:  # Likely in seconds
        rri = rri * 1000  # Convert to milliseconds

    # Compute features immediately
    metrics_time = hrv_time_domain_lib(rri)
    metrics_freq = hrv_frequency_domain(rri)
    metrics_nonlinear = hrv_nonlinear(rri)

    metrics = {"ID": patient_id, "Group": group_name}
    metrics.update(metrics_time)
    metrics.update(metrics_freq)
    metrics.update(metrics_nonlinear)

    return metrics
