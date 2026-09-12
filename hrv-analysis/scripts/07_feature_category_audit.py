from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Hard-coded mapping: feature -> (Categoria, Subcategoria)
# Categoria values expected: "clínica", "linear", "não-linear"
# Subcategoria values expected: "tempo", "frequência", "poincaré", "entropia", "compressão", or ""/None
# ---------------------------------------------------------------------
FEATURE_MAP: dict[str, tuple[str, str]] = {
    # linear / tempo
    "mean_nni": ("linear", "tempo"),
    "sdnn": ("linear", "tempo"),
    "sdsd": ("linear", "tempo"),
    "rmssd": ("linear", "tempo"),
    "nni_50": ("linear", "tempo"),
    "pnni_50": ("linear", "tempo"),
    "nni_20": ("linear", "tempo"),
    "pnni_20": ("linear", "tempo"),
    "cvsd": ("linear", "tempo"),
    "cvnni": ("linear", "tempo"),
    "mean_hr": ("linear", "tempo"),
    "max_hr": ("linear", "tempo"),
    "min_hr": ("linear", "tempo"),
    "std_hr": ("linear", "tempo"),
    # linear / frequência
    "TotalPower": ("linear", "frequência"),
    "VLF": ("linear", "frequência"),
    "LF": ("linear", "frequência"),
    "HF": ("linear", "frequência"),
    "LF/HF": ("linear", "frequência"),
    # não-linear / poincaré
    "sd1": ("não-linear", "poincaré"),
    "sd2": ("não-linear", "poincaré"),
    "ratio_sd2_sd1": ("não-linear", "poincaré"),
    # não-linear (sem subcategoria informada na sua lista)
    "csi": ("não-linear", ""),
    "cvi": ("não-linear", ""),
    "Modified_csi": ("não-linear", ""),
    "dfa_alpha1": ("não-linear", ""),
    "dfa_alpha2": ("não-linear", ""),
    # não-linear / entropia
    "Approximate_Entropy": ("não-linear", "entropia"),
    "Sample_Entropy": ("não-linear", "entropia"),
    "Fuzzy_Entropy": ("não-linear", "entropia"),
    "Kolmogorov_Entropy": ("não-linear", "entropia"),
    "Permutation_Entropy": ("não-linear", "entropia"),
    "Conditional_Entropy": ("não-linear", "entropia"),
    "Distribution_Entropy": ("não-linear", "entropia"),
    "Range_Entropy": ("não-linear", "entropia"),
    "Spectral_Entropy": ("não-linear", "entropia"),
    "Dispersion_Entropy": ("não-linear", "entropia"),
    "Symbolic_Dynamic_Entropy": ("não-linear", "entropia"),
    "Increment_Entropy": ("não-linear", "entropia"),
    "Cosine_Similarity_Entropy": ("não-linear", "entropia"),
    "Phase_Entropy": ("não-linear", "entropia"),
    "Slope_Entropy": ("não-linear", "entropia"),
    "Bubble_Entropy": ("não-linear", "entropia"),
    "Gridded_Distribution_Entropy": ("não-linear", "entropia"),
    "Entropy_of_Entropy": ("não-linear", "entropia"),
    "Attention_Entropy": ("não-linear", "entropia"),
    # não-linear / compressão
    "Scale_1_Compressed": ("não-linear", "compressão"),
    "Scale_2_Compressed": ("não-linear", "compressão"),
    "Scale_3_Compressed": ("não-linear", "compressão"),
    "Scale_4_Compressed": ("não-linear", "compressão"),
    "Scale_5_Compressed": ("não-linear", "compressão"),
    "CR1_original": ("não-linear", "compressão"),
    "CR2_original": ("não-linear", "compressão"),
    "CR3_original": ("não-linear", "compressão"),
    "CR4_original": ("não-linear", "compressão"),
    "CR5_original": ("não-linear", "compressão"),
    "CR_CI": ("não-linear", "compressão"),
    "CR_Slope": ("não-linear", "compressão"),
    "CR_CISlope": ("não-linear", "compressão"),
    "MSC_CI": ("não-linear", "compressão"),
    "MSC_Slope": ("não-linear", "compressão"),
    "MSC_CISlope": ("não-linear", "compressão"),
    # clínica
    "Sexo": ("clínica", ""),
    "Idade": ("clínica", ""),
    "HDL": ("clínica", ""),
    "Triglicerides": ("clínica", ""),
    "Colesterol": ("clínica", ""),
    "PAS": ("clínica", ""),
    "PAD": ("clínica", ""),
    "FC": ("clínica", ""),
    "Circ_Cintura": ("clínica", ""),
    "Circ_Quadril": ("clínica", ""),
    "Altura": ("clínica", ""),
    "Peso": ("clínica", ""),
    "IMC": ("clínica", ""),
    "LDL": ("clínica", ""),
    # extras da sua lista
    "Mean": ("linear", ""),
    "SD": ("linear", ""),
    "SampEn_1": ("não-linear", "entropia"),
    "SampEn_2": ("não-linear", "entropia"),
    "SampEn_3": ("não-linear", "entropia"),
    "SampEn_4": ("não-linear", "entropia"),
    "SampEn_5": ("não-linear", "entropia"),
    "CI": ("não-linear", "compressão"),
    "Slope": ("não-linear", "entropia"),
    "CISlope": ("não-linear", "compressão"),
}


# ---------------------------------------------------------------------
# Label rules for final “Category/Subcategory” display
# ---------------------------------------------------------------------
def normalize_labels(cat: str, sub: str) -> tuple[str, str]:
    cat = (cat or "").strip().lower()
    sub = (sub or "").strip().lower()

    if cat == "clínica":
        return ("Clínica", "Clínica")

    if cat == "linear":
        if sub == "tempo":
            return ("Linear", "Linear: Tempo")
        if sub == "frequência":
            return ("Linear", "Linear: Frequência")
        return ("Linear", "Linear: (outros)")

    if cat == "não-linear":
        if sub == "poincaré":
            return ("Não-Linear", "Não-Linear: Poincaré")
        if sub == "entropia":
            return ("Não-Linear", "Não-Linear: Entropia")
        if sub == "compressão":
            return ("Não-Linear", "Não-Linear: Compressão")
        return ("Não-Linear", "Não-Linear: (outros)")

    return ("Unknown", "Unknown")


def load_features_list(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Features list not found: {path}")
    feats = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features",
        type=str,
        default="outputs/ml_features_used.txt",
        help="Path to ml_features_used.txt (one feature per line)",
    )
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    ap.add_argument(
        "--outname",
        type=str,
        default="ml_features_category_audit.csv",
        help="Output CSV filename",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feats = load_features_list(args.features)

    rows = []
    unknown = []
    for f in feats:
        if f in FEATURE_MAP:
            cat_raw, sub_raw = FEATURE_MAP[f]
        else:
            cat_raw, sub_raw = ("unknown", "unknown")
            unknown.append(f)

        cat, subcat = normalize_labels(cat_raw, sub_raw)
        rows.append({"Feature": f, "Categoria": cat, "Subcategoria": subcat})

    df = pd.DataFrame(rows)

    # Summary counts
    sub_counts = (
        df.groupby(["Categoria", "Subcategoria"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["Categoria", "Subcategoria"])
    )
    cat_counts = (
        df.groupby(["Categoria"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["Categoria"])
    )

    outpath = outdir / args.outname

    # Write a single CSV file with sections:
    # 1) feature listing
    # 2) counts by subcategory
    # 3) counts by category
    with open(outpath, "w", encoding="utf-8", newline="") as f:
        f.write("# ML features used: categorized audit\n")
        f.write(f"# source_features_file: {Path(args.features).resolve()}\n")
        f.write(f"# n_features: {len(df)}\n")
        f.write("\n")

        f.write("[FEATURES]\n")
        df.to_csv(f, index=False)
        f.write("\n")

        f.write("[COUNTS_BY_SUBCATEGORY]\n")
        sub_counts.to_csv(f, index=False)
        f.write("\n")

        f.write("[COUNTS_BY_CATEGORY]\n")
        cat_counts.to_csv(f, index=False)
        f.write("\n")

        if unknown:
            f.write("[UNKNOWN_FEATURES]\n")
            pd.DataFrame({"Feature": unknown}).to_csv(f, index=False)

    print(f"[OK] Saved: {outpath.resolve()}")
    if unknown:
        print(
            f"[WARN] {len(unknown)} feature(s) were not found in FEATURE_MAP. They were labeled as Unknown."
        )


if __name__ == "__main__":
    main()