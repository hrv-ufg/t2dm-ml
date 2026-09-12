import argparse

import pandas as pd

from utils import ensure_dir, load_hrv_dataset, save_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="hrv_features.csv",
    )
    parser.add_argument(
        "--univar",
        default="outputs/blr_univariate_results.csv",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
    )

    args = parser.parse_args()
    outdir = ensure_dir(args.outdir)

    # Tag used in output filenames
    p_tag = f"p{int(args.p * 100):03d}"

    # Load data and univariate results
    X, y = load_hrv_dataset(args.data)

    univariate = pd.read_csv(args.univar)
    univariate = univariate[["Feature", "p_value"]].copy()
    univariate["p_value"] = pd.to_numeric(
        univariate["p_value"],
        errors="coerce",
    )

    # Select features based on the p-value threshold
    selected = univariate.loc[
        univariate["p_value"].notna()
        & (univariate["p_value"] < args.p),
        "Feature",
    ].tolist()

    # Keep the original order of the dataset columns
    selected = [
        feature for feature in X.columns
        if feature in selected
    ]

    # Save selected feature names
    features_path = outdir / f"selected_features_{p_tag}.txt"
    save_list(features_path, selected)

    # Save dataset with selected features
    X_selected = X[selected].copy()
    X_selected.insert(0, "Group", y.values)

    X_path = outdir / f"X_selected_{p_tag}.csv"
    X_selected.to_csv(X_path, index=False)

    print(f"Selected features: {len(selected)}")
    print(f"P-value threshold: {args.p}")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()