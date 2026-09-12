import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--blr",
        default="outputs/blr_multivariate_results.csv",
        help="Path to multivariate BLR results",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--or-max",
        type=float,
        default=10.0,
        help="Upper Odds Ratio threshold",
    )
    parser.add_argument(
        "--or-min",
        type=float,
        default=0.10,
        help="Lower Odds Ratio threshold",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=0.05,
        help="Maximum p-value",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load multivariate BLR results
    df = pd.read_csv(args.blr)

    required_columns = {"Feature", "Odds_Ratio", "p_value"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"BLR file must contain columns: {required_columns}. "
            f"Found: {set(df.columns)}"
        )

    df["Odds_Ratio"] = pd.to_numeric(
        df["Odds_Ratio"],
        errors="coerce",
    )
    df["p_value"] = pd.to_numeric(
        df["p_value"],
        errors="coerce",
    )

    # Identify dominant clinical features
    dominant = df[
        (df["p_value"] <= args.p_max)
        & (
            (df["Odds_Ratio"] >= args.or_max)
            | (df["Odds_Ratio"] <= args.or_min)
        )
    ].sort_values(
        "Odds_Ratio",
        ascending=False,
    )

    excluded_features = dominant["Feature"].astype(str).tolist()

    output_path = outdir / "excluded_clinical_features.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        for feature in excluded_features:
            file.write(f"{feature}\n")

    print(f"Excluded features: {len(excluded_features)}")
    print(f"Results saved to: {output_path}")

    if excluded_features:
        print("Excluded:")
        for feature in excluded_features:
            print(f"  {feature}")
    else:
        print("No dominant clinical features found.")


if __name__ == "__main__":
    main()