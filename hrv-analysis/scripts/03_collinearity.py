import argparse

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from utils import (
    ensure_dir,
    load_hrv_dataset,
    load_list,
    save_list,
    spearman_corr,
)


def compute_vif(X):
    X = X.astype(float)
    values = X.to_numpy()

    result = []

    for i, feature in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(values, i)
        except (ValueError, np.linalg.LinAlgError):
            vif = np.nan

        result.append({"Feature": feature, "VIF": vif})

    return pd.DataFrame(result)


def remove_correlated_features(X, p_values, threshold=0.60):
    corr = spearman_corr(X).abs()
    features = list(X.columns)
    selected = set(features)

    for i, feature_1 in enumerate(features):
        if feature_1 not in selected:
            continue

        for j in range(i + 1, len(features)):
            feature_2 = features[j]

            if feature_2 not in selected:
                continue

            rho = corr.iloc[i, j]

            if rho > threshold:
                p1 = p_values.get(feature_1, np.inf)
                p2 = p_values.get(feature_2, np.inf)

                print(
                    f"Correlation: {feature_1} - {feature_2} "
                    f"(rho={rho:.5f})"
                )

                if p1 <= p2:
                    selected.discard(feature_2)
                    print(f"Removing {feature_2} (p={p2:.10f})")
                else:
                    selected.discard(feature_1)
                    print(f"Removing {feature_1} (p={p1:.10f})")
                    break

    return [feature for feature in features if feature in selected]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="hrv_features.csv",
    )
    parser.add_argument(
        "--selected",
        default="outputs/selected_features_p020.txt",
    )
    parser.add_argument(
        "--univar",
        default="outputs/blr_univariate_results.csv",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
    )

    args = parser.parse_args()
    outdir = ensure_dir(args.outdir)

    X, y = load_hrv_dataset(args.data)

    selected = load_list(args.selected)
    selected = [feature for feature in X.columns if feature in selected]

    if not selected:
        raise ValueError(
            "No selected features found. Run 02_select_p020.py first."
        )

    X_selected = X[selected].copy()

    # Spearman correlation
    corr = spearman_corr(X_selected)
    corr.to_csv(
        outdir / "spearman_corr_selected.csv",
        index=True,
    )

    # Variance inflation factor
    vif = compute_vif(X_selected)
    vif.to_csv(
        outdir / "vif_selected.csv",
        index=False,
    )

    # Univariate logistic regression results
    univariate = pd.read_csv(args.univar)
    univariate["p_value"] = pd.to_numeric(
        univariate["p_value"],
        errors="coerce",
    )

    p_values = dict(
        zip(univariate["Feature"], univariate["p_value"])
    )

    # Remove highly correlated features
    selected_pruned = remove_correlated_features(
        X_selected,
        p_values,
        args.rho,
    )

    save_list(
        outdir / "selected_features_pruned.txt",
        selected_pruned,
    )

    X_pruned = X[selected_pruned].copy()
    X_pruned.insert(0, "Group", y.values)

    X_pruned.to_csv(
        outdir / "X_selected_pruned.csv",
        index=False,
    )

    print(f"Selected features: {len(selected)}")
    print(f"Remaining features: {len(selected_pruned)}")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    main()