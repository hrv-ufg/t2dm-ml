import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from utils import ensure_dir, load_hrv_dataset, load_list


def fit_logit(y, X):
    X = sm.add_constant(X, has_constant="add")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)

        model = sm.Logit(y, X).fit(
            disp=0,
            method="lbfgs",
            maxiter=1000,
        )

    messages = [str(w.message) for w in caught]
    notes = []

    if any("overflow encountered in exp" in msg for msg in messages):
        notes.append("warn_overflow_exp")

    if any("divide by zero encountered in log" in msg for msg in messages):
        notes.append("warn_log0")

    if any("invalid value" in msg for msg in messages):
        notes.append("warn_invalid")

    return model, "|".join(notes)


def fit_regularized_logit(y, X, alpha=1.0):
    X = sm.add_constant(X, has_constant="add")

    return sm.Logit(y, X).fit_regularized(
        alpha=alpha,
        L1_wt=0.0,
        disp=0,
    )


def build_results_table(model, features):
    params = model.params
    p_values = getattr(model, "pvalues", None)

    try:
        confidence_intervals = model.conf_int()
    except (AttributeError, ValueError):
        confidence_intervals = None

    results = []

    for feature in ["const"] + features:
        coef = float(params[feature])
        odds_ratio = float(np.exp(coef))

        if (
            confidence_intervals is not None
            and feature in confidence_intervals.index
        ):
            ci = confidence_intervals.loc[feature]
            ci_low = float(np.exp(ci.iloc[0]))
            ci_high = float(np.exp(ci.iloc[1]))
        else:
            ci_low = np.nan
            ci_high = np.nan

        if p_values is not None and feature in p_values.index:
            p_value = float(p_values[feature])
        else:
            p_value = np.nan

        results.append({
            "Feature": feature,
            "Coef_Logit": coef,
            "Odds_Ratio": odds_ratio,
            "CI_95_low": ci_low,
            "CI_95_high": ci_high,
            "p_value": p_value,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="hrv_features.csv",
    )
    parser.add_argument(
        "--features",
        default="outputs/selected_features_pruned.txt",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength used in the fallback model",
    )

    args = parser.parse_args()
    outdir = ensure_dir(args.outdir)

    X, y = load_hrv_dataset(args.data)

    features = load_list(args.features)
    features = [
        feature for feature in X.columns
        if feature in features
    ]

    if not features:
        raise ValueError("No features to fit. Provide a valid features list.")

    X_fit = X[features].copy()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_fit),
        columns=features,
        index=X_fit.index,
    )

    # Try the standard logistic regression first
    fit_mode = "standard"
    warning_note = ""

    try:
        model, warning_note = fit_logit(y, X_scaled)

        if warning_note:
            fit_mode = "regularized_due_to_warnings"
            model = fit_regularized_logit(
                y,
                X_scaled,
                alpha=args.alpha,
            )

    except (PerfectSeparationError, np.linalg.LinAlgError):
        fit_mode = "regularized_due_to_exception"
        model = fit_regularized_logit(
            y,
            X_scaled,
            alpha=args.alpha,
        )

    except Exception:
        fit_mode = "regularized_due_to_exception"
        model = fit_regularized_logit(
            y,
            X_scaled,
            alpha=args.alpha,
        )

    # Save model results
    results = build_results_table(model, features)

    results_path = outdir / "blr_multivariate_results.csv"
    results.to_csv(results_path, index=False)

    summary = (
        f"Fit mode: {fit_mode}\n"
        f"Warnings: {warning_note if warning_note else 'none'}\n"
        f"Standardization: z-score (StandardScaler)\n"
        f"Regularization alpha (if used): {args.alpha}\n\n"
    )

    try:
        summary += model.summary2().as_text()
    except (AttributeError, ValueError):
        summary += "Model summary not available for this fit mode.\n"

    summary_path = outdir / "blr_multivariate_model_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Results saved to: {results_path}")
    print(f"Model summary saved to: {summary_path}")
    print(f"Fit mode: {fit_mode}")

    if warning_note:
        print(f"Standard fit warnings: {warning_note}")


if __name__ == "__main__":
    main()