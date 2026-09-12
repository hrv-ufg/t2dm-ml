import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from utils import ensure_dir, load_hrv_dataset


def zscore_safe(x):
    """Standardize a series when its standard deviation is non-zero."""
    x = pd.to_numeric(x, errors="coerce")

    mean = float(x.mean())
    std = float(x.std(ddof=0))

    if not np.isfinite(std) or std == 0:
        return x

    return (x - mean) / (std + 1e-12)


def run_univariate_blr(X, y):
    results = []

    for feature in X.columns:
        x = pd.to_numeric(X[feature], errors="coerce")

        if x.nunique(dropna=True) < 2:
            results.append({
                "Feature": feature,
                "Coef_Logit": np.nan,
                "Odds_Ratio": np.nan,
                "CI_95_low": np.nan,
                "CI_95_high": np.nan,
                "p_value": np.nan,
                "note": "degenerate",
            })
            continue

        x = zscore_safe(x)
        x.name = feature

        X_fit = sm.add_constant(x, has_constant="add")

        data = pd.concat([y, X_fit], axis=1).dropna()
        y_fit = data.iloc[:, 0]
        X_fit = data.iloc[:, 1:]

        if len(data) < 10:
            results.append({
                "Feature": feature,
                "Coef_Logit": np.nan,
                "Odds_Ratio": np.nan,
                "CI_95_low": np.nan,
                "CI_95_high": np.nan,
                "p_value": np.nan,
                "note": "too_few_rows",
            })
            continue

        notes = []

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)

                model = sm.Logit(y_fit, X_fit).fit(
                    disp=0,
                    method="lbfgs",
                    maxiter=300,
                )

                messages = [str(w.message) for w in caught]

                if any("overflow encountered in exp" in msg for msg in messages):
                    notes.append("warn_overflow_exp")

                if any("divide by zero encountered in log" in msg for msg in messages):
                    notes.append("warn_log0")

                if any("invalid value" in msg for msg in messages):
                    notes.append("warn_invalid")

            coef = float(model.params[feature])
            p_value = float(model.pvalues[feature])
            odds_ratio = float(np.exp(coef))

            try:
                ci = model.conf_int().loc[feature]
                ci_low = float(np.exp(ci.iloc[0]))
                ci_high = float(np.exp(ci.iloc[1]))
            except (KeyError, ValueError, IndexError):
                ci_low = np.nan
                ci_high = np.nan
                notes.append("ci_unavailable")

            results.append({
                "Feature": feature,
                "Coef_Logit": coef,
                "Odds_Ratio": odds_ratio,
                "CI_95_low": ci_low,
                "CI_95_high": ci_high,
                "p_value": p_value,
                "note": "|".join(notes),
            })

        except PerfectSeparationError:
            results.append({
                "Feature": feature,
                "Coef_Logit": np.nan,
                "Odds_Ratio": np.nan,
                "CI_95_low": np.nan,
                "CI_95_high": np.nan,
                "p_value": np.nan,
                "note": "perfect_separation",
            })

        except np.linalg.LinAlgError:
            results.append({
                "Feature": feature,
                "Coef_Logit": np.nan,
                "Odds_Ratio": np.nan,
                "CI_95_low": np.nan,
                "CI_95_high": np.nan,
                "p_value": np.nan,
                "note": "linalg_error",
            })

        except Exception as exc:
            results.append({
                "Feature": feature,
                "Coef_Logit": np.nan,
                "Odds_Ratio": np.nan,
                "CI_95_low": np.nan,
                "CI_95_high": np.nan,
                "p_value": np.nan,
                "note": f"error:{type(exc).__name__}",
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="hrv_features.csv",
        help="Path to the HRV feature dataset",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--exclude-cols",
        default="excluded_columns.txt",
        help="File containing columns to exclude",
    )

    args = parser.parse_args()
    outdir = ensure_dir(args.outdir)

    X, y = load_hrv_dataset(
        args.data,
        exclude_filename=args.exclude_cols,
        outdir=outdir,
    )

    results = run_univariate_blr(X, y)

    # Keep the original feature order
    results["Feature"] = pd.Categorical(
        results["Feature"],
        categories=list(X.columns),
        ordered=True,
    )
    results = results.sort_values("Feature")

    output_path = outdir / "blr_univariate_results.csv"
    results.to_csv(output_path, index=False)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()