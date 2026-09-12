import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from utils import ensure_dir, load_hrv_dataset, load_list


MODEL_ORDER = [
    "LogReg(L2)",
    "NaiveBayes(Gaussian)",
    "SVM(RBF)",
    "KNN",
    "RandomForest",
    "ExtraTrees(modern)",
    "GradBoost",
    "AdaBoost",
    "HistGB(modern)",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "ANN(MLP)",
]


def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return float(tn / (tn + fp)) if tn + fp > 0 else np.nan


def sensitivity(y_true, y_pred):
    _, _, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return float(tp / (tp + fn)) if tp + fn > 0 else np.nan


def get_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        if probabilities is not None and probabilities.ndim == 2:
            if probabilities.shape[1] >= 2:
                return probabilities[:, 1]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).ravel()
        return (scores - scores.min()) / (
            scores.max() - scores.min() + 1e-12
        )

    return None


def make_scaler(kind):
    if kind == "none":
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()

    raise ValueError(
        "--scaler must be one of: auto, none, standard, minmax"
    )


def with_scaler(model, scaler):
    return Pipeline([
        ("scaler", scaler),
        ("clf", model),
    ])


def build_models(seed):
    models = [
        (
            "LogReg(L2)",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            ),
            True,
        ),
        (
            "NaiveBayes(Gaussian)",
            GaussianNB(),
            True,
        ),
        (
            "SVM(RBF)",
            SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ),
            True,
        ),
        (
            "KNN",
            KNeighborsClassifier(n_neighbors=7),
            True,
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=700,
                random_state=seed,
                class_weight="balanced_subsample",
                n_jobs=-1,
            ),
            False,
        ),
        (
            "GradBoost",
            GradientBoostingClassifier(random_state=seed),
            False,
        ),
        (
            "AdaBoost",
            AdaBoostClassifier(random_state=seed),
            False,
        ),
        (
            "ExtraTrees(modern)",
            ExtraTreesClassifier(
                n_estimators=700,
                random_state=seed,
                class_weight="balanced",
                n_jobs=-1,
            ),
            False,
        ),
        (
            "HistGB(modern)",
            HistGradientBoostingClassifier(random_state=seed),
            False,
        ),
    ]

    try:
        from xgboost import XGBClassifier

        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
                False,
            )
        )
    except Exception as exc:
        tqdm.write(
            f"[SKIP] XGBoost import failed: {type(exc).__name__}: {exc}"
        )

    try:
        from lightgbm import LGBMClassifier

        models.append(
            (
                "LightGBM",
                LGBMClassifier(
                    n_estimators=1200,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                ),
                False,
            )
        )
    except Exception as exc:
        tqdm.write(
            f"[SKIP] LightGBM import failed: {type(exc).__name__}: {exc}"
        )

    try:
        from catboost import CatBoostClassifier

        models.append(
            (
                "CatBoost",
                CatBoostClassifier(
                    iterations=1200,
                    learning_rate=0.03,
                    depth=6,
                    loss_function="Logloss",
                    random_seed=seed,
                    verbose=False,
                ),
                False,
            )
        )
    except Exception as exc:
        tqdm.write(
            f"[SKIP] CatBoost import failed: {type(exc).__name__}: {exc}"
        )

    return models


def evaluate_model(
    name,
    model,
    X,
    y,
    n_splits=5,
    seed=1001,
):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    metrics = {
        "accuracy": [],
        "precision": [],
        "f1": [],
        "auc": [],
        "sensitivity": [],
        "specificity": [],
    }

    confusion_total = np.zeros((2, 2), dtype=np.int64)

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        probabilities = get_probabilities(model, X_test)

        metrics["accuracy"].append(
            accuracy_score(y_test, y_pred)
        )
        metrics["precision"].append(
            precision_score(y_test, y_pred, zero_division=0)
        )
        metrics["f1"].append(
            f1_score(y_test, y_pred, zero_division=0)
        )
        metrics["auc"].append(
            roc_auc_score(y_test, probabilities)
            if probabilities is not None
            else np.nan
        )
        metrics["sensitivity"].append(
            sensitivity(y_test, y_pred)
        )
        metrics["specificity"].append(
            specificity(y_test, y_pred)
        )

        confusion_total += confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

    tn, fp, fn, tp = confusion_total.ravel()
    total = int(tn + fp + fn + tp)

    summary = {
        "model": name,
        "n_splits": int(n_splits),
        "accuracy_mean": float(np.nanmean(metrics["accuracy"])),
        "accuracy_std": float(np.nanstd(metrics["accuracy"], ddof=1)),
        "precision_mean": float(np.nanmean(metrics["precision"])),
        "precision_std": float(np.nanstd(metrics["precision"], ddof=1)),
        "f1_mean": float(np.nanmean(metrics["f1"])),
        "f1_std": float(np.nanstd(metrics["f1"], ddof=1)),
        "auc_mean": float(np.nanmean(metrics["auc"])),
        "auc_std": float(np.nanstd(metrics["auc"], ddof=1)),
        "sensitivity_mean": float(np.nanmean(metrics["sensitivity"])),
        "sensitivity_std": float(
            np.nanstd(metrics["sensitivity"], ddof=1)
        ),
        "specificity_mean": float(np.nanmean(metrics["specificity"])),
        "specificity_std": float(
            np.nanstd(metrics["specificity"], ddof=1)
        ),
    }

    confusion = {
        "model": name,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": total,
        "tn_rate": float(tn / total) if total > 0 else np.nan,
        "fp_rate": float(fp / total) if total > 0 else np.nan,
        "fn_rate": float(fn / total) if total > 0 else np.nan,
        "tp_rate": float(tp / total) if total > 0 else np.nan,
    }

    return summary, confusion


def make_ann_scaler(kind):
    if kind == "none":
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "minmax":
        return MinMaxScaler()

    raise ValueError(
        "--ann-scaler must be one of: none, standard, minmax"
    )


def parse_hidden_layers(hidden):
    sizes = tuple(
        int(value.strip())
        for value in hidden.split(",")
        if value.strip()
    )

    if not sizes:
        raise ValueError(
            "--ann-hidden must define at least one layer, "
            "e.g. '64' or '64,32'."
        )

    if any(size <= 0 for size in sizes):
        raise ValueError(
            "--ann-hidden layer sizes must be positive integers."
        )

    return sizes


def build_ann(
    seed,
    hidden,
    max_iter,
    early_stopping,
    alpha,
    activation,
    solver,
    learning_rate,
    learning_rate_init,
    batch_size,
    n_iter_no_change,
    validation_fraction,
    tol,
):
    return MLPClassifier(
        hidden_layer_sizes=parse_hidden_layers(hidden),
        activation=activation,
        solver=solver,
        alpha=float(alpha),
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=float(learning_rate_init),
        max_iter=int(max_iter),
        shuffle=True,
        random_state=int(seed),
        early_stopping=bool(early_stopping),
        n_iter_no_change=int(n_iter_no_change),
        validation_fraction=float(validation_fraction),
        tol=float(tol),
    )


def get_fitted_mlp(estimator):
    if hasattr(estimator, "named_steps"):
        return estimator.named_steps.get("clf")

    return estimator


def safe_float(value):
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def print_ann_report(
    estimator,
    *,
    fold,
    max_iter,
    scaler,
    X_train_shape,
    X_test_shape,
    y_train_pos,
    y_train_neg,
    y_test_pos,
    y_test_neg,
):
    mlp = get_fitted_mlp(estimator)

    if mlp is None:
        print(
            f"[ANN][Fold {fold}] "
            "Could not access MLPClassifier."
        )
        return

    print("\n" + "-" * 96)
    print(f"[ANN][Fold {fold}] FIT REPORT")
    print(
        f"[ANN] Data split: Xtr={X_train_shape} Xte={X_test_shape} | "
        f"ytr pos={y_train_pos} neg={y_train_neg} | "
        f"yte pos={y_test_pos} neg={y_test_neg}"
    )
    print(f"[ANN] scaler_used = {scaler}")

    print("[ANN] Hyperparameters:")
    print(f"  hidden_layer_sizes = {mlp.hidden_layer_sizes}")
    print(f"  activation         = {mlp.activation}")
    print(f"  solver             = {mlp.solver}")
    print(f"  alpha (L2)         = {mlp.alpha}")
    print(f"  learning_rate      = {mlp.learning_rate}")
    print(f"  learning_rate_init = {mlp.learning_rate_init}")
    print(f"  batch_size         = {mlp.batch_size}")
    print(f"  max_iter           = {max_iter}")
    print(f"  tol                = {mlp.tol}")
    print(f"  shuffle             = {mlp.shuffle}")
    print(f"  random_state        = {mlp.random_state}")

    print("[ANN] Early stopping:")
    print(f"  early_stopping      = {mlp.early_stopping}")
    print(
        f"  validation_fraction = "
        f"{getattr(mlp, 'validation_fraction', None)}"
    )
    print(
        f"  n_iter_no_change    = "
        f"{getattr(mlp, 'n_iter_no_change', None)}"
    )

    print("[ANN] Training outcome:")
    print(f"  n_iter_       = {getattr(mlp, 'n_iter_', None)}")
    print(f"  loss_         = {getattr(mlp, 'loss_', None)}")
    print(f"  n_layers_     = {getattr(mlp, 'n_layers_', None)}")
    print(f"  n_outputs_    = {getattr(mlp, 'n_outputs_', None)}")
    print(
        f"  out_activation_ = "
        f"{getattr(mlp, 'out_activation_', None)}"
    )
    print(f"  classes_      = {getattr(mlp, 'classes_', None)}")

    try:
        shapes = [weights.shape for weights in mlp.coefs_]
        print(f"  coefs_ shapes = {shapes}")
    except Exception:
        pass

    n_iter = getattr(mlp, "n_iter_", None)

    if mlp.early_stopping and n_iter is not None:
        stopped_early = int(n_iter) < int(max_iter)

        print("[ANN] Early stopping status:")
        print(f"  stopped_early  = {stopped_early}")
        print(f"  stop_iteration = {int(n_iter)}")

        scores = getattr(mlp, "validation_scores_", None)

        if isinstance(scores, (list, tuple)) and len(scores) > 0:
            scores = np.asarray(scores, dtype=float)

            if np.isfinite(scores).any():
                best_iteration = int(np.nanargmax(scores)) + 1
                best_score = float(np.nanmax(scores))

                print(f"  best_iteration = {best_iteration}")
                print(f"  best_val_score = {best_score}")
        else:
            print("  validation_scores_ = (not available)")

    print("-" * 96 + "\n")


@dataclass
class AnnRunConfig:
    script: str
    timestamp_local: str
    data_path: str
    features_path: str
    outdir: str
    exclude_path_used: str
    n_samples: int
    n_features_total_in_csv: int
    n_features_selected: int
    n_features_omitted: int
    n_features_used: int
    class_counts: Dict[str, int]
    n_splits: int
    seed: int
    scaler: str
    hidden: str
    activation: str
    solver: str
    alpha: float
    learning_rate: str
    learning_rate_init: float
    batch_size: str
    max_iter: int
    tol: float
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    verbose: bool


def save_ann_config(outdir, config):
    path = outdir / "ml_ann_run_config.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            asdict(config),
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved: {path}")


def evaluate_ann_cv(
    estimator,
    X,
    y,
    *,
    n_splits,
    seed,
    max_iter,
    scaler,
    verbose,
):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    metrics = {
        name: []
        for name in [
            "accuracy",
            "precision",
            "f1",
            "auc",
            "sensitivity",
            "specificity",
        ]
    }

    confusion_total = np.zeros((2, 2), dtype=np.int64)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        if verbose:
            print_ann_report(
                estimator,
                fold=fold,
                max_iter=max_iter,
                scaler=scaler,
                X_train_shape=tuple(X_train.shape),
                X_test_shape=tuple(X_test.shape),
                y_train_pos=int((y_train == 1).sum()),
                y_train_neg=int((y_train == 0).sum()),
                y_test_pos=int((y_test == 1).sum()),
                y_test_neg=int((y_test == 0).sum()),
            )

        y_pred = estimator.predict(X_test)
        probabilities = get_probabilities(estimator, X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )
        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )
        auc = (
            roc_auc_score(y_test, probabilities)
            if probabilities is not None
            else np.nan
        )
        sens = sensitivity(y_test, y_pred)
        spec = specificity(y_test, y_pred)

        metrics["accuracy"].append(acc)
        metrics["precision"].append(precision)
        metrics["f1"].append(f1)
        metrics["auc"].append(auc)
        metrics["sensitivity"].append(sens)
        metrics["specificity"].append(spec)

        confusion_fold = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )
        confusion_total += confusion_fold

        tn, fp, fn, tp = confusion_fold.ravel()

        mlp = get_fitted_mlp(estimator)
        n_iter = getattr(mlp, "n_iter_", None)
        loss = getattr(mlp, "loss_", None)

        early_stopping = bool(
            getattr(mlp, "early_stopping", False)
        )

        stopped_early = None
        best_iteration = None
        best_val_score = None

        if early_stopping and n_iter is not None:
            stopped_early = int(n_iter) < int(max_iter)

            scores = getattr(mlp, "validation_scores_", None)

            if isinstance(scores, (list, tuple)) and len(scores) > 0:
                scores = np.asarray(scores, dtype=float)

                if np.isfinite(scores).any():
                    best_iteration = int(np.nanargmax(scores)) + 1
                    best_val_score = float(np.nanmax(scores))

        fold_results.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "y_train_pos": int((y_train == 1).sum()),
                "y_train_neg": int((y_train == 0).sum()),
                "y_test_pos": int((y_test == 1).sum()),
                "y_test_neg": int((y_test == 0).sum()),
                "fit_time_seconds": float(fit_time),
                "accuracy": float(acc),
                "precision": float(precision),
                "f1": float(f1),
                "auc": float(auc) if np.isfinite(auc) else np.nan,
                "sensitivity": (
                    float(sens) if np.isfinite(sens) else np.nan
                ),
                "specificity": (
                    float(spec) if np.isfinite(spec) else np.nan
                ),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "n_iter_": (
                    int(n_iter) if n_iter is not None else np.nan
                ),
                "loss_": safe_float(loss),
                "early_stopping_enabled": early_stopping,
                "stopped_early": (
                    bool(stopped_early)
                    if stopped_early is not None
                    else np.nan
                ),
                "best_iteration": (
                    int(best_iteration)
                    if best_iteration is not None
                    else np.nan
                ),
                "best_val_score": (
                    float(best_val_score)
                    if best_val_score is not None
                    else np.nan
                ),
            }
        )

        if not verbose:
            print(
                f"[ANN][Fold {fold}/{n_splits}] "
                f"acc={acc:.4f} "
                f"f1={f1:.4f} "
                f"auc={auc:.4f} "
                f"sens={sens:.4f} "
                f"spec={spec:.4f} "
                f"fit_time={fit_time:.2f}s"
            )

    summary = {
        "model": "ANN(MLP)",
        "n_splits": int(n_splits),
        "accuracy_mean": float(np.nanmean(metrics["accuracy"])),
        "accuracy_std": float(np.nanstd(metrics["accuracy"], ddof=1)),
        "precision_mean": float(np.nanmean(metrics["precision"])),
        "precision_std": float(np.nanstd(metrics["precision"], ddof=1)),
        "f1_mean": float(np.nanmean(metrics["f1"])),
        "f1_std": float(np.nanstd(metrics["f1"], ddof=1)),
        "auc_mean": float(np.nanmean(metrics["auc"])),
        "auc_std": float(np.nanstd(metrics["auc"], ddof=1)),
        "sensitivity_mean": float(
            np.nanmean(metrics["sensitivity"])
        ),
        "sensitivity_std": float(
            np.nanstd(metrics["sensitivity"], ddof=1)
        ),
        "specificity_mean": float(
            np.nanmean(metrics["specificity"])
        ),
        "specificity_std": float(
            np.nanstd(metrics["specificity"], ddof=1)
        ),
        "scaler_used": scaler,
    }

    tn, fp, fn, tp = confusion_total.ravel()
    total = int(tn + fp + fn + tp)

    confusion = {
        "model": "ANN(MLP)",
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": total,
        "tn_rate": float(tn / total) if total > 0 else np.nan,
        "fp_rate": float(fp / total) if total > 0 else np.nan,
        "fn_rate": float(fn / total) if total > 0 else np.nan,
        "tp_rate": float(tp / total) if total > 0 else np.nan,
        "scaler_used": scaler,
    }

    return summary, confusion, pd.DataFrame(fold_results)


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
        "--splits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1001,
    )

    parser.add_argument(
        "--scaler",
        choices=["auto", "none", "standard", "minmax"],
        default="auto",
        help="Scaling strategy for classic models.",
    )

    parser.add_argument(
        "--exclude",
        default=None,
        help=(
            "Path to excluded_clinical_features.txt. "
            "If omitted, <outdir>/excluded_clinical_features.txt "
            "is used when available."
        ),
    )

    parser.add_argument(
        "--ann-scaler",
        choices=["none", "standard", "minmax"],
        default="standard",
    )
    parser.add_argument(
        "--ann-hidden",
        default="64,32",
        help="Hidden layer sizes, e.g. '64' or '64,32'.",
    )
    parser.add_argument(
        "--ann-activation",
        choices=["relu", "tanh", "logistic"],
        default="relu",
    )
    parser.add_argument(
        "--ann-solver",
        choices=["adam", "lbfgs", "sgd"],
        default="adam",
    )
    parser.add_argument(
        "--ann-alpha",
        type=float,
        default=1e-4,
        help="L2 regularization strength.",
    )
    parser.add_argument(
        "--ann-learning-rate",
        choices=["constant", "invscaling", "adaptive"],
        default="adaptive",
    )
    parser.add_argument(
        "--ann-learning-rate-init",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--ann-batch-size",
        default="auto",
    )
    parser.add_argument(
        "--ann-max-iter",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--ann-tol",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--ann-early-stopping",
        action="store_true",
    )
    parser.add_argument(
        "--ann-validation-fraction",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--ann-n-iter-no-change",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--ann-verbose",
        action="store_true",
        help="Print detailed ANN information for each fold.",
    )

    args = parser.parse_args()
    outdir = ensure_dir(args.outdir)

    X, y = load_hrv_dataset(args.data)

    selected_features = load_list(args.features)
    selected_features = [
        feature
        for feature in X.columns
        if feature in set(selected_features)
    ]

    if not selected_features:
        raise ValueError(
            "No features to evaluate. Provide a valid features list."
        )

    exclude_path = (
        Path(args.exclude)
        if args.exclude
        else outdir / "excluded_clinical_features.txt"
    )

    if exclude_path.exists():
        excluded = set(load_list(str(exclude_path)))

        print(
            f"[ML] Exclusion list loaded: {exclude_path} "
            f"-> {len(excluded)} features"
        )
    else:
        excluded = set()
        print(
            f"[ML] No exclusion file found at {exclude_path} "
            "-> using all selected features"
        )

    features_used = [
        feature
        for feature in selected_features
        if feature not in excluded
    ]
    features_omitted = [
        feature
        for feature in selected_features
        if feature in excluded
    ]

    if features_omitted:
        print(
            f"[ML] Omitted {len(features_omitted)} features:"
        )
        for feature in features_omitted:
            print(f"  - {feature}")

    if not features_used:
        raise ValueError(
            "All selected features were omitted. "
            "Adjust the exclusion criteria."
        )

    used_path = outdir / "ml_features_used.txt"
    used_path.write_text(
        "\n".join(features_used) + "\n",
        encoding="utf-8",
    )

    omitted_path = outdir / "ml_features_omitted.txt"
    omitted_path.write_text(
        "\n".join(features_omitted)
        + ("\n" if features_omitted else ""),
        encoding="utf-8",
    )

    print(
        f"[ML] Features used: {len(features_used)} -> {used_path}"
    )
    print(
        f"[ML] Features omitted: {len(features_omitted)} -> {omitted_path}"
    )

    X_fit = X[features_used].copy()

    rows = []
    confusion_matrices = []
    timing = []

    models = build_models(seed=args.seed)

    for name, model, needs_scaling in tqdm(
        models,
        desc="Models",
        unit="model",
    ):
        tqdm.write(
            f"[ML] Training and evaluating: {name}"
        )

        start = time.perf_counter()

        if args.scaler == "none":
            estimator = model
            scaler_used = "none"

        elif args.scaler == "auto":
            if needs_scaling:
                estimator = with_scaler(
                    model,
                    StandardScaler(),
                )
                scaler_used = "standard(auto)"
            else:
                estimator = model
                scaler_used = "none(auto)"

        else:
            scaler = make_scaler(args.scaler)

            if needs_scaling:
                estimator = with_scaler(model, scaler)
                scaler_used = args.scaler
            else:
                estimator = model
                scaler_used = "none(tree)"

        row, cm = evaluate_model(
            name,
            estimator,
            X_fit,
            y,
            n_splits=args.splits,
            seed=args.seed,
        )

        elapsed = time.perf_counter() - start

        row["scaler_used"] = scaler_used
        row["time_seconds"] = float(elapsed)

        cm["scaler_used"] = scaler_used

        rows.append(row)
        confusion_matrices.append(cm)

        timing.append(
            {
                "model": name,
                "time_seconds": float(elapsed),
                "scaler_used": scaler_used,
                "n_splits": int(args.splits),
            }
        )

        tqdm.write(
            f"[ML] Finished {name} in {elapsed:.2f} seconds"
        )

    ann = build_ann(
        seed=args.seed,
        hidden=args.ann_hidden,
        max_iter=args.ann_max_iter,
        early_stopping=args.ann_early_stopping,
        alpha=args.ann_alpha,
        activation=args.ann_activation,
        solver=args.ann_solver,
        learning_rate=args.ann_learning_rate,
        learning_rate_init=args.ann_learning_rate_init,
        batch_size=args.ann_batch_size,
        n_iter_no_change=args.ann_n_iter_no_change,
        validation_fraction=args.ann_validation_fraction,
        tol=args.ann_tol,
    )

    ann_scaler = make_ann_scaler(args.ann_scaler)

    if ann_scaler is None:
        ann_estimator = ann
        ann_scaler_used = "none"
    else:
        ann_estimator = Pipeline([
            ("scaler", ann_scaler),
            ("clf", ann),
        ])
        ann_scaler_used = args.ann_scaler

    class_counts = {
        "n_total": int(len(y)),
        "n_class_0": int((y == 0).sum()),
        "n_class_1": int((y == 1).sum()),
    }

    ann_config = AnnRunConfig(
        script="06_ml_compare.py",
        timestamp_local=time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        data_path=str(Path(args.data).resolve()),
        features_path=str(Path(args.features).resolve()),
        outdir=str(Path(args.outdir).resolve()),
        exclude_path_used=(
            str(exclude_path.resolve())
            if exclude_path.exists()
            else str(exclude_path)
        ),
        n_samples=int(len(y)),
        n_features_total_in_csv=int(X.shape[1]),
        n_features_selected=int(len(selected_features)),
        n_features_omitted=int(len(features_omitted)),
        n_features_used=int(len(features_used)),
        class_counts=class_counts,
        n_splits=int(args.splits),
        seed=int(args.seed),
        scaler=str(args.ann_scaler),
        hidden=str(args.ann_hidden),
        activation=str(args.ann_activation),
        solver=str(args.ann_solver),
        alpha=float(args.ann_alpha),
        learning_rate=str(args.ann_learning_rate),
        learning_rate_init=float(args.ann_learning_rate_init),
        batch_size=str(args.ann_batch_size),
        max_iter=int(args.ann_max_iter),
        tol=float(args.ann_tol),
        early_stopping=bool(args.ann_early_stopping),
        validation_fraction=float(args.ann_validation_fraction),
        n_iter_no_change=int(args.ann_n_iter_no_change),
        verbose=bool(args.ann_verbose),
    )

    save_ann_config(outdir, ann_config)

    tqdm.write(
        f"[ANN] Starting CV: splits={args.splits}, "
        f"seed={args.seed}, scaler={ann_scaler_used}, "
        f"hidden={args.ann_hidden}, "
        f"alpha={args.ann_alpha}, "
        f"early_stopping={args.ann_early_stopping}"
    )

    start = time.perf_counter()

    ann_summary, ann_cm, ann_folds = evaluate_ann_cv(
        ann_estimator,
        X_fit,
        y,
        n_splits=args.splits,
        seed=args.seed,
        max_iter=args.ann_max_iter,
        scaler=ann_scaler_used,
        verbose=args.ann_verbose,
    )

    ann_time = time.perf_counter() - start

    ann_fold_path = outdir / "ml_ann_fold_log.csv"
    ann_folds.to_csv(ann_fold_path, index=False)
    print(f"Saved: {ann_fold_path}")

    ann_summary["time_seconds"] = float(ann_time)
    ann_cm["scaler_used"] = ann_scaler_used

    rows.append(ann_summary)
    confusion_matrices.append(ann_cm)

    timing.append(
        {
            "model": "ANN(MLP)",
            "time_seconds": float(ann_time),
            "scaler_used": ann_scaler_used,
            "n_splits": int(args.splits),
        }
    )

    results = pd.DataFrame(rows)

    present_models = [
        model
        for model in MODEL_ORDER
        if model in set(results["model"].astype(str))
    ]

    results["model"] = pd.Categorical(
        results["model"],
        categories=present_models,
        ordered=True,
    )
    results = results.sort_values("model")

    results_path = outdir / "ml_cv_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    timing_df = pd.DataFrame(timing)
    present_models = [
        model
        for model in MODEL_ORDER
        if model in set(timing_df["model"].astype(str))
    ]

    timing_df["model"] = pd.Categorical(
        timing_df["model"],
        categories=present_models,
        ordered=True,
    )
    timing_df = timing_df.sort_values("model")

    timing_path = outdir / "ml_timing_log.csv"
    timing_df.to_csv(timing_path, index=False)
    print(f"Saved: {timing_path}")

    confusion_df = pd.DataFrame(confusion_matrices)
    present_models = [
        model
        for model in MODEL_ORDER
        if model in set(confusion_df["model"].astype(str))
    ]

    confusion_df["model"] = pd.Categorical(
        confusion_df["model"],
        categories=present_models,
        ordered=True,
    )
    confusion_df = confusion_df.sort_values("model")

    confusion_path = outdir / "ml_confusion_matrices.csv"
    confusion_df.to_csv(confusion_path, index=False)
    print(f"Saved: {confusion_path}")


if __name__ == "__main__":
    main()