from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


# -------------------------
# Mapping for outcome
# -------------------------
DEFAULT_GROUP_MAP = {
    "control": 0,
    "diabetic": 1,
    "Controle": 0,
    "Diabetes": 1,
    "CTRL": 0,
    "DM": 1,
    "T2DM": 1,
}


# -------------------------
# Load excluded columns
# -------------------------
def load_excluded_columns(
    data_path: str | Path, filename: str = "excluded_columns.txt"
) -> set[str]:
    """
    Load list of columns to exclude from analysis.
    Looks for the file in the same directory as the data file.
    """
    data_path = Path(data_path)
    exclude_file = data_path.parent / filename

    if not exclude_file.exists():
        print(f"[INFO] No excluded columns file found at {exclude_file}")
        return set()

    with open(exclude_file, "r", encoding="utf-8") as f:
        excluded = {line.strip() for line in f if line.strip()}

    # Extra safeguard against invisible leading/trailing whitespace
    excluded = {c.strip() for c in excluded}

    print(f"[INFO] Loaded {len(excluded)} excluded columns from {exclude_file}")
    return excluded


# -------------------------
# Main dataset loader
# -------------------------
def load_hrv_dataset(
    csv_path: str | Path,
    group_col: str = "Group",
    exclude_filename: str = "excluded_columns.txt",
    outdir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load HRV (or clinical) dataset and return (X, y), where:
      - X: numeric features, cleaned, with a priori excluded columns removed
      - y: binary outcome (0/1)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    # -------------------------
    # Read CSV (robust separator)
    # -------------------------
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path, sep=";")

    # -------------------------
    # Normalize columns names
    # -------------------------
    df.columns = df.columns.str.strip()

    # -------------------------
    # Load excluded columns (a priori)
    # -------------------------
    excluded_cols = load_excluded_columns(csv_path, exclude_filename)

    # -------------------------
    # Sex handling
    # -------------------------
    # df["Sexo"] = df["Sexo"].map({"F": 0, "M": 1})

    # -------------------------
    # Outcome handling
    # -------------------------
    if group_col not in df.columns:
        raise ValueError(
            f"Outcome column '{group_col}' not found. Columns: {df.columns.tolist()}"
        )

    y = df[group_col].copy()

    if y.dtype == "O":
        y = y.map(DEFAULT_GROUP_MAP)

    y = pd.to_numeric(y, errors="coerce")

    if y.isna().any():
        bad = df.loc[y.isna(), group_col].unique().tolist()
        raise ValueError(
            f"Outcome '{group_col}' has non-mappable values. Examples: {bad}. "
            f"Edit DEFAULT_GROUP_MAP in utils.py if needed."
        )

    y = y.astype(int)
    uniq = set(y.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Outcome must be binary 0/1. Found: {sorted(uniq)}")

    # -------------------------
    # Drop outcome + excluded columns (only those that actually exist)
    # -------------------------
    existing_excluded = sorted(c for c in excluded_cols if c in df.columns)

    if existing_excluded and outdir is not None:
        save_excluded_columns(existing_excluded, outdir)
    else:
        print("[INFO] No a priori excluded columns were present in the dataset")

    # Always drop outcome column
    print(f"[INFO] Dropping outcome column: '{group_col}'")

    X = df.drop(columns=[group_col] + existing_excluded)

    # -------------------------
    # Keep numeric only
    # -------------------------
    X = X.select_dtypes(include=[np.number]).copy()

    # Remove all-zero columns
    X = X.loc[:, (X != 0).any(axis=0)]

    # Coerce numeric and handle NaNs
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # Align indices explicitly
    X = X.loc[y.index].copy()

    # Remove constant columns
    nun = X.nunique(dropna=True)
    X = X.loc[:, nun > 1]

    return X, y


# -------------------------
# Utility helpers
# -------------------------
def save_list(path: str | Path, items: Iterable[str]) -> None:
    path = Path(path)
    path.write_text("\n".join(list(items)), encoding="utf-8")


def load_list(path: str | Path) -> List[str]:
    path = Path(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_excluded_columns(
    excluded_columns: Iterable[str],
    outdir: str | Path,
    filename: str = "excluded_columns_initial.txt",
) -> Path:
    """
    Save the list of actually excluded columns for traceability.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    excluded_columns = sorted(excluded_columns)

    print(f"[INFO] Dropping a priori excluded columns: {excluded_columns}")

    path = outdir / filename
    with open(path, "w", encoding="utf-8") as f:
        for col in excluded_columns:
            f.write(f"{col}\n")

    print(f"[INFO] Excluded columns saved to: {path}")
    return path


def spearman_corr(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="spearman")
