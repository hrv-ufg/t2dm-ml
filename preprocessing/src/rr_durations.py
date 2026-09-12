from __future__ import annotations

import unicodedata

import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np


# -----------------------------
# Leitura tolerante de TXT
# -----------------------------
def load_txt(path: Path) -> np.ndarray:
    """
    Lê txt tolerante a:
      - separadores: whitespace, tab
      - comentários iniciados por '#'
    Se detectar vírgula/; no arquivo, tenta também CSV/; como fallback.
    """
    data = np.genfromtxt(
        path,
        delimiter=None,   # whitespace
        comments="#",
        dtype=float,
        invalid_raise=False,
    )

    # fallback se o arquivo parece usar ',' ou ';'
    txt = path.read_text(errors="ignore")
    if ("," in txt) or (";" in txt):
        for delim in [",", ";", "\t"]:
            d2 = np.genfromtxt(
                path,
                delimiter=delim,
                comments="#",
                dtype=float,
                invalid_raise=False,
            )
            # Se mudou o formato (ex.: passou de 1D pra 2D, ou aumentou colunas), adotamos
            if np.size(d2) > 0 and (np.ndim(d2) != np.ndim(data) or (hasattr(d2, "shape") and hasattr(data, "shape") and d2.shape != data.shape)):
                data = d2
                break

    if np.size(data) == 0:
        return np.array([], dtype=float)

    return np.array(data, dtype=float)


# -----------------------------
# Detecção do padrão [t, RR]
# -----------------------------
def detect_two_col_timestamp_rr(data: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Detecta padrão [timestamp, RR] em 2 colunas:
      - timestamp crescente (quase sempre)
      - diff(timestamp) ~ RR (para a maior parte dos pontos)
    Retorna (t, rr) se reconhecer; senão None.
    """
    if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 5:
        return None

    t = data[:, 0].astype(float)
    rr = data[:, 1].astype(float)

    m = np.isfinite(t) & np.isfinite(rr) & (rr > 0)
    t, rr = t[m], rr[m]
    if t.size < 5:
        return None

    dt = np.diff(t)
    if np.mean(dt >= 0) < 0.95:
        return None

    rr2 = rr[1:]
    dt2 = dt[: rr2.size]
    if rr2.size < 3:
        return None

    err = np.abs(dt2 - rr2)
    med_err = float(np.median(err))

    # tolerância: 30ms ou 10% do RR mediano
    tol = max(0.03, 0.10 * float(np.median(rr2)))

    if med_err <= tol:
        return t, rr

    return None


# -----------------------------
# Escolha de coluna RR (genérica)
# -----------------------------
def _robust_rr_score(x: np.ndarray) -> float:
    """
    Nota para quão 'RR-like' uma coluna é.
    Heurísticas: positivos, faixa plausível, variabilidade plausível.
    """
    x = x[np.isfinite(x)]
    if x.size < 10:
        return -1.0

    pos_frac = float(np.mean(x > 0))
    if pos_frac < 0.95:
        return -1.0

    med = float(np.median(x))
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    mad = float(np.median(np.abs(x - med)) + 1e-12)

    plausible_s = 0.25 <= med <= 2.5
    plausible_ms = 250 <= med <= 2500
    plaus = 1.0 if (plausible_s or plausible_ms) else 0.0

    rel_iqr = iqr / (abs(med) + 1e-12)
    var_ok = 0.005 <= rel_iqr <= 0.7

    score = 0.0
    score += 2.0 * plaus
    score += 1.5 * (1.0 if var_ok else 0.0)
    score += 1.0 * pos_frac
    score -= 0.5 if mad < 1e-6 else 0.0

    p99 = float(np.percentile(x, 99))
    if plausible_s and p99 > 10:
        score -= 1.0
    if plausible_ms and p99 > 10000:
        score -= 1.0

    return score


def pick_rr_column(data: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Se data for 1D -> RR é o próprio vetor.
    Se for 2D -> escolhe a coluna mais RR-like.
    """
    if data.ndim == 1:
        return data.astype(float), 0

    best_col = 0
    best_score = -math.inf
    for j in range(data.shape[1]):
        col = data[:, j].astype(float)
        score = _robust_rr_score(col)
        if score > best_score:
            best_score = score
            best_col = j

    rr = data[:, best_col].astype(float)
    return rr, best_col


# -----------------------------
# Inferência de unidade + duração
# -----------------------------
def infer_unit_and_minutes_from_rr(rr: np.ndarray) -> Tuple[Optional[str], float, int]:
    """
    Calcula duração por sum(RR) e tenta inferir unidade (ms ou s).
    Retorna (unit, minutes, n_valid).
    """
    rr = rr.astype(float)
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0]
    n = int(rr.size)
    if n < 2:
        return None, float("nan"), n

    med = float(np.median(rr))
    if med > 10:  # ms
        unit = "ms"
        total_seconds = float(np.sum(rr)) / 1000.0
    else:         # s
        unit = "s"
        total_seconds = float(np.sum(rr))

    return unit, total_seconds / 60.0, n


def duration_minutes(data: np.ndarray) -> Tuple[Optional[str], float, int, str, Optional[int]]:
    """
    Retorna:
      (unit, minutes, n_valid, method, rr_col_index)

    method:
      - 'timestamp+rr' (quando detecta [t, RR] em segundos)
      - 'sum(rr)'      (fallback)
    rr_col_index:
      - índice da coluna RR quando 'sum(rr)' em 2D
      - 1 quando 'timestamp+rr' (RR é a 2ª coluna)
      - 0 quando 1D
    """
    # 1) Caso especial: duas colunas [timestamp, RR]
    two_col = detect_two_col_timestamp_rr(data)
    if two_col is not None:
        t, rr = two_col
        dur_s = float(t[-1] - t[0] + rr[-1])
        return "s", dur_s / 60.0, int(rr.size), "timestamp+rr", 1

    # 2) Fallback: usa RR (1 coluna ou escolhe coluna RR-like)
    rr, rr_col = pick_rr_column(data)
    unit, minutes, n_valid = infer_unit_and_minutes_from_rr(rr)
    return unit, minutes, n_valid, "sum(rr)", rr_col


def sort_key(path: Path):
    # Nome do arquivo sem caminho
    name = path.name

    # Normaliza acentos (ç → c, á → a, etc.)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Ignora maiúsculas/minúsculas
    return name.casefold()


# -----------------------------
# Processamento da pasta
# -----------------------------
def process_folder(folder: Path, out_csv: Optional[Path] = None) -> List[Dict]:
    rows: List[Dict] = []

    for path in sorted(folder.rglob("*.txt"), key=sort_key):
        try:
            data = load_txt(path)

            if data.size == 0:
                rows.append(
                    {
                        "file": str(path.relative_to(folder)),
                        "duration_min": float("nan"),
                        "unit": "",
                        "n_valid": 0,
                        "method": "",
                        "rr_col_index": "",
                        "error": "empty file or unreadable",
                    }
                )
                continue

            unit, minutes, n_valid, method, rr_col_index = duration_minutes(data)

            rows.append(
                {
                    "file": str(path.relative_to(folder)),
                    "duration_min": minutes,
                    "unit": unit or "",
                    "n_valid": n_valid,
                    "method": method,
                    "rr_col_index": rr_col_index if rr_col_index is not None else "",
                    "error": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "file": str(path.relative_to(folder)),
                    "duration_min": float("nan"),
                    "unit": "",
                    "n_valid": 0,
                    "method": "",
                    "rr_col_index": "",
                    "error": repr(e),
                }
            )

    # salva CSV se solicitado
    if out_csv is not None:
        fieldnames = ["file", "duration_min", "unit", "n_valid", "method", "rr_col_index", "error"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                for k in fieldnames:
                    r.setdefault(k, "")
                w.writerow(r)

    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Varre uma pasta com .txt de intervalos RR (1 ou 2 colunas) e calcula duração (min)."
    )
    ap.add_argument("folder", type=str, help="Pasta contendo arquivos .txt (pode ter subpastas).")
    ap.add_argument("--csv", type=str, default="", help="Caminho para salvar CSV (opcional).")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Pasta inválida: {folder}")

    out_csv = Path(args.csv).expanduser().resolve() if args.csv else None
    rows = process_folder(folder, out_csv=out_csv)

    # imprime lista simples: arquivo -> duração
    for r in rows:
        if r.get("error"):
            print(f"{r['file']}: ERROR {r['error']}")
        else:
            print(
                f"{r['file']}: {r['duration_min']:.3f} min "
                f"(unit={r['unit']}, method={r['method']}, rr_col={r['rr_col_index']}, n={r['n_valid']})"
            )

    if out_csv:
        print(f"\nCSV salvo em: {out_csv}")


if __name__ == "__main__":
    main()