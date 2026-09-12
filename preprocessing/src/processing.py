# Copyright (C) [2025] [Aura Healthcare]
# Modificado por [André Coimbra] em [2025]
#
# https://pypi.org/project/hrv-analysis/
#
# Este programa é software livre; você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral GNU conforme publicada pela Free Software Foundation,
# na versão 3 da licença ou (a seu critério) qualquer versão posterior.
#
# Este programa é distribuído na esperança de que seja útil, mas SEM QUALQUER GARANTIA;
# sem mesmo a garantia implícita de COMERCIALIZAÇÃO ou ADEQUAÇÃO A UM DETERMINADO FIM.
# Consulte a Licença Pública Geral GNU para mais detalhes.
#
# Você deve ter recebido uma cópia da Licença Pública Geral GNU junto com este programa;
# se não, consulte <https://www.gnu.org/licenses/>.


import logging
import numpy as np
import pandas as pd

# from scipy.signal import medfilt
from config import (
    OUTLIER_THRESHOLD,
    MEDIAN_FILTER_KERNEL_SIZE,
    LOW_RRI,
    HIGH_RRI,
    POLICY,
)


def detect_outliers(rr_intervals, low_rri=LOW_RRI, high_rri=HIGH_RRI):
    """Function that detects RR-interval outliers."""
    rr_intervals = np.array(rr_intervals)

    # Máscara booleana para detectar valores fora dos limites
    outliers_mask = (rr_intervals < low_rri) | (rr_intervals > high_rri)

    logging.info(f"{np.sum(outliers_mask)} outlier(s) encontrado(s).")

    return outliers_mask


def remove_outliers(rr_intervals, low_rri=LOW_RRI, high_rri=HIGH_RRI):
    """
    Function that replace RR-interval outlier by nan.

    Parameters
    ---------
    rr_intervals : list
        raw signal extracted.
    low_rri : int
        lowest RrInterval to be considered plausible.
    high_rri : int
        highest RrInterval to be considered plausible.
    verbose : bool
        Print information about deleted outliers.

    Returns
    ---------
    rr_intervals_cleaned : list
        list of RR-intervals without outliers

    References
    ----------
    .. [1] O. Inbar, A. Oten, M. Scheinowitz, A. Rotstein, R. Dlin, R.Casaburi. Normal \
    cardiopulmonary responses during incremental exercise in 20-70-yr-old men.

    .. [2] W. C. Miller, J. P. Wallace, K. E. Eggert. Predicting max HR and the HR-VO2 relationship\
    for exercise prescription in obesity.

    .. [3] H. Tanaka, K. D. Monahan, D. R. Seals. Age-predictedmaximal heart rate revisited.

    .. [4] M. Gulati, L. J. Shaw, R. A. Thisted, H. R. Black, C. N. B.Merz, M. F. Arnsdorf. Heart \
    rate response to exercise stress testing in asymptomatic women.
    """

    # Conversion RrInterval to Heart rate ==> rri (ms) =  1000 / (bpm / 60)
    # rri 2000 => bpm 30 / rri 300 => bpm 200

    rr_intervals = np.array(rr_intervals)

    # Substitui outliers por NaN
    outliers = (rr_intervals < low_rri) | (rr_intervals > high_rri)
    rr_intervals_cleaned = np.where(outliers, np.nan, rr_intervals)

    # Coleta os valores considerados outliers
    outliers_list = rr_intervals[outliers].tolist()

    # Log dos resultados
    nan_count = np.sum(outliers)
    logging.info(f"{nan_count} outlier(s) removido(s).")
    if nan_count:
        logging.debug(f"Outlier(s): {outliers_list}")

    return rr_intervals_cleaned


# Função para detectar batimentos ectópicos
def detect_ectopic_beats(rr_intervals, threshold=0.2):
    rr_intervals = np.array(rr_intervals)
    rr_ratio = rr_intervals[1:] / rr_intervals[:-1]
    # Detecta batimentos ectópicos com base na variação do threshold
    ectopic_beats = np.full(rr_intervals.shape, False)
    ectopic_beats[1:] = np.abs(rr_ratio - 1) > (1 + threshold)

    logging.info(f"{np.sum(ectopic_beats)} batimentos(s) ectópico(s) encontrado(s).")

    return ectopic_beats


def remove_ectopic_beats(rr_intervals, threshold=0.2):
    """
    RR-intervals differing by more than the threshold from the one proceeding it are removed.

    Parameters
    ---------
    rr_intervals : list of RR-intervals
    threshold : percentage criteria of difference with previous RR-interval at which we consider that it is abnormal.

    Returns
    ---------
    nn_intervals : list of NN Interval

    References
    ----------
    - Kamath M.V., Fallen E.L.: Correction of the Heart Rate Variability Signal for Ectopics and Missing Beats, In: Malik M., Camm A.J.

    - Geometric Methods for Heart Rate Variability Assessment - Malik M et al
    """
    rr_intervals = np.array(rr_intervals)
    rr_ratio = rr_intervals[1:] / rr_intervals[:-1]

    # Detecta batimentos ectópicos com base no threshold
    ectopic_beats = np.full(rr_intervals.shape, False)
    ectopic_beats[1:] = np.abs(rr_ratio - 1) > (1 + threshold)

    # Cria array para armazenar os intervalos limpos
    nn_intervals = np.full(rr_intervals.shape, np.nan)
    nn_intervals[0] = rr_intervals[0]
    nn_intervals[1:] = np.where(~ectopic_beats[1:], rr_intervals[1:], np.nan)

    # Coleta os valores dos batimentos ectópicos removidos
    removed_beats = rr_intervals[ectopic_beats].tolist()

    # Log dos resultados
    outlier_count = len(removed_beats)
    logging.info(f"{outlier_count} batimento(s) ectópico(s) removido(s).")
    if outlier_count:
        logging.debug(f"Batimento(s) ectópico(s) removido(s): {removed_beats}")

    return nn_intervals


# Função para avaliar a qualidade do sinal
def evaluate_signal_quality(rr_intervals):
    logging.debug(f"Avaliando a qualidade do sinal")
    rr_intervals = rr_intervals * 1000  # Convertendo para milissegundos
    outliers = detect_outliers(rr_intervals)
    ectopic_beats = detect_ectopic_beats(rr_intervals)

    total_beats = len(rr_intervals)
    # Máscara booleana que seleciona apenas os batimentos válidos
    valid_beats = np.sum(~ectopic_beats & ~outliers)
    valid_percentage = valid_beats / total_beats

    ectopic_percentage = np.sum(ectopic_beats) / total_beats
    outlier_percentage = np.sum(outliers) / total_beats

    logging.debug(f"Percentual de batimentos válidos: {valid_percentage*100:.2f}%")
    logging.debug(f"Percentual de outliers: {outlier_percentage*100:.2f}%")
    logging.debug(f"Percentual de batimentos ectópicos: {ectopic_percentage*100:.2f}%")

    return valid_percentage


def interpolate_nan_values(
    rr_intervals,
    interpolation_method="linear",
    limit=None,
    limit_area=None,
    limit_direction="forward",
):
    """
    Function that interpolate Nan values with linear interpolation

    Parameters
    ---------
    rr_intervals : list
        RrIntervals list.
    interpolation_method : str
        Method used to interpolate Nan values of series.
    limit_area: str
        If limit is specified, consecutive NaNs will be filled with this restriction.
    limit_direction: str
        If limit is specified, consecutive NaNs will be filled in this direction.
    limit: int
        Maximum number of NaNs to fill consecutively.
    ---------
    interpolated_rr_intervals : list
        new list with outliers replaced by interpolated values.
    """
    if np.isnan(rr_intervals[0]):
        # Preenche os valores iniciais NaN com o primeiro valor não NaN
        start_idx = np.where(~np.isnan(rr_intervals))[0][0]
        rr_intervals[:start_idx] = rr_intervals[start_idx]

    # Converte para pandas Series e interpola valores NaN
    interpolated_rr_intervals = pd.Series(rr_intervals).interpolate(
        method=interpolation_method,
        limit=limit,
        limit_area=limit_area,
        limit_direction=limit_direction,
    )

    return interpolated_rr_intervals.values.tolist()


def get_nn_intervals(rr_intervals, low_rri=LOW_RRI, high_rri=HIGH_RRI):
    """Function that computes NN Intervals from RR-intervals."""
    logging.debug("Iniciando conversão do sinal de RRi para NNi")
    rr_intervals = rr_intervals * 1000  # Convertendo para milissegundos
    rr_intervals_without_outliers = remove_outliers(rr_intervals, low_rri, high_rri)
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals_without_outliers)
    rr_intervals_without_ectopic = remove_ectopic_beats(interpolated_rr_intervals)
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals_without_ectopic)
    nn_intervals = np.array(interpolated_nn_intervals) / 1000
    logging.debug("Conversão para NNi concluída")
    return nn_intervals


def truncate_rr_intervals(rr_intervals, target_duration, policy=POLICY):
    """
    Corta os intervalos RR para garantir que todos tenham a mesma duração.
    """
    logging.debug("Iniciando truncamento do sinal")

    accumulated_time = 0.0
    truncated_rr = []

    for rr in rr_intervals:
        if accumulated_time > target_duration:
            break
        truncated_rr.append(rr)
        accumulated_time += rr

    logging.debug(
        f"Tamanho inicial: {len(rr_intervals)}, tamanho após truncamento: {len(truncated_rr)}"
    )
    logging.debug(
        f"Tempo acumulado após truncamento: {accumulated_time:.2f} s (limite: {target_duration:.2f} s)"
    )

    return np.array(truncated_rr), accumulated_time
