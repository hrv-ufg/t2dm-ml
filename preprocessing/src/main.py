import os
import logging
import numpy as np
import shutil
from file_io import load_rr_intervals, save_rr_intervals, save_removed_files
from statistics_dir import (
    generate_statistics_report,
    generate_duration_and_quality_file_report,
)
from processing import (
    get_nn_intervals,
    evaluate_signal_quality,
    truncate_rr_intervals,
)
from utils import list_rr_files, get_output_path, ask_user, get_relative_output_path
from config import (
    OUTPUT_DIR,
    DENOISED_OUTPUT_DIR,
    TRUNCATED_OUTPUT_DIR,
    CONTROL_DIR,
    TEST_DIR,
    POLICY,
    LOW_RRI,
    HIGH_RRI,
    CLIP_START_LENGTH,
    QUALITY_THRESHOLD,
    MIN_LENGTH_SEG,
)
from logging_config import setup_logging


def empty_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)


def create_truncated_output_dirs(truncatedOutputDir, control_dir, test_dir):
    os.makedirs(
        os.path.join(truncatedOutputDir, os.path.basename(control_dir)),
        exist_ok=True,
    )
    os.makedirs(
        os.path.join(truncatedOutputDir, os.path.basename(test_dir)),
        exist_ok=True,
    )


def process_data(
    control_dir=CONTROL_DIR,
    test_dir=TEST_DIR,
    min_length_seg=MIN_LENGTH_SEG,
    policy=POLICY,
    clip_start_length=CLIP_START_LENGTH,
    quality_threshold=QUALITY_THRESHOLD,
):

    logging.debug(
        f"Iniciando o processamento dos diretórios: '{control_dir}' e '{test_dir}'"
    )
    # Limpa o diretório DENOISED_OUTPUT_DIR
    empty_directory(DENOISED_OUTPUT_DIR)

    files = []
    min_length = float("inf")
    min_file = None

    for directory in [control_dir, test_dir]:
        files.extend(list_rr_files(directory))

    if not files:
        logging.warning(
            f"Nenhum arquivo encontrado nos diretórios '{control_dir}' e '{test_dir}'"
        )
        return

    removed_low_quality = {}

    for file in files[:]:
        rr_intervals = load_rr_intervals(file)

        if rr_intervals is not None:

            logging.info(f"Removendo os {clip_start_length} primeiros RRis do arquivo")
            rr_intervals = rr_intervals[clip_start_length:]

            signal_quality = evaluate_signal_quality(rr_intervals)
            if signal_quality < quality_threshold:
                files.remove(file)
                # Salva o nome do arquivo e a qualidade no dicionário
                removed_low_quality[file] = round(signal_quality * 100, 2)
                logging.warning(
                    f"Sinal com baixa qualidade ({signal_quality*100:.2f}%), removido da análise: '{file}'"
                )
            else:
                logging.info(
                    f"Sinal com boa qualidade ({signal_quality*100:.2f}%), mantido na análise: '{file}'"
                )
                rr_cleaned = get_nn_intervals(rr_intervals, LOW_RRI, HIGH_RRI)

                length = np.sum(rr_cleaned)
                if length < min_length:
                    min_length = length
                    min_file = file

                output_file = get_output_path(
                    file, DENOISED_OUTPUT_DIR, control_dir, test_dir, "_denoised.txt"
                )
                save_rr_intervals(output_file, rr_cleaned)

    save_removed_files(
        removed_low_quality,
        "qualidade (%)",
        round((quality_threshold * 100), 1),
        DENOISED_OUTPUT_DIR,
        "removidos_baixa_qualidade.txt",
    )

    min_length_minute = round((min_length / 60), 1)
    logging.info(
        f"O arquivo com menor duração é '{min_file}' com {min_length_minute} minutos"
    )

    if min_length_seg:
        min_length = min_length_seg
        min_length_minute = round((min_length / 60), 1)

    logging.info(f"Truncando os sinais para {min_length_minute} minutos")

    truncatedOutputDir = f"{OUTPUT_DIR}/truncated_{min_length_minute}min_{policy}"

    create_truncated_output_dirs(truncatedOutputDir, control_dir, test_dir)

    empty_directory(truncatedOutputDir)

    removed_low_duration = {}
    for file in files[:]:
        denoised_file = get_output_path(
            file, DENOISED_OUTPUT_DIR, control_dir, test_dir, "_denoised.txt"
        )

        rr_intervals = load_rr_intervals(denoised_file)

        rr_truncated, duration_truncated = truncate_rr_intervals(
            rr_intervals, min_length
        )

        if duration_truncated < min_length:
            logging.warning(
                f"Arquivo '{file}' removido: tempo acumulado ({(duration_truncated):.2f} s) abaixo do limite mínimo ({(min_length):.2f} s)"
            )
            removed_low_duration[file] = round((duration_truncated / 60), 1)
            files.remove(file)
        else:

            output_file = get_output_path(
                file,
                truncatedOutputDir,
                control_dir,
                test_dir,
                f"_trunc_{min_length_minute}_min.txt",
            )
            save_rr_intervals(output_file, rr_truncated)

    save_removed_files(
        removed_low_duration,
        "duração (min)",
        round((min_length_seg / 60), 1),
        truncatedOutputDir,
        "removidos_pouca_duracao.txt",
    )
    logging.info(f"Processo de truncamento concluído")


def run_data_processing_and_analysis():
    logging.info("Iniciando o processamento dos dados...")
    process_data(CONTROL_DIR, TEST_DIR, MIN_LENGTH_SEG, POLICY)
    logging.info("Processamento de todos os grupos concluído")

    trunc_control_dir = get_relative_output_path(TRUNCATED_OUTPUT_DIR, CONTROL_DIR)
    trunc_test_dir = get_relative_output_path(TRUNCATED_OUTPUT_DIR, TEST_DIR)

    report_file = os.path.join(OUTPUT_DIR, "relatorio_trunc.txt")
    generate_statistics_report(trunc_control_dir, trunc_test_dir, report_file)


def run_data_analysis(
    output_dir,
    control_dir,
    test_dir,
    report_filename="relatorio.txt",
    quality_threshold=QUALITY_THRESHOLD,
):
    logging.info("Realizando uma análise básica dos dados...")
    # Definir o nome do arquivo de saída para o relatório
    report_file = os.path.join(output_dir, report_filename)

    # # Gera o relatório estatístico
    _, control_file_stats, test_file_stats = generate_statistics_report(
        control_dir, test_dir, report_file, quality_threshold
    )

    if control_file_stats is None and test_file_stats is None:
        logging.warning("Nenhuma relatório de duração foi gerado — arquivos ausentes.")
    else:
        generate_duration_and_quality_file_report(
            control_file_stats,
            test_file_stats,
            report_file.replace(".txt", "_duracao_e_qualidade.txt"),
        )

    logging.info("Análise de dados concluída com sucesso.")


def main():
    setup_logging()

    if ask_user("Deseja realizar uma análise inicial dos dados?") == "s":
        run_data_analysis(OUTPUT_DIR, CONTROL_DIR, TEST_DIR, "relatorio_inicial.txt")
        if ask_user("Deseja continuar com o processamento dos dados?") == "s":
            run_data_processing_and_analysis()
    else:
        run_data_processing_and_analysis()


if __name__ == "__main__":
    main()
