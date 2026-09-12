import os
import numpy as np
import logging
from config import QUALITY_THRESHOLD
from processing import evaluate_signal_quality
from utils import list_rr_files
from file_io import load_rr_intervals


def evaluate_directory_statistics(directory, quality_threshold=QUALITY_THRESHOLD):
    """
    Avalia estatísticas gerais dos arquivos de um diretório.

    Args:
        directory (str): Caminho do diretório a ser avaliado.

    Returns:
        dict: Estatísticas sobre o diretório.
    """
    files_stats = {}  # Alterado para usar o nome do arquivo como chave
    num_files = 0

    # Percorre os arquivos do diretório
    for file in list_rr_files(directory):
        num_files += 1

        # Lê os valores dos intervalos RR
        rr_intervals = load_rr_intervals(file)
        duration = np.sum(rr_intervals)  # Soma total em segundos
        quality = evaluate_signal_quality(rr_intervals)

        # Armazenando as informações dentro de files_stats usando o nome do arquivo como chave
        files_stats[file] = {"duration": duration, "quality": quality}

    if num_files == 0:
        logging.warning(f"Nenhum arquivo encontrado em {directory}")
        return None, None

    # Estatísticas básicas
    durations = [stats["duration"] for stats in files_stats.values()]
    max_duration = np.max(durations)
    min_duration = np.min(durations)
    mean_duration = np.mean(durations)

    qualities = [stats["quality"] for stats in files_stats.values()]
    mean_quality = np.mean(qualities)
    below_threshold = sum(q < quality_threshold for q in qualities)
    above_threshold = len(qualities) - below_threshold

    stats = {
        "Diretório": directory,
        "Número de Arquivos": num_files,
        "Maior Duração (min)": round(max_duration / 60, 2),
        "Arquivo com Maior Duração": os.path.basename(
            max(files_stats, key=lambda x: files_stats[x]["duration"])
        ),
        "Menor Duração (min)": round(min_duration / 60, 2),
        "Arquivo com Menor Duração": os.path.basename(
            min(files_stats, key=lambda x: files_stats[x]["duration"])
        ),
        "Duração Média (min)": round(mean_duration / 60, 2),
        "Limite de Qualidade (%)": round(quality_threshold * 100, 1),
        "Qualidade Média (%)": round(mean_quality * 100, 2),
        "Arquivos Abaixo do Limite": below_threshold,
        "Arquivos Acima do Limite": above_threshold,
    }

    return stats, files_stats


def generate_section_lines(group_name, metrics, report):
    """
    Gera as linhas formatadas para uma seção de um grupo (Control ou Test).

    Args:
        group_name (str): Nome do grupo (Control ou Test).
        metrics (list): Lista das métricas a serem exibidas.
        report (dict): Dicionário com as estatísticas para cada grupo.

    Returns:
        str: Linhas formatadas para a seção.
    """
    lines = [f"{'-'*10} GRUPO: {group_name.upper()} {'-'*10}"]
    lines.extend([f"{metric}: {report[group_name][metric]}" for metric in metrics])
    return "\n".join(lines)


def generate_statistics_report(
    control_dir, test_dir, output_file, quality_threshold=QUALITY_THRESHOLD
):
    """
    Gera um relatório consolidado das estatísticas dos diretórios.

    Args:
        control_dir (str): Caminho para o diretório de controle.
        test_dir (str): Caminho para o diretório de teste.
        output_file (str): Arquivo para salvar o relatório.
    """
    control_stats, control_file_stats = evaluate_directory_statistics(
        control_dir, quality_threshold
    )
    test_stats, test_file_stats = evaluate_directory_statistics(
        test_dir, quality_threshold
    )

    if control_stats is None and test_stats is None:
        logging.warning("Nenhuma estatística foi gerada — arquivos ausentes.")
        return

    report = {"Controle": control_stats, "Teste": test_stats}

    metrics = list(report["Controle"].keys())

    # Gerar a seção do Grupo de Controle
    control_section = generate_section_lines("Controle", metrics, report)

    # Gerar a seção do Grupo de Teste
    test_section = generate_section_lines("Teste", metrics, report)

    # Exibir no console
    title_initial = f"\n{'='*15} INFORMAÇÕES BÁSICAS SOBRE OS GRUPOS {'='*15}"
    logging.info(title_initial)
    logging.info(control_section)
    logging.info(test_section)

    # Salvar em arquivo
    with open(output_file, "w") as f:
        f.write(f"{'='*15} INFORMAÇÕES BÁSICAS SOBRE OS GRUPOS {'='*15}\n\n")
        f.write(control_section + "\n\n")
        f.write(test_section + "\n")

    logging.info(f"Relatório salvo em: {output_file}")

    return report, control_file_stats, test_file_stats


def generate_duration_and_quality_file_report(
    control_file_stats, test_file_stats, output_file
):
    title_duration = (
        f"{'='*10} RELATÓRIO SOBRE A DURAÇÃO E QUALIDADE DOS ARQUIVOS {'='*10}\n\n"
        + "-> Os arquivos estão organizados em ordem crescente de duração.\n\n"
        + "Formato: X.Nome do Arquivo | Duração (min) | Qualidade (%)"
    )

    control_duration_report = generate_group_duration_report(
        control_file_stats, "Controle"
    )

    test_duration_report = generate_group_duration_report(test_file_stats, "Teste")

    logging.info(title_duration)
    logging.info(control_duration_report)
    logging.info(control_duration_report)

    with open(output_file, "w") as f:
        f.write(title_duration)
        f.write(control_duration_report)
        f.write(test_duration_report)


def generate_group_duration_report(group_file_stats, group_name):
    """
    Gera um relatório detalhado com todos os arquivos e seus tempos em minutos,
    separados por grupos (controle e teste), organizados em ordem crescente.

    Args:
        control_dir (str): Caminho para o diretório de controle.
        test_dir (str): Caminho para o diretório de teste.
        output_file (str): Arquivo para salvar o relatório.
    """

    # Gerar linhas formatadas para o relatório
    # Cabeçalho do relatório
    lines = []

    lines.append(f"\n\nGRUPO: {group_name.upper()}")

    total_files = len(group_file_stats)
    # Ordena os arquivos pela duração de forma crescente
    sorted_files = sorted(group_file_stats.items(), key=lambda x: x[1]["duration"])

    for i, (file_path, file_data) in zip(range(total_files, 0, -1), sorted_files):
        file_name = os.path.basename(file_path)
        quality_percent = file_data["quality"] * 100
        duration_minute = file_data["duration"] / 60
        lines.append(
            f"   {i}. {file_name} | {duration_minute:.2f} | {quality_percent:.1f}"
        )

    logging.info("\n".join(lines))
    report = "\n".join(lines)

    return report
