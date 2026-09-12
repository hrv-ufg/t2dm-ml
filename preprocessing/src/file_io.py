import numpy as np
import logging
import os
from config import CONTROL_DIR, TEST_DIR


def load_rr_intervals(file_path):
    """Carrega os intervalos RR de um arquivo."""
    try:
        logging.debug(f"Carregando arquivo: {file_path}")
        data = np.loadtxt(file_path, dtype=float, delimiter=" ", encoding="ISO-8859-1")
        logging.debug(f"Arquivo carregado com sucesso: {file_path}")

        # Se os dados tiverem apenas uma coluna, retorna diretamente
        if data.ndim == 1:  # Caso em que há apenas uma coluna
            logging.debug(f"Arquivo possui apenas uma coluna.")
            return data / 1000 if max(data) > 100 else data

        else:
            # Se houver duas colunas, retorna apenas a segunda (intervalos RR)
            logging.debug(f"Arquivo possui duas colunas.")
            return data[:, 1]  # Retorna a segunda coluna (intervalos RR)

    except Exception as e:
        logging.error(f"Erro ao carregar arquivo {file_path}: {e}")
        return None


def save_rr_intervals(file_path, data):
    """Salva os intervalos RR processados em um arquivo."""
    try:
        logging.debug(f"Salvando arquivo: {file_path}")
        np.savetxt(file_path, data, fmt="%.3f")
        logging.debug(f"Arquivo salvo com sucesso: {file_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar arquivo {file_path}: {e}")


def save_removed_files(removed_files, param, threshold, output_dir, file_name):
    """Salva os arquivos removidos em um arquivo."""
    title = f"{'='*10} LISTA DE ARQUIVOS REMOVIDOS COM {param.upper()} INFERIOR A {threshold:.1f} {'='*10}\n\n"

    output_file = os.path.join(output_dir, file_name)
    logging.debug(f"Salvando arquivo de arquivos removidos: {output_file}")

    count_control = sum(CONTROL_DIR in file for file in removed_files.keys())
    count_test = sum(TEST_DIR in file for file in removed_files.keys())

    first_subtitle = f"Qtd. de arquivos removidos: {CONTROL_DIR}: {count_control} e {TEST_DIR}: {count_test}\n\n"
    second_subtitle = f"Formato: X.Nome do Arquivo | {param.capitalize()}\n\n"

    with open(output_file, "w") as f:
        f.write(title)
        f.write(first_subtitle)
        f.write(second_subtitle)
        for file, quality in removed_files.items():
            f.write(file)
            f.write(f" | {quality:.2f}\n")

    logging.debug(f"Arquivo de arquivos removidos salvo com sucesso em: {output_file}")
