import os
import logging


def list_rr_files(directory):
    """Lista todos os arquivos RR em um diretório."""
    files = sorted(
        [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".txt")
        ]
    )
    logging.info(f"{len(files)} arquivo(s) encontrado(s) em {directory}")
    return files


def get_output_path(file, output_dir, control_dir, test_dir, suffix):
    """
    Retorna o caminho de saída para um arquivo processado.
    """
    dir = control_dir if control_dir in file else test_dir
    output_dir = os.path.join(output_dir, os.path.basename(dir))
    output_file = os.path.join(
        output_dir, os.path.basename(file).replace(".txt", suffix)
    )
    return output_file

def get_relative_output_path(output_dir, group_dir):
    path = os.path.join(
                os.path.relpath(output_dir), os.path.basename(group_dir)
            )
    return path

def ask_user(question=None):
    while True:
        response = (
            input("Deseja continuar?" if question is None else question + "(s/n): ")
            .strip()
            .lower()
        )
        if response in ["s", "n"]:
            return response
        print("Entrada inválida! Digite 's' para continuar ou 'n' para sair.")
