import os
import numpy as np
import locale

from src.extract_features import extract_features

locale.setlocale(locale.LC_COLLATE, "pt_BR.UTF-8")


def process_group(data_dir, base_dir, group_name):
    """
    Process all patients in a group folder and return list of feature dicts
    Each dict contains: {ID, Group, features...}
    """
    features = []
    folder = os.path.join(data_dir, base_dir)
    count = 0

    for filename in sorted(os.listdir(folder), key=locale.strxfrm):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder, filename)
            patient_id = os.path.splitext(filename)[0]

            # Load RRi for this patient
            with open(filepath, "r") as f:
                rri = [float(line.strip()) for line in f if line.strip()]

            metrics = extract_features(rri, patient_id, group_name)
            features.append(metrics)
            count += 1

    return features, count


def save_log_summary(log_path, count_control, count_diabetic):
    """
    Salva um resumo do carregamento dos pacientes em um arquivo de log.
    """
    from datetime import datetime

    with open(log_path, "w") as f:
        f.write("Patient loading summary\n")
        f.write("=======================\n")
        f.write(f"Control group: {count_control} patients\n")
        f.write(f"Diabetic group: {count_diabetic} patients\n")
        f.write(f"Total: {count_control + count_diabetic} patients\n\n")
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
