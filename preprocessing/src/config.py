import os

DATA_DIR = "../data"
CONTROL_DIR = f"{DATA_DIR}/control"
TEST_DIR = f"{DATA_DIR}/diabetes"

MIN_LENGTH_SEG = 180  # Duracao desejada para os arquivos truncados em segundos
POLICY = "early"  # "early", "late", "best"

CLIP_START_LENGTH = 10  # Amount of entries (RR) to clip from the start of the data

QUALITY_THRESHOLD = 0.0  # Threshold for quality of the data

# THRESHOLDS IN MILLISECONDS
LOW_RRI = 300
HIGH_RRI = 2000

# Diretório base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretórios de saída
OUTPUT_DIR = os.path.join(BASE_DIR, f"{DATA_DIR}/output")
DENOISED_OUTPUT_DIR = os.path.join(BASE_DIR, f"{DATA_DIR}/output/denoised")
TRUNCATED_OUTPUT_DIR = os.path.join(
    BASE_DIR, f"{DATA_DIR}/output/truncated_{round(MIN_LENGTH_SEG/60, 1)}min_{POLICY}"
)

# Parâmetros para processamento
OUTLIER_THRESHOLD = 3
MEDIAN_FILTER_KERNEL_SIZE = 5

# Configurações de logging
LOG_FILE = os.path.join(BASE_DIR, f"{DATA_DIR}/logs/rr_processing.log")
LOG_LEVEL = "ERROR"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

control_basename = os.path.basename(CONTROL_DIR)
test_basename = os.path.basename(TEST_DIR)

# Criação de diretórios se não existirem
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(DENOISED_OUTPUT_DIR, control_basename), exist_ok=True)
os.makedirs(os.path.join(DENOISED_OUTPUT_DIR, test_basename), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
