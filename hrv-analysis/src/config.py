import os

# Paths
DATA_DIR = "../../data/denoised/4/"
CONTROL_DIR = "control"
DIABETIC_DIR = "diabetes"
RESULTS_DIR = "../results/denoised/4/"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
