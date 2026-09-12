import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from datetime import datetime
import numpy as np

from src.config import DATA_DIR, CONTROL_DIR, DIABETIC_DIR, RESULTS_DIR
from src.io_utils import process_group, save_log_summary


# ---------- Run pipeline ----------
date_str = datetime.now().strftime("%Y_%m_%d")

features_all = []
control_features, count_control = process_group(DATA_DIR, CONTROL_DIR, "control")
diabetic_features, count_diabetic = process_group(DATA_DIR, DIABETIC_DIR, "diabetic")
features_all.extend(control_features)
features_all.extend(diabetic_features)

print(f"Loaded {count_control} patients from group 'control'")
print(f"Loaded {count_diabetic} patients from group 'diabetic'")

# Build dataframe
df_hrv = pd.DataFrame(features_all)

# Save consolidated
output_all = os.path.join(RESULTS_DIR, f"hrv_features_{date_str}.csv")
df_hrv.to_csv(output_all, index=False, encoding="utf-8")
print(f"File successfully saved at: {output_all}")

# Save per group
for group in df_hrv["Group"].unique():
    output_group = os.path.join(RESULTS_DIR, f"hrv_features_{group}_{date_str}.csv")
    df_hrv[df_hrv["Group"] == group].to_csv(output_group, index=False, encoding="utf-8")
    print(f"File successfully saved at: {output_group}")

# Save log
log_path = os.path.join(RESULTS_DIR, f"load_summary_{date_str}.txt")
save_log_summary(log_path, count_control, count_diabetic)
print(f"Summary log saved at: {log_path}")
