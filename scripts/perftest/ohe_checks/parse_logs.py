import os
import re
import pandas as pd

# Patterns to search for
patterns = {
    "num_col_groups": re.compile(r"--num col groups:\s+(\d+)"),
    "compressed_size": re.compile(r"--compressed size:\s+(\d+)"),
    "compression_ratio": re.compile(r"--compression ratio:\s+([\d.]+)"),
    "execution_time": re.compile(r"Total execution time:\s+([\d.]+) sec.")
}

# Function to extract metrics from the text files
def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        metrics = {}
        for key, pattern in patterns.items():
            match = pattern.search(content)
            if match:
                metrics[key] = match.group(1)
            else:
                metrics[key] = None
    return metrics

# Directories for baseline and OHE
baseline_dir = "BaselineLogs"
ohe_dir = "OHELogs"

# Data storage
data_combined = []

# Process baseline and corresponding OHE files
for file_name in os.listdir(baseline_dir):
    if file_name.endswith("_encoded_base.txt"):
        experiment_name = file_name[:-4]  # Remove the .txt extension
        file_path_baseline = os.path.join(baseline_dir, file_name)
        metrics_baseline = extract_metrics(file_path_baseline)

        file_name_ohe = file_name.replace('_base.txt', '_ohe.txt').replace('Baseline', 'OHE')
        file_path_ohe = os.path.join(ohe_dir, file_name_ohe)
        if os.path.exists(file_path_ohe):
            metrics_ohe = extract_metrics(file_path_ohe)
        else:
            metrics_ohe = {key: None for key in patterns.keys()}

        combined_metrics = {
            'experiment': experiment_name,
            'baseline_num_col_groups': metrics_baseline.get('num_col_groups'),
            'baseline_compressed_size': metrics_baseline.get('compressed_size'),
            'baseline_compression_ratio': metrics_baseline.get('compression_ratio'),
            'baseline_execution_time': metrics_baseline.get('execution_time'),
            'ohe_num_col_groups': metrics_ohe.get('num_col_groups'),
            'ohe_compressed_size': metrics_ohe.get('compressed_size'),
            'ohe_compression_ratio': metrics_ohe.get('compression_ratio'),
            'ohe_execution_time': metrics_ohe.get('execution_time')
        }

        data_combined.append(combined_metrics)

# Create DataFrame
df_combined = pd.DataFrame(data_combined)

# Write to Excel
output_file = 'combined_metrics.xlsx'
df_combined.to_excel(output_file, index=False)

print(f'Excel file "{output_file}" has been created successfully.')
