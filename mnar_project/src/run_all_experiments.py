"""
Skript for running the whole experiment pipeline. 

It works as follows
- load dataset
- check if the dataset matches a supported generator
- prepare a numerical version if necessary
- generate missing values at 10% and 5%
- run baselines
- run Python prototype
- run DML prototype
- consolidate results into comparison tables
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


mnar_project = Path(__file__).resolve().parents[1]
data_dir = mnar_project / "data" / "processed"
results_dir = mnar_project / "results"
dml_dir = mnar_project / "dml"
systemds_root = mnar_project.parents[0] / "systemds"

gen_diabetes_script = mnar_project / "src" / "gen_hybrid_miss.py"
gen_aq_script = mnar_project / "src" / "gen_hybrid_miss_AQ.py"
baselines_script = mnar_project / "src" / "baselines_general.py"
python_prototype_script = mnar_project / "src" / "rebalanced_mice_general.py"
dml_script = dml_dir / "rebalanced_mice_complete.dml"

random_state = 100
default_java_home = "/opt/homebrew/Cellar/openjdk@17/17.0.19/libexec/openjdk.jdk/Contents/Home"
full_baseline_row_limit = 50000


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type = str, required = True)
    parser.add_argument("--dataset-type", type = str, default = "auto", choices = ["auto", "diabetes", "AQ"])
    parser.add_argument("--generator-script", type = str, default = "")
    parser.add_argument("--max-iterations", type = int, default = 5)
    parser.add_argument("--baseline-mode", type = str, default = "auto", choices = ["auto", "full", "key"])
    parser.add_argument("--summary-name", type = str, default = "")

    return parser.parse_args()


def sanitize_name(name):

    cleaned = re.sub(r"[^A-Za-z0-9_\\-]+", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    return cleaned


def detect_dataset_type(df):

    diabetes_targets = {"BMI", "Income", "MentHlth"}
    aq_targets = {"NOx(GT)", "NO2(GT)", "CO(GT)"}

    columns = set(df.columns)

    if diabetes_targets.issubset(columns):
        return "diabetes"

    if aq_targets.issubset(columns):
        return "AQ"

    raise ValueError(
        "This data set is not currently supported."
    )


def get_target_columns(dataset_type):

    if dataset_type == "diabetes":
        return ["BMI", "Income", "MentHlth"]

    if dataset_type == "AQ":
        return ["CO(GT)", "NOx(GT)", "NO2(GT)"]

    raise ValueError(f"Unbekannter dataset_type: {dataset_type}")


def get_generator_script(dataset_type):

    if dataset_type == "diabetes":
        return gen_diabetes_script

    if dataset_type == "AQ":
        return gen_aq_script

    raise ValueError(f"unknown dataset_type: {dataset_type}")


def resolve_generator_script(args, df):

    if args.generator_script:
        generator_script = Path(args.generator_script).resolve()

        if not generator_script.exists():
            raise FileNotFoundError(f"Generator script not found: {generator_script}")

        if args.dataset_type == "auto":
            dataset_type = "custom"
        else:
            dataset_type = args.dataset_type

        return dataset_type, generator_script

    if args.dataset_type == "auto":
        dataset_type = detect_dataset_type(df)
    else:
        dataset_type = args.dataset_type

    generator_script = get_generator_script(dataset_type)

    return dataset_type, generator_script


def run_command(cmd, env = None):

    print("\nRunning:")
    print(" ".join(str(part) for part in cmd))

    subprocess.run(
        [str(part) for part in cmd],
        check = True,
        env = env
    )


def prepare_numeric_copy(input_path, run_dir):

    df = pd.read_csv(input_path)
    df_numeric = df.select_dtypes(include = ["number"]).copy()

    numeric_path = run_dir / f"{input_path.stem}_numeric_only.csv"
    df_numeric.to_csv(numeric_path, index = False)

    return numeric_path, df.shape[1], df_numeric.shape[1]


def save_for_systemds_generic(input_path, output_path):

    df = pd.read_csv(input_path)
    df_numeric = df.select_dtypes(include = ["number"]).copy()
    df_numeric.to_csv(output_path, index = False, na_rep = "NaN")


def eval_imp_complete(df_imputed, mask, column, method_name, missing_rate_label):

    results = []

    mask_column = mask[mask["column"] == column].copy()
    group_filter = mask_column["is_missing"] == 1
    idx = mask_column[group_filter]["row_id"]

    actual_val = mask_column.loc[group_filter, "ground_truth"]
    pred_val = df_imputed.loc[idx, column]

    errors = pred_val.reset_index(drop = True) - actual_val.reset_index(drop = True)
    errors_abs = errors.abs()

    mae = errors_abs.mean()
    rmse = (errors.pow(2).mean()) ** 0.5
    bias = errors.mean()

    results.append({
        "method": method_name,
        "column": column,
        "group": "all",
        "missing_rate": missing_rate_label,
        "n": len(idx),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "actual_mean": actual_val.mean(),
        "predicted_mean": pred_val.mean(),
    })

    return results


def run_key_baselines(file_path, mask_path, results_path, missing_rate_label):

    print("\nRunning key baselines.")

    df = pd.read_csv(file_path)
    mask = pd.read_csv(mask_path)
    df_numeric = df.select_dtypes(include = [np.number]).copy()

    columns_with_missing = [
        column for column in df_numeric.columns
        if df_numeric[column].isna().sum() > 0
    ]

    all_results = []

    for column_missing in columns_with_missing:
        observed = df_numeric.loc[df_numeric[column_missing].notna(), column_missing]

        substitutes = {
            "mean": observed.mean(),
            "median": observed.median(),
            "mode": observed.mode().iloc[0],
        }

        for method_name, value in substitutes.items():
            df_imputed = df_numeric.copy()
            df_imputed[column_missing] = df_imputed[column_missing].fillna(value)

            all_results.extend(
                eval_imp_complete(
                    df_imputed = df_imputed,
                    mask = mask,
                    column = column_missing,
                    method_name = method_name,
                    missing_rate_label = missing_rate_label
                )
            )

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_numeric)
    df_scaled = pd.DataFrame(
        scaled_values,
        columns = df_numeric.columns,
        index = df_numeric.index
    )

    imputer = IterativeImputer(
        max_iter = 5,
        random_state = random_state,
        sample_posterior = False
    )

    imputed_scaled_values = imputer.fit_transform(df_scaled)
    imputed_values = scaler.inverse_transform(imputed_scaled_values)

    df_imputed = pd.DataFrame(
        imputed_values,
        columns = df_numeric.columns,
        index = df_numeric.index
    )

    for column_missing in columns_with_missing:
        all_results.extend(
            eval_imp_complete(
                df_imputed = df_imputed,
                mask = mask,
                column = column_missing,
                method_name = "mice5",
                missing_rate_label = missing_rate_label
            )
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_path, index = False)

    return results_df


def run_baselines(file_path, mask_path, results_path, baseline_mode, missing_rate_label):

    df = pd.read_csv(file_path)

    effective_mode = baseline_mode

    if baseline_mode == "auto":
        if len(df) > full_baseline_row_limit:
            effective_mode = "key"
        else:
            effective_mode = "full"

    if effective_mode == "full":
        run_command([
            sys.executable,
            baselines_script,
            "--file-path", file_path,
            "--mask-path", mask_path,
            "--results-path", results_path,
        ])

        results_df = pd.read_csv(results_path)
        results_df["missing_rate"] = missing_rate_label
        results_df.to_csv(results_path, index = False)
        return results_df, effective_mode

    results_df = run_key_baselines(
        file_path = file_path,
        mask_path = mask_path,
        results_path = results_path,
        missing_rate_label = missing_rate_label
    )

    return results_df, effective_mode


def run_python_prototype(file_path, mask_path, results_path, tuning_path, max_iterations, method_name):

    run_command([
        sys.executable,
        python_prototype_script,
        "--file-path", file_path,
        "--mask-path", mask_path,
        "--results-path", results_path,
        "--tuning-path", tuning_path,
        "--method-name", method_name,
        "--max-iterations", str(max_iterations),
    ])

    return pd.read_csv(results_path), pd.read_csv(tuning_path)


def run_dml_prototype(file_path, mask_path, output_path, tuning_path, eval_path, max_iterations, method_name, missing_rate_label):

    systemds_input_path = output_path.parent / f"{output_path.stem}_systemds_input.csv"
    save_for_systemds_generic(file_path, systemds_input_path)

    env = os.environ.copy()

    if not env.get("JAVA_HOME"):
        env["JAVA_HOME"] = default_java_home

    env["PATH"] = f"{env['JAVA_HOME']}/bin:{env.get('PATH', '')}"
    env["SYSTEMDS_ROOT"] = str(systemds_root)
    systemds_jar_file = systemds_root / "target" / "systemds-3.4.0-SNAPSHOT.jar"

    if systemds_jar_file.exists():
        env["SYSTEMDS_JAR_FILE"] = str(systemds_jar_file)

    run_command([
        systemds_root / "bin" / "systemds",
        "-f", dml_script,
        "-args",
        systemds_input_path,
        output_path,
        tuning_path,
        str(max_iterations),
    ], env = env)

    df_imputed = pd.read_csv(output_path, header = None)
    df_original = pd.read_csv(file_path)
    df_numeric = df_original.select_dtypes(include = ["number"]).copy()
    df_imputed.columns = df_numeric.columns

    mask = pd.read_csv(mask_path)
    columns_with_missing = [
        column for column in df_numeric.columns
        if df_numeric[column].isna().sum() > 0
    ]

    all_results = []

    for column_missing in columns_with_missing:
        all_results.extend(
            eval_imp_complete(
                df_imputed = df_imputed,
                mask = mask,
                column = column_missing,
                method_name = method_name,
                missing_rate_label = missing_rate_label
            )
        )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(eval_path, index = False)

    tuning_df = pd.read_csv(tuning_path, header = None)

    return results_df, tuning_df


def build_summary_table(baselines_df, python_df, dml_df, summary_path, missing_rate_label):

    method_order = ["mean", "median", "mode", "mice5", "rebalanced_mice_general", "rebalanced_mice_dml"]

    baseline_keep = baselines_df[baselines_df["method"].isin(["mean", "median", "mode", "mice5"])].copy()
    python_keep = python_df.copy()
    dml_keep = dml_df.copy()

    summary_df = pd.concat([baseline_keep, python_keep, dml_keep], ignore_index = True)
    summary_df["missing_rate"] = missing_rate_label
    summary_df["method"] = pd.Categorical(summary_df["method"], categories = method_order, ordered = True)
    summary_df = summary_df.sort_values(["column", "method"]).reset_index(drop = True)
    summary_df.to_csv(summary_path, index = False)

    return summary_df


def print_summary_table(summary_df, title):

    print(f"\n{title}")
    print(summary_df[[
        "method",
        "column",
        "missing_rate",
        "mae",
        "rmse",
        "bias",
        "actual_mean",
        "predicted_mean",
    ]].to_string(index = False))


def main():
    args = parse_args()

    input_path = Path(args.input_path).resolve()
    df = pd.read_csv(input_path)

    dataset_type, generator_script = resolve_generator_script(args, df)

    dataset_name = sanitize_name(input_path.stem)
    summary_name = args.summary_name or dataset_name

    run_dir = results_dir / f"all_experiments_{summary_name}"
    run_dir.mkdir(parents = True, exist_ok = True)

    numeric_copy_path, n_columns_all, n_columns_numeric = prepare_numeric_copy(input_path, run_dir)

    print("input shape", df.shape)
    print("dataset type", dataset_type)
    print("all columns", n_columns_all)
    print("numeric columns", n_columns_numeric)

    if n_columns_numeric != n_columns_all:
        print("For the general procedures, non-numeric columns were also prepared as numeric copies:")
        print(numeric_copy_path)

    if dataset_type in ["diabetes", "AQ"]:
        target_columns = get_target_columns(dataset_type)
    else:
        target_columns = ["wird vom Generator bestimmt"]

    all_summary_tables = []

    for missing_rate in [0.10, 0.05]:
        rate_label = f"{int(missing_rate * 100)}pct"
        rate_suffix = f"{int(missing_rate * 100)}"

        corrupted_name = f"{dataset_name}_mnar_{rate_suffix}.csv"
        mask_name = f"mask_{dataset_name}_mnar_{rate_suffix}.csv"

        corrupted_path = data_dir / corrupted_name
        mask_path = data_dir / mask_name

        print(f"\nMissingness {missing_rate:.0%}")

        run_command([
            sys.executable,
            generator_script,
            "--input-path", input_path,
            "--missing-rate", str(missing_rate),
            "--output-name", corrupted_name,
            "--mask-name", mask_name,
        ])

        baselines_results_path = run_dir / f"baselines_{dataset_name}_{rate_label}.csv"
        python_results_path = run_dir / f"python_rebalanced_{dataset_name}_{rate_label}.csv"
        python_tuning_path = run_dir / f"python_rebalanced_tuning_{dataset_name}_{rate_label}.csv"
        dml_output_path = run_dir / f"dml_output_{dataset_name}_{rate_label}.csv"
        dml_tuning_path = run_dir / f"dml_tuning_{dataset_name}_{rate_label}.csv"
        dml_eval_path = run_dir / f"dml_eval_{dataset_name}_{rate_label}.csv"
        summary_path = run_dir / f"summary_{dataset_name}_{rate_label}.csv"

        baselines_df, effective_baseline_mode = run_baselines(
            file_path = corrupted_path,
            mask_path = mask_path,
            results_path = baselines_results_path,
            baseline_mode = args.baseline_mode,
            missing_rate_label = rate_label
        )

        print("Baseline mode:", effective_baseline_mode)

        python_df, _ = run_python_prototype(
            file_path = corrupted_path,
            mask_path = mask_path,
            results_path = python_results_path,
            tuning_path = python_tuning_path,
            max_iterations = args.max_iterations,
            method_name = "rebalanced_mice_general"
        )
        python_df["missing_rate"] = rate_label

        dml_df, _ = run_dml_prototype(
            file_path = corrupted_path,
            mask_path = mask_path,
            output_path = dml_output_path,
            tuning_path = dml_tuning_path,
            eval_path = dml_eval_path,
            max_iterations = args.max_iterations,
            method_name = "rebalanced_mice_dml",
            missing_rate_label = rate_label
        )

        summary_df = build_summary_table(
            baselines_df = baselines_df,
            python_df = python_df,
            dml_df = dml_df,
            summary_path = summary_path,
            missing_rate_label = rate_label
        )

        summary_df["dataset"] = dataset_name
        summary_df["dataset_type"] = dataset_type
        summary_df["target_columns"] = ",".join(target_columns)

        all_summary_tables.append(summary_df)

        print_summary_table(
            summary_df = summary_df,
            title = f"Summary {dataset_name} {rate_label}"
        )

    combined_summary = pd.concat(all_summary_tables, ignore_index = True)
    combined_summary_path = run_dir / f"summary_{dataset_name}_combined.csv"
    combined_summary.to_csv(combined_summary_path, index = False)

    print("\nfiles:")
    print("output ordner:", run_dir)
    print("combined summary:", combined_summary_path)


if __name__ == "__main__":
    main()
