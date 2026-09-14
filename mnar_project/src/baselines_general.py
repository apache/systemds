"""
Allgemeinere Baselines und einfache Imputationsverfahren für numerische CSV-Datensätze.

Getestet werden:
- mean
- median
- mode
- knn
- mice

Voraussetzung:
- csv mit numerischen Spalten
- mask-csv mit row_id, column, is_missing und ground_truth
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


mnar_project = Path(__file__).resolve().parents[1]

file_path = mnar_project / "data" / "processed" / "AQ_mnar.csv"
mask_path = mnar_project / "data" / "processed" / "mask_AQ_mnar.csv"
results = mnar_project / "results"
results.mkdir(parents=True, exist_ok=True)


neighbors = [3, 5, 10, 20]
iterations = [1, 2, 3, 4, 5, 10]
random_state = 100


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type = str, default = str(file_path))
    parser.add_argument("--mask-path", type = str, default = str(mask_path))
    parser.add_argument("--results-path", type = str, default = "")

    return parser.parse_args()


def get_columns_with_missing(df):

    columns_with_missing = []

    for column in df.columns:
        if df[column].isna().sum() > 0:
            columns_with_missing.append(column)

    return columns_with_missing


def eval_imp_complete(df_imputed, mask, column, method_name):

    results = []

    mask_column = mask[mask["column"] == column].copy()

    groups = [
        ("all", mask_column["is_missing"] == 1),
    ]

    for group_name, group_filter in groups:
        idx = mask_column[group_filter]["row_id"]

        actual_val = mask_column.loc[group_filter, "ground_truth"]
        pred_val = df_imputed.loc[idx, column]

        errors = pred_val.reset_index(drop=True) - actual_val.reset_index(drop=True)
        errors_abs = errors.abs()

        mae = errors_abs.mean()
        rmse = (errors.pow(2).mean()) ** 0.5
        bias = errors.mean()

        results.append({
            "method": method_name,
            "column": column,
            "group": group_name,
            "n": len(idx),
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "actual_mean": actual_val.mean(),
            "predicted_mean": pred_val.mean(),
        })

    return results


def main():
    args = parse_args()

    current_file_path = Path(args.file_path)
    current_mask_path = Path(args.mask_path)

    df = pd.read_csv(current_file_path)
    mask = pd.read_csv(current_mask_path)

    df_numeric = df.select_dtypes(include = [np.number]).copy()

    if df_numeric.shape[1] != df.shape[1]:
        print("\nNur numerische Spalten werden verwendet.")

    columns_with_missing = get_columns_with_missing(df_numeric)

    print("Input shape:", df_numeric.shape)
    print("Missing columns:", columns_with_missing)

    all_results = []

    print("\n Ergebnisse mean, median und mode:")

    for column_missing in columns_with_missing:

        # Nur echte & beobachtete Werte verwenden, denn die
        # künstlich entfernten Werte wollen wir nicht in die Schätzung mit eingehen lassen
        observed = df_numeric.loc[df_numeric[column_missing].notna(), column_missing]

        substitutes = {
            "mean": observed.mean(),
            "median": observed.median(),
            "mode": observed.mode().iloc[0],
        }

        print(f"\n{column_missing}:")

        for name, wert in substitutes.items():
            print(f"{name}: {wert}")

        for methode, wert in substitutes.items():

            daten_imp = df_numeric.copy()
            daten_imp[column_missing] = daten_imp[column_missing].fillna(wert)

            all_results.extend(
                eval_imp_complete(daten_imp, mask, column_missing, methode)
            )

    for k in neighbors:

        df_knn_input = df_numeric.copy()
        scaler = StandardScaler()

        scaled_values = scaler.fit_transform(df_knn_input)
        df_scaled = pd.DataFrame(
            scaled_values,
            columns = df_knn_input.columns,
            index = df_knn_input.index
        )

        imputer = KNNImputer(n_neighbors = k)

        imputed_scaled_values = imputer.fit_transform(df_scaled)
        imputed_values = scaler.inverse_transform(imputed_scaled_values)

        df_imputed = pd.DataFrame(
            imputed_values,
            columns = df_knn_input.columns,
            index = df_knn_input.index
        )

        method_name = f"knn_k{k}"

        for column_missing in columns_with_missing:
            all_results.extend(
                eval_imp_complete(df_imputed, mask, column_missing, method_name)
            )

    for max_iter in iterations:

        df_mice_input = df_numeric.copy()

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_mice_input)
        df_scaled = pd.DataFrame(
            scaled_values,
            columns = df_mice_input.columns,
            index = df_mice_input.index
        )

        imputer = IterativeImputer(
            max_iter = max_iter,
            random_state  = random_state,
            sample_posterior = False
        )

        imputed_scaled_values = imputer.fit_transform(df_scaled)
        imputed_values = scaler.inverse_transform(imputed_scaled_values)

        df_imputed = pd.DataFrame(
            imputed_values,
            columns = df_mice_input.columns,
            index = df_mice_input.index
        )

        method_name = f"mice{max_iter}"

        for column_missing in columns_with_missing:
            all_results.extend(
                eval_imp_complete(df_imputed, mask, column_missing, method_name)
            )

    results_df = pd.DataFrame(all_results)

    print("\nVergleich der Baselines")
    print(results_df)

    if args.results_path:
        results_path = Path(args.results_path)
        results_df.to_csv(results_path, index = False)


main()
