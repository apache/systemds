"""
A more general version of 'general' rebalanced MICE for numeric CSV datasets.

The idea remains the same:
- MICE-style loop over multiple missing columns
- target weights for rare/skewed target ranges
- similarity weights for rows that resemble missing rows
- a small validation setup for selecting the rebalancing configuration

This version should run on other numerical datasets with minimal adjustment.
Prerequisites:
- CSV with numerical columns
- mask-CSV with row_id, column, is_missing, and ground_truth
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score


mnar_project = Path(__file__).resolve().parents[1]

file_path = mnar_project / "data" / "processed" / "AQ_mnar.csv"
mask_path = mnar_project / "data" / "processed" / "mask_AQ_mnar.csv"
results = mnar_project / "results"
results.mkdir(parents=True, exist_ok=True)


max_iterations = 5
random_state = 100
validation_share = 0.2


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type = str, default = str(file_path))
    parser.add_argument("--mask-path", type = str, default = str(mask_path))
    parser.add_argument("--results-path", type = str, default = "")
    parser.add_argument("--tuning-path", type = str, default = "")
    parser.add_argument("--method-name", type = str, default = "rebalanced_mice_general")
    parser.add_argument("--max-iterations", type = int, default = max_iterations)

    return parser.parse_args()


def stabilizing(weights):

    weights = weights.astype(float)

    if weights.mean() == 0:
        weights = weights + 1.0

        return weights

    weights = weights / weights.mean()

    return weights


def get_columns_with_missing(df):

    columns_with_missing = []

    for column in df.columns:
        if df[column].isna().sum() > 0:
            columns_with_missing.append(column)

    return columns_with_missing


def initial_fill_complete(df, columns_with_missing):

    df_imputed = df.copy()

    for column_missing in columns_with_missing:
        observed = df_imputed.loc[df_imputed[column_missing].notna(), column_missing]
        observed_mean = observed.mean()
        df_imputed[column_missing] = df_imputed[column_missing].fillna(observed_mean)

    return df_imputed


def scew_row_weights(y_train, n_bins = 6, smoothing = 10.0, max_weight = 5.0, scew_threshold = 0.25):

    y_train = y_train.copy()

    scew = y_train.skew()

    unique_vals = y_train.nunique()
    n_bins_effective = min(n_bins, unique_vals)

    if n_bins_effective <= 1:
        zielvariable_weights = pd.Series(1.0, index = y_train.index)
        return zielvariable_weights, scew

    bin_id = pd.cut(
        y_train,
        bins = n_bins_effective,
        labels = False,
        include_lowest = True,
        duplicates = "drop"
    )

    bin_counts = bin_id.value_counts().sort_index()
    mean_bin_count = bin_counts.mean()

    weight_by_size = bin_id.map(
        lambda current_bin: mean_bin_count / (bin_counts.loc[current_bin] + smoothing)
    )

    weight_by_size = weight_by_size.astype(float)

    n_bin = int(bin_id.max())
    normal_people_n_bin = int(bin_id.max()) + 1

    if normal_people_n_bin <= 1:
        zielvariable_weights = pd.Series(1.0, index = y_train.index)
        return zielvariable_weights, scew

    if scew > scew_threshold:

        tail_direction = bin_id.map(
            lambda current_bin: 1.0 + (current_bin / max(n_bin, 1)) * (max_weight - 1.0)
        )

    elif scew < -scew_threshold:

        denominator = max(n_bin - 1, 1)
        tail_direction = bin_id.map(
            lambda current_bin: 1.0 + ((n_bin - current_bin) / denominator) * (max_weight - 1.0)
        )

    else:

        tail_direction = pd.Series(1.0, index = y_train.index)

    zielvariable_weights = weight_by_size * tail_direction
    zielvariable_weights = zielvariable_weights.clip(upper = max_weight)
    zielvariable_weights = stabilizing(zielvariable_weights)

    return zielvariable_weights, scew


def compute_similarity_weights(df_input, missing_indicator, observed_idx, feature_cols, max_weight = 3.0):

    x = df_input[feature_cols].copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clf = LogisticRegression(
        max_iter = 1000,
        class_weight = "balanced",
        random_state = random_state
    )

    clf.fit(x_scaled, missing_indicator)

    missing_probs = clf.predict_proba(x_scaled)[:, 1]

    try:
        auc = roc_auc_score(missing_indicator, missing_probs)
    except ValueError:
        auc = np.nan

    propensity = pd.Series(
        missing_probs,
        index = df_input.index
    )

    observed_propensity = propensity.loc[observed_idx]

    similarity_weights = stabilizing(observed_propensity)
    similarity_weights = similarity_weights.clip(upper = max_weight)
    similarity_weights = stabilizing(similarity_weights)

    return similarity_weights, auc


def fit_one_column(df_input, missing_indicator, column, config):

    missing_idx = missing_indicator[missing_indicator == 1].index
    observed_idx = missing_indicator[missing_indicator == 0].index

    feature_cols = [col for col in df_input.columns if col != column]

    x_train = df_input.loc[observed_idx, feature_cols]
    y_train = df_input.loc[observed_idx, column]
    x_missing = df_input.loc[missing_idx, feature_cols]

    target_weights, scew_value = scew_row_weights(
        y_train = y_train,
        n_bins = config["n_bins"],
        smoothing = config["smoothing"],
        max_weight = config["max_weight_target"],
        scew_threshold = config["scew_threshold"]
    )

    similarity_weight_values, auc = compute_similarity_weights(
        df_input = df_input,
        missing_indicator = missing_indicator,
        observed_idx = observed_idx,
        feature_cols = feature_cols,
        max_weight = config["max_weight_similarity"]
    )

    final_weights = target_weights * similarity_weight_values
    final_weights = final_weights.clip(upper = config["final_weight_cap"])
    final_weights = stabilizing(final_weights)

    scaler = StandardScaler()
    x_train_trans = scaler.fit_transform(x_train)
    x_missing_trans = scaler.transform(x_missing)

    model = Ridge(
        alpha = 1.0,
        random_state = random_state
    )

    model.fit(
        x_train_trans,
        y_train,
        sample_weight = final_weights
    )

    predictions = pd.Series(
        model.predict(x_missing_trans),
        index = missing_idx
    )

    meta = {
        "auc": auc,
        "scew": scew_value,
        "n_observed": len(observed_idx),
        "n_missing": len(missing_idx),
    }

    return predictions, meta


def draw_validation_idx(y_observed, validation_share = 0.2, n_bins = 6):

    unique_vals = y_observed.nunique()
    n_bins_effective = min(n_bins, unique_vals)

    if n_bins_effective <= 1:
        n_val = max(1, int(len(y_observed) * validation_share))
        return y_observed.sample(n = n_val, random_state = random_state).index

    bin_id = pd.cut(
        y_observed,
        bins = n_bins_effective,
        labels = False,
        include_lowest = True,
        duplicates = "drop"
    )

    validation_idx = []

    for _, idx_in_bin in bin_id.groupby(bin_id):
        idx_values = idx_in_bin.index
        n_val = max(1, int(len(idx_values) * validation_share))
        sampled = pd.Series(idx_values).sample(
            n = min(n_val, len(idx_values)),
            random_state = random_state
        )
        validation_idx.extend(sampled.tolist())

    return pd.Index(validation_idx)


def build_candidate_configs():

    candidate_configs = [
        {
            "name": "weighted_only",
            "n_bins": 6,
            "smoothing": 10.0,
            "max_weight_target": 4.0,
            "scew_threshold": 0.25,
            "max_weight_similarity": 2.0,
            "final_weight_cap": 6.0,
            "bias_weight": 0.5,
        },
        {
            "name": "weighted_stronger",
            "n_bins": 6,
            "smoothing": 8.0,
            "max_weight_target": 5.0,
            "scew_threshold": 0.25,
            "max_weight_similarity": 3.0,
            "final_weight_cap": 8.0,
            "bias_weight": 0.5,
        },
        {
            "name": "weighted_smoother",
            "n_bins": 8,
            "smoothing": 20.0,
            "max_weight_target": 4.0,
            "scew_threshold": 0.25,
            "max_weight_similarity": 2.0,
            "final_weight_cap": 6.0,
            "bias_weight": 0.5,
        },
    ]

    return candidate_configs


candidate_configs = build_candidate_configs()


def tune_rebalancing_for_column(df_start, column, candidate_configs):

    observed_idx = df_start[df_start[column].notna()].index
    y_observed = df_start.loc[observed_idx, column]

    validation_idx = draw_validation_idx(
        y_observed = y_observed,
        validation_share = validation_share,
        n_bins = 6
    )

    missing_indicator_validation = pd.Series(0, index = df_start.index)
    missing_indicator_validation.loc[validation_idx] = 1

    df_validation = df_start.copy()
    df_validation.loc[validation_idx, column] = pd.NA

    best_score = np.inf
    best_config = None
    best_meta = None

    for config in candidate_configs:

        predictions, meta = fit_one_column(
            df_input = df_validation,
            missing_indicator = missing_indicator_validation,
            column = column,
            config = config
        )

        actual_val = df_start.loc[validation_idx, column]
        pred_val = predictions.loc[validation_idx]

        errors = pred_val.reset_index(drop=True) - actual_val.reset_index(drop=True)
        mae = errors.abs().mean()
        bias = errors.mean()
        score = mae + config["bias_weight"] * abs(bias)

        if score < best_score:
            best_score = score
            best_config = config.copy()
            best_meta = {
                "validation_mae": mae,
                "validation_bias": bias,
                "validation_score": score,
                "validation_n": len(validation_idx),
                "auc": meta["auc"],
                "scew": meta["scew"],
            }

    return best_config, best_meta


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


def rebalanced_mice_general(df, mask, columns_with_missing, max_iter):

    df_start = initial_fill_complete(df, columns_with_missing)
    df_imputed = df_start.copy()

    best_configs = {}
    tuning_rows = []

    print("\nValidation für die Rebalancing-Konfigurationen:")

    for column_missing in columns_with_missing:
        best_config, best_meta = tune_rebalancing_for_column(
            df_start = df_start,
            column = column_missing,
            candidate_configs = candidate_configs
        )

        best_configs[column_missing] = best_config

        tuning_row = {
            "column": column_missing,
            "config_name": best_config["name"],
            "n_bins": best_config["n_bins"],
            "smoothing": best_config["smoothing"],
            "max_weight_target": best_config["max_weight_target"],
            "max_weight_similarity": best_config["max_weight_similarity"],
            "final_weight_cap": best_config["final_weight_cap"],
            "bias_weight": best_config["bias_weight"],
            **best_meta
        }

        tuning_rows.append(tuning_row)
        print(f"{column_missing}: {best_config['name']} | score = {best_meta['validation_score']:.4f}")

    print("\nMICE-Loop:")

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}")

        for column_missing in columns_with_missing:
            missing_indicator_real = df[column_missing].isna().astype(int)

            predictions, meta = fit_one_column(
                df_input = df_imputed,
                missing_indicator = missing_indicator_real,
                column = column_missing,
                config = best_configs[column_missing]
            )

            missing_idx = missing_indicator_real[missing_indicator_real == 1].index
            df_imputed.loc[missing_idx, column_missing] = predictions

            print(
                f"{column_missing}: auc = {meta['auc']:.4f} | "
                f"scew = {meta['scew']:.4f}"
            )

    tuning_df = pd.DataFrame(tuning_rows)

    return df_imputed, tuning_df


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

    df_rebalanced, tuning_df = rebalanced_mice_general(
        df = df_numeric,
        mask = mask,
        columns_with_missing = columns_with_missing,
        max_iter = args.max_iterations
    )

    all_results = []

    for column_missing in columns_with_missing:
        all_results.extend(
            eval_imp_complete(
                df_imputed = df_rebalanced,
                mask = mask,
                column = column_missing,
                method_name = args.method_name
            )
        )

    results_df = pd.DataFrame(all_results)

    print("\nComparison:")
    print(results_df)

    if args.results_path:
        results_path = Path(args.results_path)
        results_df.to_csv(results_path, index = False)

    if args.tuning_path:
        tuning_path = Path(args.tuning_path)
        tuning_df.to_csv(tuning_path, index = False)


main()
