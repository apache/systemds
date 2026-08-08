"""
Rebalanced MICE-like prototype for MNAR imputation on numeric CSV datasets

core idea:
 - MICE-like loop over multiple incomplete target columns
 - target weights for rare or skewed target regions
- similarity weights for observed rows that resemble missing rows
- validation-based selection of the rebalancing configuration

This version represents the final specialized prototype used for the main experiments.
Prerequisites:
- CSV with numerical columns
- mask-CSV with row_id, column, is_missing, and ground_truth
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score


mnar_project = Path(__file__).resolve().parents[1]

file_path = mnar_project / "data" / "processed" / "diabetes_mnar.csv"
mask_path = mnar_project / "data" / "processed" / "mask_diabetes_mnar.csv"
results = mnar_project / "results"
results.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(file_path)
mask = pd.read_csv(mask_path)

columns_with_missing = ["BMI", "Income", "MentHlth"]
max_iterations = 5
random_state = 100
validation_share = 0.2

candidate_configs = [
    {
        "name": "weighted_only",
        "n_bins": 6,
        "smoothing": 10.0,
        "max_weight_target": 4.0,
        "scew_threshold": 0.25,
        "max_weight_similarity": 2.0,
        "final_weight_cap": 6.0,
        "oversampling_method": "none",
        "oversample_factor": 1.0,
    },
    {
        "name": "weighted_stronger",
        "n_bins": 6,
        "smoothing": 8.0,
        "max_weight_target": 5.0,
        "scew_threshold": 0.25,
        "max_weight_similarity": 3.0,
        "final_weight_cap": 8.0,
        "oversampling_method": "none",
        "oversample_factor": 1.0,
    },
    {
        "name": "weighted_simple_oversampling",
        "n_bins": 6,
        "smoothing": 8.0,
        "max_weight_target": 5.0,
        "scew_threshold": 0.25,
        "max_weight_similarity": 3.0,
        "final_weight_cap": 8.0,
        "oversampling_method": "simple",
        "oversample_factor": 1.25,
    },
    {
        "name": "weighted_smote",
        "n_bins": 6,
        "smoothing": 8.0,
        "max_weight_target": 5.0,
        "scew_threshold": 0.25,
        "max_weight_similarity": 3.0,
        "final_weight_cap": 8.0,
        "oversampling_method": "smote",
        "oversample_factor": 1.25,
    },
]


# Gewichte stabilisieren, sd. Modell nicht durch zu hohe/niedrige Gewichte instabil wird
# (relative Gewichtung bleibt erhalten)
def stabilizing(weights):

    weights = weights.astype(float)

    if weights.mean() == 0:
        weights = weights + 1.0

        return weights

    weights = weights / weights.mean()

    return weights


def initial_fill_complete(df, columns_with_missing):

    df_imputed = df.copy()

    for column_missing in columns_with_missing:
        observed = df_imputed.loc[df_imputed[column_missing].notna(), column_missing]
        observed_mean = observed.mean()
        df_imputed[column_missing] = df_imputed[column_missing].fillna(observed_mean)

    return df_imputed


# welche gemessenen Zeilen liegen im Scew-Bereich?
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


# Zeilen, die Zeilen mit fehlendem Wert ähneln, sollen stärker gewichtet werden
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


def build_bin_id(y_train, n_bins = 6):

    unique_vals = y_train.nunique()
    n_bins_effective = min(n_bins, unique_vals)

    if n_bins_effective <= 1:
        return pd.Series(0, index = y_train.index)

    return pd.cut(
        y_train,
        bins = n_bins_effective,
        labels = False,
        include_lowest = True,
        duplicates = "drop"
    )


def simple_oversample(x_train, y_train, final_weights, n_bins = 6, oversample_factor = 1.25, rng = None):

    if oversample_factor <= 1.0:
        return x_train, y_train, final_weights

    bin_id = build_bin_id(y_train, n_bins = n_bins)
    bin_counts = bin_id.value_counts().sort_index()
    target_count = int(np.ceil(bin_counts.max() * oversample_factor))

    x_extra = []
    y_extra = []
    weight_extra = []

    for current_bin, count in bin_counts.items():
        idx_in_bin = bin_id[bin_id == current_bin].index

        if count >= target_count:
            continue

        n_add = target_count - count
        sampled_idx = rng.choice(idx_in_bin.to_numpy(), size = n_add, replace = True)

        x_extra.append(x_train.loc[sampled_idx])
        y_extra.append(y_train.loc[sampled_idx])
        weight_extra.append(final_weights.loc[sampled_idx])

    if not x_extra:
        return x_train, y_train, final_weights

    x_aug = pd.concat([x_train] + x_extra, axis = 0)
    y_aug = pd.concat([y_train] + y_extra, axis = 0)
    w_aug = pd.concat([final_weights] + weight_extra, axis = 0)

    return x_aug, y_aug, w_aug


def smote_oversample(x_train, y_train, final_weights, n_bins = 6, oversample_factor = 1.25, rng = None):

    if oversample_factor <= 1.0:
        return x_train, y_train, final_weights

    bin_id = build_bin_id(y_train, n_bins = n_bins)
    bin_counts = bin_id.value_counts().sort_index()
    target_count = int(np.ceil(bin_counts.max() * oversample_factor))

    synthetic_x_rows = []
    synthetic_y_rows = []
    synthetic_w_rows = []

    for current_bin, count in bin_counts.items():
        idx_in_bin = bin_id[bin_id == current_bin].index

        if count >= target_count:
            continue

        n_add = target_count - count

        for _ in range(n_add):
            if len(idx_in_bin) == 1:
                idx_a = idx_in_bin[0]
                idx_b = idx_in_bin[0]
            else:
                idx_a, idx_b = rng.choice(idx_in_bin.to_numpy(), size = 2, replace = True)

            lam = rng.uniform(0.0, 1.0)

            x_a = x_train.loc[idx_a]
            x_b = x_train.loc[idx_b]
            y_a = y_train.loc[idx_a]
            y_b = y_train.loc[idx_b]
            w_a = final_weights.loc[idx_a]
            w_b = final_weights.loc[idx_b]

            x_syn = x_a + lam * (x_b - x_a)
            y_syn = y_a + lam * (y_b - y_a)
            w_syn = (w_a + w_b) / 2

            synthetic_x_rows.append(x_syn)
            synthetic_y_rows.append(y_syn)
            synthetic_w_rows.append(w_syn)

    if not synthetic_x_rows:
        return x_train, y_train, final_weights

    x_syn_df = pd.DataFrame(synthetic_x_rows, columns = x_train.columns)
    y_syn_series = pd.Series(synthetic_y_rows)
    w_syn_series = pd.Series(synthetic_w_rows)

    x_syn_df.index = range(x_train.index.max() + 1, x_train.index.max() + 1 + len(x_syn_df))
    y_syn_series.index = x_syn_df.index
    w_syn_series.index = x_syn_df.index

    x_aug = pd.concat([x_train, x_syn_df], axis = 0)
    y_aug = pd.concat([y_train, y_syn_series], axis = 0)
    w_aug = pd.concat([final_weights, w_syn_series], axis = 0)

    return x_aug, y_aug, w_aug


def apply_oversampling(x_train, y_train, final_weights, config, seed):

    rng = np.random.default_rng(seed)
    method = config["oversampling_method"]

    if method == "none":
        return x_train, y_train, final_weights

    if method == "simple":
        return simple_oversample(
            x_train = x_train,
            y_train = y_train,
            final_weights = final_weights,
            n_bins = config["n_bins"],
            oversample_factor = config["oversample_factor"],
            rng = rng
        )

    if method == "smote":
        return smote_oversample(
            x_train = x_train,
            y_train = y_train,
            final_weights = final_weights,
            n_bins = config["n_bins"],
            oversample_factor = config["oversample_factor"],
            rng = rng
        )

    raise ValueError(f"Unknown oversampling method: {method}")


def fit_one_column(df_input, missing_indicator, column, config, seed):

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

    x_train_aug, y_train_aug, final_weights_aug = apply_oversampling(
        x_train = x_train,
        y_train = y_train,
        final_weights = final_weights,
        config = config,
        seed = seed
    )

    scaler = StandardScaler()
    x_train_trans = scaler.fit_transform(x_train_aug)
    x_missing_trans = scaler.transform(x_missing)

    model = Ridge(
        alpha = 1.0,
        random_state = random_state
    )

    model.fit(
        x_train_trans,
        y_train_aug,
        sample_weight = final_weights_aug
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
        "oversampling_method": config["oversampling_method"],
        "oversample_factor": config["oversample_factor"],
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


def tune_rebalancing_for_column(df_start, column, candidate_configs):

    observed_idx = df_start[df_start[column].notna()].index
    y_observed = df_start.loc[observed_idx, column]

    validation_idx = draw_validation_idx(
        y_observed = y_observed,
        validation_share = validation_share,
        n_bins = 6
    )

    train_idx = observed_idx.difference(validation_idx)

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
            config = config,
            seed = random_state
        )

        actual_val = df_start.loc[validation_idx, column]
        pred_val = predictions.loc[validation_idx]

        errors = pred_val.reset_index(drop=True) - actual_val.reset_index(drop=True)
        mae = errors.abs().mean()
        bias = errors.mean()
        score = mae + 0.5 * abs(bias)

        if score < best_score:
            best_score = score
            best_config = config.copy()
            best_meta = {
                "validation_mae": mae,
                "validation_bias": bias,
                "validation_score": score,
                "validation_n": len(validation_idx),
                "train_n": len(train_idx),
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


def rebalanced_mice_complete(df, mask, columns_with_missing, max_iter):

    df_start = initial_fill_complete(df, columns_with_missing)
    df_imputed = df_start.copy()

    best_configs = {}
    tuning_rows = []

    print("\nvalidation for rebalancing configs")

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
            **best_meta
        }

        tuning_rows.append(tuning_row)
        print(f"{column_missing}: {best_config['name']} | score = {best_meta['validation_score']:.4f}")

    #print("\nmice loop")

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}")

        for column_missing in columns_with_missing:
            missing_indicator_real = df[column_missing].isna().astype(int)

            predictions, meta = fit_one_column(
                df_input = df_imputed,
                missing_indicator = missing_indicator_real,
                column = column_missing,
                config = best_configs[column_missing],
                seed = random_state + iteration
            )

            missing_idx = missing_indicator_real[missing_indicator_real == 1].index
            df_imputed.loc[missing_idx, column_missing] = predictions

            print(
                f"{column_missing}: auc = {meta['auc']:.4f} | "
                f"scew = {meta['scew']:.4f} | "
                f"oversampling = {meta['oversampling_method']}"
            )

    tuning_df = pd.DataFrame(tuning_rows)

    return df_imputed, tuning_df


df_rebalanced, tuning_df = rebalanced_mice_complete(
    df = df,
    mask = mask,
    columns_with_missing = columns_with_missing,
    max_iter = max_iterations
)

all_results = []

for column_missing in columns_with_missing:
    all_results.extend(
        eval_imp_complete(
            df_imputed = df_rebalanced,
            mask = mask,
            column = column_missing,
            method_name = "rebalanced_mice_complete"
        )
    )

results_df = pd.DataFrame(all_results)

print("\ncomparison")
print(results_df)

results_path = results / "rebalanced_mice_complete_diabetes_mnar.csv"
tuning_path = results / "rebalanced_mice_complete_tuning_diabetes_mnar.csv"

results_df.to_csv(results_path, index = False)
tuning_df.to_csv(tuning_path, index = False)
