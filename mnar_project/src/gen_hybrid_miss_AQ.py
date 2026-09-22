import numpy as np
import pandas as pd
from pathlib import Path
import argparse


mnar_project = Path(__file__).resolve().parents[1]

input_path = mnar_project / "data" / "processed" / "AQ_clean.csv"
output_dir = mnar_project / "data" / "processed"

output_dir.mkdir(parents = True, exist_ok = True)

random_state = 100


target_configs = {
    "NOx(GT)": {
        "missing_rate": 0.10,
        "self_direction": "high",
        "mar_features": {
            "NO2(GT)": 0.6,
            "CO(GT)": 0.4,
            "PT08.S3(NOx)": -0.4,
            "T": 0.2,
        },
        "beta_mnar": 1.2,
        "beta_mar": 1.0,
        "noise_scale": 0.3,
    },
    "NO2(GT)": {
        "missing_rate": 0.10,
        "self_direction": "high",
        "mar_features": {
            "NOx(GT)": 0.5,
            "CO(GT)": 0.3,
            "PT08.S4(NO2)": -0.4,
            "RH": 0.2,
        },
        "beta_mnar": 1.0,
        "beta_mar": 1.0,
        "noise_scale": 0.3,
    },
    "CO(GT)": {
        "missing_rate": 0.10,
        "self_direction": "high",
        "mar_features": {
            "NOx(GT)": 0.4,
            "C6H6(GT)": 0.5,
            "PT08.S1(CO)": -0.4,
            "AH": 0.2,
        },
        "beta_mnar": 1.0,
        "beta_mar": 1.0,
        "noise_scale": 0.3,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type = str, default = str(input_path))
    parser.add_argument("--missing-rate", type = float, default = None)
    parser.add_argument("--output-name", type = str, default = "AQ_mnar.csv")
    parser.add_argument("--mask-name", type = str, default = "mask_AQ_mnar.csv")

    return parser.parse_args()


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def standardize_series(series):
    mean = series.mean()
    std = series.std()

    if std == 0:
        return pd.Series(0.0, index = series.index)

    return (series - mean) / std


def build_self_signal(df, column, direction):
    standardized = standardize_series(df[column].astype(float))

    if direction == "high":
        return standardized.clip(lower=0)

    if direction == "low":
        return (-standardized).clip(lower=0)

    if direction == "both":
        return standardized.abs()

    raise ValueError(f"Unknown self_direction: {direction}")


def build_observed_signal(df, mar_features):
    if not mar_features:
        return pd.Series(0.0, index = df.index)

    signal = pd.Series(0.0, index = df.index)

    for feature, weight in mar_features.items():
        standardized_feature = standardize_series(df[feature].astype(float))
        signal += weight * standardized_feature

    return signal


def calibrate_alpha(base_score, target_missing_rate, tolerance = 0.0005, max_iter = 100):
    low = -20
    high = 20

    for _ in range(max_iter):
        mid = (low + high) / 2
        probabilities = sigmoid(mid + base_score)
        current_rate = probabilities.mean()

        if abs(current_rate - target_missing_rate) <= tolerance:
            return mid

        if current_rate < target_missing_rate:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def dominant_mechanism(mnar_component, mar_component, noise_component):
    component_values = {
        "mnar": abs(mnar_component),
        "mar": abs(mar_component),
        "noise": abs(noise_component),
    }

    return max(component_values, key = component_values.get)


def generate_missingness_for_column(df_original, df_corrupted, column, config, rng):
    target_missing_rate = config["missing_rate"]
    beta_mnar = config["beta_mnar"]
    beta_mar = config["beta_mar"]
    noise_scale = config["noise_scale"]

    self_signal = build_self_signal(
        df=df_original,
        column=column,
        direction=config["self_direction"]
    )

    observed_signal = build_observed_signal(
        df=df_original,
        mar_features=config["mar_features"]
    )

    noise = pd.Series(
        rng.normal(loc=0.0, scale=noise_scale, size=len(df_original)),
        index = df_original.index
    )

    mnar_component = beta_mnar * self_signal
    mar_component = beta_mar * observed_signal
    noise_component = noise

    base_score = mnar_component + mar_component + noise_component

    alpha = calibrate_alpha(
        base_score=base_score,
        target_missing_rate=target_missing_rate
    )

    raw_score = alpha + base_score
    p_missing = pd.Series(sigmoid(raw_score), index = df_original.index)

    missing_indicator = pd.Series(
        rng.binomial(n = 1, p = p_missing),
        index = df_original.index
    )

    missing_idx = missing_indicator[missing_indicator == 1].index

    df_corrupted.loc[missing_idx, column] = pd.NA

    mask_rows = []

    for idx in df_original.index:
        is_missing = int(missing_indicator.loc[idx])

        row = {
            "row_id": idx,
            "column": column,
            "is_missing": is_missing,
            "ground_truth": df_original.loc[idx, column] if is_missing else pd.NA,
            "p_missing": p_missing.loc[idx],
            "raw_score": raw_score.loc[idx],
            "alpha": alpha,
            "mnar_component": mnar_component.loc[idx],
            "mar_component": mar_component.loc[idx],
            "noise_component": noise_component.loc[idx],
            "dominant_mechanism": dominant_mechanism(
                mnar_component.loc[idx],
                mar_component.loc[idx],
                noise_component.loc[idx],
            ),
            "target_missing_rate": target_missing_rate,
            "actual_missing_rate": missing_indicator.mean(),
            "beta_mnar": beta_mnar,
            "beta_mar": beta_mar,
            "noise_scale": noise_scale,
            "self_direction": config["self_direction"],
        }

        mask_rows.append(row)

    return df_corrupted, mask_rows


def generate_hybrid_missingness(df, target_configs, random_state = 100):
    rng = np.random.default_rng(random_state)

    df_original = df.copy()
    df_corrupted = df.copy()

    all_mask_rows = []

    for column, config in target_configs.items():

        if column not in df.columns:
            raise ValueError(f"not in dataset {column}")

        df_corrupted, mask_rows = generate_missingness_for_column(
            df_original=df_original,
            df_corrupted=df_corrupted,
            column=column,
            config=config,
            rng=rng
        )

        all_mask_rows.extend(mask_rows)

        column_mask = pd.DataFrame(mask_rows)
        actual_rate = column_mask["is_missing"].mean()


    mask = pd.DataFrame(all_mask_rows)

    return df_corrupted, mask


def main():
    args = parse_args()
    current_input_path = Path(args.input_path)
    df = pd.read_csv(current_input_path)

    current_target_configs = {
        column: config.copy()
        for column, config in target_configs.items()
    }

    if args.missing_rate is not None:
        for column in current_target_configs:
            current_target_configs[column]["missing_rate"] = args.missing_rate

    print("Input shape:", df.shape)
    print("Target columns:", list(current_target_configs.keys()))

    df_corrupted, mask = generate_hybrid_missingness(
        df = df,
        target_configs = current_target_configs,
        random_state = random_state
    )

    corrupted_path = output_dir / args.output_name
    mask_path = output_dir / args.mask_name

    df_corrupted.to_csv(corrupted_path, index = False)
    mask.to_csv(mask_path, index = False)

    print("\nsaved missing dataset to")
    print(corrupted_path)

    print("\nsaved mask to")
    print(mask_path)

    print("\nnumber of missing values")
    print(df_corrupted.isna().sum())

    print("\ndominant mechanism among actually missing values")
    print(
        mask[mask["is_missing"] == 1]
        .groupby(["column", "dominant_mechanism"])
        .size()
    )


if __name__ == "__main__":
    main()
