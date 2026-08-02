from pathlib import Path

import pandas as pd


base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data" / "processed"

corrupted_path = data_dir / "diabetes_mnar.csv"
mask_path = data_dir / "mask_diabetes_mnar.csv"

target_columns = ["BMI", "Income", "MentHlth"]
target_rate = 0.10
rate_tolerance = 0.02


def test_hybrid_files_exist():
    assert corrupted_path.exists(), f"missing in corrupted data {corrupted_path}"
    assert mask_path.exists(), f"missing mask data {mask_path}"


def test_hybrid_missingness_mask_matches_dataset():
    df = pd.read_csv(corrupted_path)
    mask = pd.read_csv(mask_path)

    for column in target_columns:
        column_mask = mask[mask["column"] == column].copy()

        assert len(column_mask) == len(df), (
            f"Maske für{column} sollte pro datensatz eine zeile enthalten"
        )

        mask_missing_idx = column_mask.loc[column_mask["is_missing"] == 1, "row_id"]
        dataset_missing_idx = df.index[df[column].isna()]

        assert set(mask_missing_idx) == set(dataset_missing_idx), (
            f"Indizes der Maske und des Datensatzen stimmen nicht überein in:  {column}."
        )


def test_missing_values_are_nan_not_zero():
    df = pd.read_csv(corrupted_path)
    mask = pd.read_csv(mask_path)

    for column in target_columns:
        column_mask = mask[mask["column"] == column].copy()
        missing_idx = column_mask.loc[column_mask["is_missing"] == 1, "row_id"]

        missing_values = df.loc[missing_idx, column]

        assert missing_values.isna().all(), (
            f"{column} nicht alle NaN"
        )


def test_target_missing_rates_are_close_to_config():
    mask = pd.read_csv(mask_path)

    for column in target_columns:
        column_mask = mask[mask["column"] == column].copy()
        actual_rate = column_mask["is_missing"].mean()

        assert abs(actual_rate - target_rate) <= rate_tolerance, (
            f"Ausfallquote in {column} ist {actual_rate:.4f}, "
            f"aber erwartet wurden {target_rate:.4f}."
        )
