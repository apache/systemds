from pathlib import Path

import pandas as pd


base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data" / "processed"

corrupted_path = data_dir / "diabetes_mnar.csv"
mask_path = data_dir / "mask_diabetes_mnar.csv"
source_path = data_dir / "diabetes_sample.csv"


def get_removed_idx(mask, column):
    column_mask = mask[mask["column"] == column].copy()
    return column_mask.loc[column_mask["is_missing"] == 1, "row_id"]


def build_missingness_rate_by_bin(source, removed_idx, column):

    if column == "BMI":
        bin_id = pd.cut(
            source[column],
            bins = [-float("inf"), 30, 35, 40, float("inf")],
            labels = ["<30", "30-35", "35-40", "40+"]
        )

    elif column == "Income":
        bin_id = pd.cut(
            source[column],
            bins = [0, 2, 4, 6, 8],
            labels = ["1-2", "3-4", "5-6", "7-8"],
            include_lowest = True
        )

    elif column == "MentHlth":
        bin_id = pd.cut(
            source[column],
            bins = [-1, 0, 5, 15, 30],
            labels = ["0", "1-5", "6-15", "16-30"],
            include_lowest = True
        )

    else:
        raise ValueError(f"Unknown column: {column}")

    tmp = pd.DataFrame({
        "bin": bin_id,
        "is_missing": 0
    }, index = source.index)

    tmp.loc[removed_idx, "is_missing"] = 1

    rates = tmp.groupby("bin", observed = False)["is_missing"].mean()

    return rates


def test_removed_values_show_expected_distribution_shift():
    df = pd.read_csv(corrupted_path)
    mask = pd.read_csv(mask_path)
    source = pd.read_csv(source_path)

    expected_direction = {
        "BMI": "higher",
        "Income": "lower",
        "MentHlth": "higher",
    }

    for column, direction in expected_direction.items():
        removed_idx = get_removed_idx(mask, column)

        observed = df.loc[df[column].notna(), column]
        removed = source.loc[removed_idx, column]

        if direction == "higher":
            assert removed.mean() > observed.mean(), (
                f" {column} sollten höheren mean als beobachtete Werte"
            )
        else:
            assert removed.mean() < observed.mean(), (
                f"{column} sollten niedrigeren mean als beobachtete Werte haben"
            )


def test_missingness_rates_follow_expected_bin_patterns():
    mask = pd.read_csv(mask_path)
    source = pd.read_csv(source_path)

    expected_order = {
        "BMI": ["<30", "30-35", "35-40", "40+"],
        "Income": ["1-2", "3-4", "5-6", "7-8"],
        "MentHlth": ["0", "1-5", "6-15", "16-30"],
    }

    expected_direction = {
        "BMI": "increasing",
        "Income": "decreasing",
        "MentHlth": "increasing",
    }

    for column, bin_order in expected_order.items():
        removed_idx = get_removed_idx(mask, column)
        rates = build_missingness_rate_by_bin(
            source = source,
            removed_idx = removed_idx,
            column = column
        )

        ordered_rates = rates.loc[bin_order].tolist()

        if expected_direction[column] == "increasing":
            assert ordered_rates == sorted(ordered_rates), (
                f"{column} Missingness raten sollten ansteigen."
            )
        else:
            assert ordered_rates == sorted(ordered_rates, reverse = True), (
                f"{column} sollten abfallen"
            )
