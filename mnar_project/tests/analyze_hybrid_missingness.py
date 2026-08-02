import os
from pathlib import Path

import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data" / "processed"
output_dir = base_dir / "tests" / "output" / "hybrid_missingness"
output_dir.mkdir(parents = True, exist_ok = True)

os.environ["MPLCONFIGDIR"] = str(output_dir / ".mpl_cache")
os.environ["XDG_CACHE_HOME"] = str(output_dir / ".mpl_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


original_path = data_dir / "diabetes_sample.csv"
corrupted_path = data_dir / "diabetes_mnar.csv"
mask_path = data_dir / "mask_diabetes_mnar.csv"

target_columns = ["BMI", "Income", "MentHlth"]

bin_configs = {
    "BMI": {
        "bin_edges": [-float("inf"), 30, 35, 40, float("inf")],
        "bin_labels": ["<30", "30-35", "35-40", "40+"],
        "plot_bins": 30,
    },
    "Income": {
        "bin_edges": [0.5, 2.5, 4.5, 6.5, 8.5],
        "bin_labels": ["1-2", "3-4", "5-6", "7-8"],
        "plot_bins": list(range(1, 10)),
    },
    "MentHlth": {
        "bin_edges": [-0.5, 0.5, 5.5, 15.5, 30.5],
        "bin_labels": ["0", "1-5", "6-15", "16-30"],
        "plot_bins": list(range(0, 32)),
    },
}


def load_data():
    original = pd.read_csv(original_path)
    corrupted = pd.read_csv(corrupted_path)
    mask = pd.read_csv(mask_path)
    return original, corrupted, mask


def split_series(original, corrupted, mask, column):
    column_mask = mask[mask["column"] == column].copy()
    missing_idx = column_mask.loc[column_mask["is_missing"] == 1, "row_id"]
    observed_idx = column_mask.loc[column_mask["is_missing"] == 0, "row_id"]

    original_values = original[column].copy()
    observed_values = corrupted.loc[observed_idx, column].copy()
    removed_values = column_mask.loc[column_mask["is_missing"] == 1, "ground_truth"].astype(float)

    return original_values, observed_values, removed_values, column_mask


def build_summary_rows(column, original_values, observed_values, removed_values):
    summaries = []
    groups = {
        "original": original_values,
        "observed": observed_values,
        "removed": removed_values,
    }

    for group_name, values in groups.items():
        row = {
            "column": column,
            "group": group_name,
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "q10": values.quantile(0.10),
            "q25": values.quantile(0.25),
            "q50": values.quantile(0.50),
            "q75": values.quantile(0.75),
            "q90": values.quantile(0.90),
        }
        summaries.append(row)

    return summaries


def assign_bins(values, column):
    config = bin_configs[column]
    return pd.cut(
        values,
        bins=config["bin_edges"],
        labels=config["bin_labels"],
        include_lowest = True
    )


def build_missingness_rate_rows(column, original_values, column_mask):
    rates_df = pd.DataFrame({
        "value": original_values,
        "is_missing": column_mask["is_missing"].values,
    })
    rates_df["bin"] = assign_bins(rates_df["value"], column)

    grouped = (
        rates_df.groupby("bin", observed=False)
        .agg(
            n=("is_missing", "size"),
            missing_rate=("is_missing", "mean"),
            mean_value=("value", "mean"),
        )
        .reset_index()
    )
    grouped.insert(0, "column", column)
    return grouped


def plot_distributions(column, original_values, observed_values, removed_values):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    config = bin_configs[column]

    ax.hist(
        original_values,
        bins=config["plot_bins"],
        alpha=0.4,
        density=True,
        label = "Original",
    )
    ax.hist(
        observed_values,
        bins=config["plot_bins"],
        alpha=0.4,
        density=True,
        label = "Observed",
    )
    ax.hist(
        removed_values,
        bins=config["plot_bins"],
        alpha=0.4,
        density=True,
        label = "Removed",
    )

    ax.set_title(f"{column}: Original vs Observed vs Removed")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.legend()

    fig.tight_layout()
    out_path = output_dir / f"{column.lower()}_distribution.png"
    fig.savefig(out_path, dpi = 200, bbox_inches = "tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")


def plot_missingness_rate(column, rate_df):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))

    sns.barplot(
        data=rate_df,
        x="bin",
        y="missing_rate",
        ax=ax,
        color=sns.color_palette("deep")[0]
    )

    ax.set_title(f"{column}: Missingness Rate by Value Range")
    ax.set_xlabel("Value range")
    ax.set_ylabel("Missingness rate")
    ax.set_ylim(0, max(0.01, rate_df["missing_rate"].max() * 1.15))

    fig.tight_layout()
    out_path = output_dir / f"{column.lower()}_missingness_rate.png"
    fig.savefig(out_path, dpi = 200, bbox_inches = "tight")
    plt.close(fig)

    print(f"Saved plot: {out_path}")


def main():
    original, corrupted, mask = load_data()

    summary_rows = []
    rate_frames = []

    for column in target_columns:
        original_values, observed_values, removed_values, column_mask = split_series(
            original, corrupted, mask, column
        )

        summary_rows.extend(
            build_summary_rows(column, original_values, observed_values, removed_values)
        )

        rate_df = build_missingness_rate_rows(column, original_values, column_mask)
        rate_frames.append(rate_df)

        plot_distributions(column, original_values, observed_values, removed_values)
        plot_missingness_rate(column, rate_df)

    summary_df = pd.DataFrame(summary_rows)
    rate_df = pd.concat(rate_frames, ignore_index = True)

    summary_path = output_dir / "distribution_summary.csv"
    rate_path = output_dir / "missingness_rate_by_bin.csv"

    summary_df.to_csv(summary_path, index = False)
    rate_df.to_csv(rate_path, index = False)

    print("\ndistribution summary")
    print(summary_df.to_string(index=False))

    print("\nmissingness rate per bin")
    print(rate_df.to_string(index=False))

    print(f"\nsummary table: {summary_path}")
    print(f"missingness rate table: {rate_path}")


if __name__ == "__main__":
    main()
