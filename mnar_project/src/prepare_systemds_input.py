from pathlib import Path

import pandas as pd


base_dir = Path(__file__).resolve().parents[1]
data_dir = base_dir / "data" / "processed"


def save_for_systemds(input_name, output_name, numeric_only = False):
    file_path = data_dir / input_name
    output_path = data_dir / output_name

    df = pd.read_csv(file_path)

    if numeric_only:
        df = df.select_dtypes(include = ["number"]).copy()

    df.to_csv(output_path, index=False, na_rep="NaN")

    print(f"saved {output_path}")


if __name__ == "__main__":
    save_for_systemds("diabetes_mnar.csv", "diabetes_mnar_systemds.csv")
    save_for_systemds("diabetes_full_mnar.csv", "diabetes_full_mnar_systemds.csv")
    save_for_systemds("AQ_mnar.csv", "AQ_mnar_systemds.csv", numeric_only = True)
