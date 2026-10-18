
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "dataset.csv"


def main():
    print("Downloading AI4I 2020 Predictive Maintenance Dataset...")

    dataset = fetch_ucirepo(id=601)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    # Combine features and labels without duplicating column names.
    targets = targets.loc[
        :, ~targets.columns.isin(features.columns)
    ]
    df = pd.concat([features, targets], axis=1)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDownload completed.")
    print("File:", OUTPUT_FILE)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nSample data:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
