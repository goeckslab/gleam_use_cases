from pathlib import Path

import requests
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
CSV_URL = "https://zenodo.org/records/17702692/files/HAM10000_metadata.csv?download=1"
DATASET_DIR = Path("HAM10000_dataset")
RAW_CSV_PATH = DATASET_DIR / "HAM10000_metadata.csv"
PREP_CSV_PATH = DATASET_DIR / "HAM10000_metadata_preprocessed.csv"
SAMPLES_PER_CLASS = 200
RANDOM_STATE = 42


# -------------------------------------------------------------------------
# 1) Download CSV
# -------------------------------------------------------------------------
def download_metadata(url: str, dest_path: Path) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading metadata from:\n  {url}")
    resp = requests.get(url)
    resp.raise_for_status()

    with open(dest_path, "wb") as f:
        f.write(resp.content)

    print(f"Saved metadata to: {dest_path}")


# -------------------------------------------------------------------------
# 2) Prepare stratified, lesion-unique CSV
# -------------------------------------------------------------------------
def prepare_preprocessed_csv(
    raw_csv_path: Path,
    out_csv_path: Path,
    samples_per_class: int = SAMPLES_PER_CLASS,
    random_state: int = RANDOM_STATE,
) -> None:
    df = pd.read_csv(raw_csv_path)

    # Expecting at least: lesion_id, image_id, dx
    required_cols = {"lesion_id", "image_id", "dx"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    rng = np.random.RandomState(random_state)
    sampled_parts = []

    # For each diagnosis (class), sample up to N lesions, 1 image per lesion
    for dx_value in sorted(df["dx"].unique()):
        df_class = df[df["dx"] == dx_value]

        # One random row per lesion within this class
        df_class = (
            df_class.sample(frac=1.0, random_state=rng)  # shuffle
            .drop_duplicates(subset="lesion_id", keep="first")
        )

        n_available = len(df_class)
        n_to_sample = min(samples_per_class, n_available)

        if n_to_sample == 0:
            continue

        df_sampled = df_class.sample(
            n=n_to_sample,
            random_state=rng
        )
        sampled_parts.append(df_sampled)

    if not sampled_parts:
        raise RuntimeError("No samples found for any class.")

    sampled_df = pd.concat(sampled_parts, ignore_index=True)

    # Add required columns:
    # - label: copy of dx
    # - image_path: content of image_id
    sampled_df = sampled_df.copy()
    sampled_df["label"] = sampled_df["dx"]
    sampled_df["image_path"] = sampled_df["image_id"]

    # Ensure global uniqueness of lesion_id across final set (safety)
    sampled_df = (
        sampled_df.sample(frac=1.0, random_state=rng)  # shuffle
        .drop_duplicates(subset="lesion_id", keep="first")
        .reset_index(drop=True)
    )

    # At the end keep ONLY: lesion_id, image_path, label
    sampled_df = sampled_df[["lesion_id", "image_path", "label"]]

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(out_csv_path, index=False)
    print(f"Saved preprocessed CSV to: {out_csv_path}")
    print(f"Total rows: {len(sampled_df)}")
    print("Class distribution (label):")
    print(sampled_df["label"].value_counts())


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    if not RAW_CSV_PATH.exists():
        download_metadata(CSV_URL, RAW_CSV_PATH)
    else:
        print(f"Metadata CSV already exists at: {RAW_CSV_PATH}")

    prepare_preprocessed_csv(RAW_CSV_PATH, PREP_CSV_PATH)


if __name__ == "__main__":
    main()

