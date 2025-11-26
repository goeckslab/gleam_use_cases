from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


SPLIT_FILE_MAP = {
    "in_distribution": "dataset_split_in.json",
    "out_distribution": "dataset_split_out.json",
    "oropharynx": "dataset_split_Oropharynx.json",
}


def load_features(features_dir: Path) -> pd.DataFrame:
    """Load and outer-merge all feature tables on patient_id."""
    clinical = pd.read_csv(features_dir / "clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(features_dir / "pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(features_dir / "blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(features_dir / "icd_codes.csv", dtype={"patient_id": str})
    cell_density = pd.read_csv(features_dir / "tma_cell_density.csv", dtype={"patient_id": str})

    df = (
        clinical.merge(patho, on="patient_id", how="outer")
        .merge(blood, on="patient_id", how="outer")
        .merge(icd, on="patient_id", how="outer")
        .merge(cell_density, on="patient_id", how="outer")
        .reset_index(drop=True)
    )
    return df


def load_targets(features_dir: Path) -> pd.DataFrame:
    """Load targets table."""
    return pd.read_csv(features_dir / "targets.csv", dtype={"patient_id": str})


def build_dataset_for(
    target_label: str,
    split_type: str,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    datasplit_dir: Path,
    output_dir: Path,
) -> Path:
    """Build a single CSV dataset for a given target_label and split_type."""

    if split_type not in SPLIT_FILE_MAP:
        raise ValueError(f"Unsupported split_type: {split_type}")

    split_file = datasplit_dir / SPLIT_FILE_MAP[split_type]
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    # Columns coming from targets.csv (we will remove them later, except patient_id)
    target_cols_original = [c for c in targets_df.columns if c != "patient_id"]

    # Load split definition (patient_id + dataset)
    df_split = pd.read_json(split_file, dtype={"patient_id": str})[["patient_id", "dataset"]]

    # Attach targets
    df = df_split.merge(targets_df, on="patient_id", how="inner")

    # Apply target-specific filtering and create numeric target
    if target_label == "recurrence":
        required_cols = [
            "recurrence",
            "days_to_recurrence",
            "days_to_last_information",
            "survival_status",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for recurrence: {missing}")

        df = df[
            ((df["recurrence"] == "yes") & (df["days_to_recurrence"] <= 365 * 3))
            | (
                (df["recurrence"] == "no")
                & (
                    (df["days_to_last_information"] > 365 * 3)
                    | (df["survival_status"] == "living")
                )
            )
        ].copy()
        df["target"] = df["recurrence"].map({"no": 0, "yes": 1}).astype("int64")

    elif target_label == "survival_status":
        required_cols = ["survival_status", "survival_status_with_cause"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for survival_status: {missing}")

        df = df[df["survival_status_with_cause"] != "deceased not tumor specific"].copy()
        df["target"] = df["survival_status"].map({"living": 0, "deceased": 1}).astype("int64")

    else:
        raise ValueError(f"Unsupported target_label: {target_label}")

    # Merge with features
    df_merged = df.merge(features_df, on="patient_id", how="inner")

    # Map dataset to numeric: training -> 0, test -> 2
    dataset_map = {"training": 0, "test": 2}
    df_merged["dataset"] = df_merged["dataset"].map(dataset_map).astype("Int64")

    # Drop ALL original columns coming from targets.csv (except patient_id)
    df_merged = df_merged.drop(columns=target_cols_original, errors="ignore")

    # Re-order columns: patient_id, dataset, target, then all remaining features
    base_cols = ["patient_id", "dataset", "target"]
    other_cols = [c for c in df_merged.columns if c not in base_cols]
    df_final = df_merged[base_cols + other_cols]

    # Ensure output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: e.g., recurrence_in_distribution.csv
    outfile = output_dir / f"{target_label}_{split_type}.csv"
    df_final.to_csv(outfile, index=False)

    print(
        f"Wrote {len(df_final)} rows to {outfile} "
        f"(target={target_label}, split_type={split_type})"
    )
    return outfile


def main():
    parser = ArgumentParser(
        description=(
            "Build merged tabular datasets (CSV) for outcome prediction. "
            "Each CSV row is one patient with merged features and target."
        )
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Path to directory where CSV datasets will be saved",
    )
    parser.add_argument(
        "--target_label",
        type=str,
        choices=["recurrence", "survival_status"],
        default="recurrence",
        help="Outcome to encode as numeric label (default: recurrence)",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        choices=["in_distribution", "out_distribution", "oropharynx"],
        default="in_distribution",
        help="Which split definition to use (default: in_distribution)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "If set, generate all combinations of target_label "
            "and split_type, ignoring --target_label/--split_type."
        ),
    )

    args = parser.parse_args()

    # Hard-coded locations relative to this script
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "paper_dataset"  # contains features + split JSONs

    datasplit_dir = data_dir
    features_dir = data_dir
    output_dir = Path(args.output_directory)

    features_df = load_features(features_dir)
    targets_df = load_targets(features_dir)

    if args.all:
        for target_label in ["recurrence", "survival_status"]:
            for split_type in ["in_distribution", "out_distribution", "oropharynx"]:
                build_dataset_for(
                    target_label,
                    split_type,
                    features_df,
                    targets_df,
                    datasplit_dir,
                    output_dir,
                )
    else:
        build_dataset_for(
            args.target_label,
            args.split_type,
            features_df,
            targets_df,
            datasplit_dir,
            output_dir,
        )


if __name__ == "__main__":
    main()

