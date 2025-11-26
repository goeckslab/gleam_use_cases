from argparse import ArgumentParser
from pathlib import Path
import pandas as pd


SPLIT_FILE_MAP = {
    "in_distribution": "dataset_split_in.json",
    "out_distribution": "dataset_split_out.json",
    "oropharynx": "dataset_split_Oropharynx.json",
}

# ICD code columns and CD3/CD8 columns to drop in non-paper_like mode
ICD_CODE_COLS = [
    "c020", "c021", "c022", "c028", "c029", "c048", "c051", "c052", "c058",
    "c068", "c090", "c091", "c098", "c099", "c100", "c102", "c108", "c109",
    "c111", "c130", "c131", "c132", "c138", "c139", "c148", "c320", "c321",
    "c322", "c323", "c328", "c329", "c770", "c778", "c800", "d000", "d370",
    "d380", "r590", "r599", "t810",
]

CD3_CD8_COLS = ["cd3_z", "cd3_inv", "cd8_z", "cd8_inv"]


def _normalize_patient_id_series(s: pd.Series) -> pd.Series:
    """Normalize patient_id values to a canonical non-padded numeric string.

    Strategy:
    - Convert values that look numeric to integers then back to str (removes leading zeros)
    - For non-numeric values, strip whitespace and leading zeros
    - Preserve missing values as NA
    """
    if s is None:
        return s
    s = s.astype(str).str.strip()
    s = s.replace({"nan": pd.NA, "None": pd.NA})

    nums = pd.to_numeric(s, errors="coerce")
    out = s.copy()
    mask = nums.notna()
    if mask.any():
        out.loc[mask] = nums.loc[mask].astype("Int64").astype(str)

    nonmask = ~mask
    if nonmask.any():
        out.loc[nonmask] = out.loc[nonmask].astype(str).str.lstrip("0")

    out = out.replace({"": pd.NA})
    return out


def load_features(features_dir: Path, paper_like: bool) -> pd.DataFrame:
    """Load and outer-merge feature tables on patient_id.

    Always includes: clinical, pathological, blood, tma_cell_density.
    Only includes: icd_text and CD3_CD8_raw_images when paper_like is False.
    """
    clinical = pd.read_csv(features_dir / "clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(features_dir / "pathological.csv", dtype={"patient_id": str})
    blood = pd.read_csv(features_dir / "blood.csv", dtype={"patient_id": str})
    cell_density = pd.read_csv(features_dir / "tma_cell_density.csv", dtype={"patient_id": str})

    # Normalize patient_id columns to canonical form across all sources
    clinical["patient_id"] = _normalize_patient_id_series(clinical.get("patient_id"))
    patho["patient_id"] = _normalize_patient_id_series(patho.get("patient_id"))
    blood["patient_id"] = _normalize_patient_id_series(blood.get("patient_id"))
    cell_density["patient_id"] = _normalize_patient_id_series(cell_density.get("patient_id"))

    # Base merges (always included)
    df = (
        clinical.merge(patho, on="patient_id", how="outer")
        .merge(blood, on="patient_id", how="outer")
        .merge(cell_density, on="patient_id", how="outer")
        .reset_index(drop=True)
    )

    # Only include ICD and CD3/CD8 when not paper_like
    if not paper_like:
        # Use the textual ICD file (icd_text.csv) which contains free-text
        icd = pd.read_csv(features_dir / "icd_text.csv", dtype={"patient_id": str})
        cd3_cd8 = pd.read_csv(features_dir / "CD3_CD8_raw_images.csv", dtype={"patient_id": str})

        # Normalize their patient_id columns as well
        icd["patient_id"] = _normalize_patient_id_series(icd.get("patient_id"))
        cd3_cd8["patient_id"] = _normalize_patient_id_series(cd3_cd8.get("patient_id"))

        df = (
            df.merge(icd, on="patient_id", how="outer")
              .merge(cd3_cd8, on="patient_id", how="outer")
        )

    return df


def load_targets(features_dir: Path) -> pd.DataFrame:
    """Load targets table."""
    targets = pd.read_csv(features_dir / "targets.csv", dtype={"patient_id": str})
    targets["patient_id"] = _normalize_patient_id_series(targets.get("patient_id"))
    return targets


def build_dataset_for(
    target_label: str,
    split_type: str,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    datasplit_dir: Path,
    output_dir: Path,
    paper_like: bool,
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
    df_split["patient_id"] = _normalize_patient_id_series(df_split.get("patient_id"))

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

    # In non-paper_like mode, drop ICD code columns and CD3/CD8 columns
    if not paper_like:
        cols_to_drop = ICD_CODE_COLS + CD3_CD8_COLS
        df_merged = df_merged.drop(columns=cols_to_drop, errors="ignore")

    # Re-order columns: patient_id, dataset, target, then all remaining features
    base_cols = ["patient_id", "dataset", "target"]
    other_cols = [c for c in df_merged.columns if c not in base_cols]
    df_final = df_merged[base_cols + other_cols]

    # Ensure output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename: e.g., recurrence_in_distribution.csv or recurrence_in_distribution_paper_like.csv
    suffix = "_paper_like" if paper_like else ""
    outfile = output_dir / f"{target_label}_{split_type}{suffix}.csv"
    df_final.to_csv(outfile, index=False)

    print(
        f"Wrote {len(df_final)} rows to {outfile} "
        f"(target={target_label}, split_type={split_type}, paper_like={paper_like})"
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
    parser.add_argument(
        "--paper_like",
        action="store_true",
        help=(
            "If set, DO NOT read/merge icd_text.csv or CD3_CD8_raw_images.csv; "
            "produce the paper-like feature set and append '_paper_like' to the output filename."
        ),
    )

    args = parser.parse_args()

    # Hard-coded locations relative to this script
    root_dir = Path(__file__).resolve().parent
    data_dir = root_dir / "paper_dataset"  # contains features + split JSONs

    datasplit_dir = data_dir
    features_dir = data_dir
    output_dir = Path(args.output_directory)

    features_df = load_features(features_dir, paper_like=args.paper_like)
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
                    paper_like=args.paper_like,
                )
    else:
        build_dataset_for(
            args.target_label,
            args.split_type,
            features_df,
            targets_df,
            datasplit_dir,
            output_dir,
            paper_like=args.paper_like,
        )


if __name__ == "__main__":
    main()
