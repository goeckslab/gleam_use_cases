#!/usr/bin/env python
"""
LORIS → Tabular Learner Preprocessor
Generates clean, supervised TSV files (with 'Response' column only)
Fixed, tested, and super verbose
"""

import argparse
from pathlib import Path
import pandas as pd
import requests
from io import BytesIO
from typing import List


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
DEFAULT_URL = "https://raw.githubusercontent.com/rootchang/LORIS/main/02.Input/AllData.xlsx"
DEFAULT_SHEETS = ["Chowell_train", "Chowell_test"]
DEFAULT_OUTPUT_DIR = Path("loris_processed")


# ----------------------------------------------------------------------
# Core functions
# ----------------------------------------------------------------------
def truncate_values(df: pd.DataFrame) -> pd.DataFrame:
    print("   → Clipping outliers: TMB ≤ 50, Age ≤ 85, NLR ≤ 25")
    df = df.copy()
    df["TMB"] = df["TMB"].clip(upper=50)
    df["Age"] = df["Age"].clip(upper=85)
    df["NLR"] = df["NLR"].clip(upper=25)
    return df


def get_feature_columns() -> List[str]:
    return [
        'TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age',
        'CancerType1', 'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5',
        'CancerType6', 'CancerType7', 'CancerType8', 'CancerType9', 'CancerType10',
        'CancerType11', 'CancerType12', 'CancerType13', 'CancerType14', 'CancerType15', 'CancerType16',
        'Response'
    ]


def download_excel(url: str) -> BytesIO:
    print(f"Downloading Excel file from:")
    print(f"   {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    print("Download completed successfully!")
    return BytesIO(response.content)


def load_excel_sheets(excel_bytes: BytesIO, requested_sheets: List[str]) -> dict:
    print("Opening Excel file to read sheet names...")
    excel_file = pd.ExcelFile(excel_bytes, engine="openpyxl")
    available = excel_file.sheet_names
    print(f"Found {len(available)} sheets: {', '.join(available)}")

    sheets_to_load = [s for s in requested_sheets if s in available]
    missing = set(requested_sheets) - set(sheets_to_load)

    if missing:
        print(f"Warning: These sheets were not found → skipped: {missing}")
    if not sheets_to_load:
        raise ValueError("No valid sheets to process! Check sheet names.")

    print(f"Loading {len(sheets_to_load)} sheet(s): {sheets_to_load}")

    data = {}
    for sheet in sheets_to_load:
        print(f"   → Loading sheet: {sheet}")
        df = pd.read_excel(excel_bytes, sheet_name=sheet, engine="openpyxl")
        print(f"      Done: {df.shape[0]:,} rows × {df.shape[1]} columns")
        data[sheet] = df
    return data


def process_sheet(df: pd.DataFrame, sheet_name: str, output_dir: Path) -> None:
    print(f"Processing sheet → {sheet_name}")

    cols = get_feature_columns()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in '{sheet_name}': {missing}")

    df_clean = df[cols].copy()
    df_clean = truncate_values(df_clean)

    filepath = output_dir / f"{sheet_name}.tsv"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(filepath, sep="\t", index=False)
    print(f"Saved → {filepath}")
    print(f"   Shape: {df_clean.shape[0]:,} samples × {df_clean.shape[1]} features (including 'Response')")
    print(f"   Ready for Tabular Learner!\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LORIS dataset → clean supervised TSVs (with Response column)"
    )
    parser.add_argument("-u", "--url", default=DEFAULT_URL,
                        help="URL of AllData.xlsx")
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for TSV files")
    parser.add_argument("-s", "--sheets", nargs="+", default=DEFAULT_SHEETS,
                        help="Sheets to process")
    parser.add_argument("--local-file", type=Path,
                        help="Use local Excel file instead of downloading")

    args = parser.parse_args()

    print("="*70)
    print(" LORIS → TABULAR LEARNER PREPROCESSOR")
    print("="*70)
    print(f"Output directory : {args.output_dir.resolve()}")
    print(f"Sheets to process : {args.sheets}")
    print(f"Target column     : Response (included)")
    print("-"*70)

    # Load data
    if args.local_file:
        if not args.local_file.exists():
            raise FileNotFoundError(f"File not found: {args.local_file}")
        print(f"Loading local file: {args.local_file}")
        excel_bytes = BytesIO(args.local_file.read_bytes())
    else:
        excel_bytes = download_excel(args.url)

    # Process
    sheets_data = load_excel_sheets(excel_bytes, args.sheets)

    print(f"\nStarting preprocessing...\n")
    for sheet_name, df in sheets_data.items():
        process_sheet(df, sheet_name, args.output_dir)

    print("="*70)
    print("ALL DONE!")
    print(f"Generated {len(sheets_data)} clean TSV files")
    print(f"Location: {args.output_dir.resolve()}")
    print("You can now run:")
    print(f"   tabular-learner train --data-path {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
