# LORIS Dataset Preprocessing for Tabular Learner

`preprocessing_data.py`

This script downloads the public **LORIS immunotherapy cohort**
(`AllData.xlsx`) and converts the three main sheets — `Chowell_train`,
`Chowell_test`, and `MSK1` — into clean, supervised, ready-to-train TSV files
that can be used directly with **Tabular Learner**.

## What the script produces

- All required features (`TMB`, `Age`, `NLR`, `Albumin`,
`Systemic_therapy_history`, 16 `CancerType` one-hot columns)
- Target column **`Response`** (included for supervised training)
- Outliers clipped: `TMB ≤ 50`, `Age ≤ 85`, `NLR ≤ 25`
- Simple filenames:
  - `Chowell_train.tsv`
  - `Chowell_test.tsv`
  - `MSK1.tsv`

## Requirements

`pip install pandas openpyxl requests`

## Quick start (one command)

From inside the `tabular_learner/` directory:
`python preprocessing_data.py --output-dir loris_datasets`

### Result

texttabular_learner/
└── loris_datasets/
    ├── Chowell_train.tsv
    ├── Chowell_test.tsv
    └── MSK1.tsv

### All command-line options

Bashpython preprocessing_data.py \

### Where to save the TSV files (default: loris_processed)

    --output-dir PATH/TO/FOLDER

### Space-separated list of sheets (default: all three)

    --sheets Chowell_train Chowell_test MSK1

### Override the default Excel source

    --url https://custom-url/AllData.xlsx

### Use a local Excel file instead of downloading

    --local-file ./AllData.xlsx

### Example usages

    # Default (recommended)
    python preprocessing_data.py --output-dir loris_datasets

    # Put files directly into your dataset repository
    python preprocessing_data.py --output-dir ../datasets-repo/tabular_learner/loris

    # Process only train & test splits
    python preprocessing_data.py --sheets Chowell_train Chowell_test
    --output-dir loris_datasets

    # Use a locally saved Excel file
    python preprocessing_data.py --local-file ./raw/AllData.xlsx
    --output-dir loris_datasets
