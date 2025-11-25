# LORIS Dataset Preprocessing for Tabular Learner

## Overview

The `preprocessing_data.py` script prepares the public **LORIS immunotherapy cohort** for use with the **Tabular Learner** tool. It enables reproduction and extension of the **LLR6** pan-cancer model described in:

> Chang, T.G., Cao, Y., Sfreddo, H.J. et al.  
> LORIS robustly predicts patient outcomes with immune checkpoint blockade therapy using common clinical, pathologic and genomic features.  
> *Nat Cancer* 5, 1158–1175 (2024).  
> https://doi.org/10.1038/s43018-024-00772-7

The script downloads the original `AllData.xlsx` file and converts three main sheets — `Chowell_train`, `Chowell_test`, and `MSK1` — into clean, supervised TSV files matching the LLR6 feature space.

## What the Script Produces

For each selected sheet, the script:

- Downloads and parses the **LORIS** Excel file (`AllData.xlsx`)
- Selects and formats **LLR6** features:
  - **Clinical and pathologic variables**: `Age`, `Systemic_therapy_history`, `Albumin`, etc.
  - **Genomic/immune variables**: `TMB`, `NLR`, etc.
  - **Cancer type**: One-hot encoded as 16 `CancerType_*` columns
- Builds a **supervised table** with:
  - All required predictors for the LLR6 pan-cancer model
  - Target column: **`Response`** (treatment outcome)
- Applies outlier clipping:
  - `TMB ≤ 50`
  - `Age ≤ 85`
  - `NLR ≤ 25`
- Outputs one TSV file per sheet:
  - `Chowell_train.tsv`
  - `Chowell_test.tsv`
  - `MSK1.tsv`

These TSV files can be uploaded directly to Galaxy and used with **Tabular Learner** for LLR6-compatible model configurations.

## Requirements

Install the required dependencies:

```bash
pip install pandas openpyxl requests
```

## Quick Start

From the `tabular_learner/` directory:

```bash
python preprocessing_data.py --output-dir loris_datasets
```

### Expected Output

```
tabular_learner/
└── loris_datasets/
    ├── Chowell_train.tsv
    ├── Chowell_test.tsv
    └── MSK1.tsv
```

Each TSV is ready for training with the LLR6 feature set and `Response` as the target label.

## Command-Line Options

### Basic Usage

```bash
python preprocessing_data.py \
    [--output-dir PATH/TO/FOLDER] \
    [--sheets SHEET1 SHEET2 ...] \
    [--url CUSTOM_EXCEL_URL] \
    [--local-file PATH/TO/AllData.xlsx]
```

### Options

#### `--output-dir PATH/TO/FOLDER`

Where to save the TSV files (default: `loris_processed`)

The script creates the folder if it doesn't exist and writes one TSV file per selected sheet.

#### `--sheets SHEET1 SHEET2 ...`

Space-separated list of sheets to process (default: all three)

Use this to restrict preprocessing to specific LORIS cohorts:

```bash
python preprocessing_data.py --sheets Chowell_train Chowell_test
```

#### `--url CUSTOM_EXCEL_URL`

Override the default Excel source (default: public LORIS URL)

Use this if you maintain a mirrored or modified Excel file:

```bash
python preprocessing_data.py --url https://custom-url/AllData.xlsx
```

#### `--local-file PATH/TO/AllData.xlsx`

Use a local Excel file instead of downloading

Skips the download step and reads from the provided path:

```bash
python preprocessing_data.py --local-file ./AllData.xlsx
```

## Examples

### Default (Recommended)
Process all three sheets into a local folder:

```bash
python preprocessing_data.py --output-dir loris_datasets
```

### Custom Output Location
Put files into your dataset repository:

```bash
python preprocessing_data.py \
  --output-dir ../datasets-repo/tabular_learner/loris
```

### Subset of Sheets
Process only train & test splits (exclude MSK1):

```bash
python preprocessing_data.py \
  --sheets Chowell_train Chowell_test \
  --output-dir loris_datasets
```

### Local Excel File
Use a locally saved Excel file:

```bash
python preprocessing_data.py \
  --local-file ./raw/AllData.xlsx \
  --output-dir loris_datasets
```

## Next Steps

The TSV outputs can be uploaded to Galaxy and passed to **Tabular Learner** to train or evaluate models consistent with the LLR6 pan-cancer configuration described in Chang et al. (2024).
