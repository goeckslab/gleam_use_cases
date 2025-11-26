# HANCOCK Preprocessing for GLEAM-Multimodal Learner

## Overview

This repository contains a preprocessing script that prepares the **HANCOCK** head and neck cancer dataset for use with the **GLEAM-Multimodal Learner** tool in Galaxy.

The goal is to take the **preprocessed, structured** HANCOCK tables (clinical, pathological, blood, ICD, and image-derived features) and reshape them into a **clean tabular supervision file** that can be paired with **unstructured modalities** (raw pixel images and text) inside Multimodal Learner.

The script builds patient-level CSV files with:

- One row per `patient_id`
- A numeric split label for training vs test
- A binary target (`recurrence` or `survival_status`)
- A curated set of tabular features, with the option to keep or exclude structured ICD / CD3–CD8–derived features depending on the downstream multimodal setup.

## Dataset and Citation

This work is based on the **HANCOCK** dataset:

> Dörrich, M., Balk, M., Heusinger, T. et al.  
> A multimodal dataset for precision oncology in head and neck cancer.  
> *Nat Commun* 16, 7163 (2025).  
> https://doi.org/10.1038/s41467-025-62386-6

If you use this preprocessing and the underlying data, you should cite the original publication above.

The preprocessing script is designed so that the resulting tabular files can be used:

- Either to approximate the **paper-like** structured setting (using only classical clinical / pathological / blood / TMA features), or  
- As the **tabular backbone** in a fully multimodal setup where ICD codes and image content are provided as separate unstructured modalities to GLEAM-Multimodal Learner.

## Required Files

All inputs are expected under a directory called `paper_dataset` (hard-coded relative to the script location):

```text
multimodal_learner/
├── preprocessing_data.py
└── paper_dataset/
    ├── clinical.csv
    ├── pathological.csv
    ├── blood.csv
    ├── tma_cell_density.csv
    ├── icd_codes.csv
    ├── CD3_CD8_raw_images.csv
    ├── targets.csv
    ├── dataset_split_in.json
    ├── dataset_split_out.json
    └── dataset_split_Oropharynx.json
Feature Tables (CSV)
All tables are keyed by patient_id:

clinical.csv
Core demographic and clinical variables.

pathological.csv
Pathology and tumor characteristics.

blood.csv
# HANCOCK Preprocessing for GLEAM-Multimodal Learner

## Overview

`preprocessing_data.py` prepares HANCOCK patient-level CSVs that serve as the tabular supervision input for the GLEAM-Multimodal Learner. The script merges structured tables (clinical, pathological, blood, TMA), optionally includes ICD and CD3/CD8-derived encodings, attaches split and target labels, and writes clean CSVs ready for downstream multimodal experiments.

Key features:
- One row per `patient_id`
- Numeric split column (`dataset`): training/test
- Binary `target` column (e.g., `recurrence` or `survival_status`)
- Two output modes: multimodal-friendly (drops structured ICD / CD3–CD8 columns) and `--paper_like` (tabular-only features)

## Citation

If you use the dataset or preprocessing, cite:

> Dörrich, M., Balk, M., Heusinger, T. et al. A multimodal dataset for precision oncology in head and neck cancer. *Nat Commun* 16, 7163 (2025). https://doi.org/10.1038/s41467-025-62386-6

## Required inputs

Place input files in `paper_dataset/` (relative to `preprocessing_data.py`):

```
multimodal_learner/
├── preprocessing_data.py
└── paper_dataset/
    ├── clinical.csv
    ├── pathological.csv
    ├── blood.csv
    ├── tma_cell_density.csv
    ├── icd_codes.csv
    ├── CD3_CD8_raw_images.csv
    ├── targets.csv
    ├── dataset_split_in.json
    ├── dataset_split_out.json
    └── dataset_split_Oropharynx.json
```

Short descriptions:
- `clinical.csv`, `pathological.csv`, `blood.csv`, `tma_cell_density.csv`: core feature tables keyed by `patient_id`.
- `icd_codes.csv`: structured ICD encodings (optional for multimodal workflow).
- `CD3_CD8_raw_images.csv`: structured CD3/CD8 image-derived features (optional).
- `targets.csv`: outcome and follow-up information.
- `dataset_split_*.json`: split definitions mapping `patient_id` → `"training"`/`"test"` for in-distribution, out-of-distribution and Oropharynx splits.

## Requirements

Python 3.8+ and:

```bash
pip install pandas
```

## Quick start

From the `multimodal_learner/` directory, run:

```bash
python preprocessing_data.py --output_directory hancock_datasets
```

This produces files such as:

```
hancock_datasets/
└── recurrence_in_distribution.csv
```

### Paper-like tabular-only mode

Append `--paper_like` to restrict features to classical tabular variables and suffix output filenames with `_paper_like`:

```bash
python preprocessing_data.py --output_directory hancock_datasets --paper_like
```

### Use survival status as target

```bash
python preprocessing_data.py --output_directory hancock_datasets --target_label survival_status
```

### Produce all target × split combinations

```bash
python preprocessing_data.py --output_directory hancock_datasets --all
```

With `--paper_like`, this will generate corresponding `_paper_like` files.

## Output format and conventions

Each CSV contains:
- `patient_id` (unique identifier)
- `dataset` (numeric split: `0` = training, `2` = test)
- `target` (binary label)
- Remaining feature columns (merged from input tables)

Filenames follow the pattern:

```
<output_directory>/<target_label>_<split_type>[optional_suffix].csv
```

Examples:
- `recurrence_in_distribution.csv`
- `survival_status_out_distribution_paper_like.csv`

## What the script does (high-level)

1. Loads required feature tables from `paper_dataset/` and the requested split JSON.
2. Merges clinical, pathological, blood and TMA features (outer join on `patient_id`).
3. Conditionally reads `icd_codes.csv` and `CD3_CD8_raw_images.csv` when not using `--paper_like`.
4. Merges targets and selected split mapping; maps string splits to numeric codes (`training` → `0`, `test` → `2`).
5. Constructs the binary `target` according to `--target_label` rules (recurrence: 3-year logic; survival_status: living/deceased encoding).
6. Drops original outcome columns (keeps only `target`) and, in multimodal-friendly mode, drops structured ICD / CD3–CD8 columns so those modalities can be supplied separately as unstructured inputs.
7. Writes final CSV(s) with a consistent column order: `patient_id`, `dataset`, `target`, then features.

## Target definition details

- `recurrence` (default): positive if `recurrence == "yes"` and `days_to_recurrence <= 3*365`; negative if `recurrence == "no"` and adequate follow-up or living status; other rows may be discarded.
- `survival_status`: excludes non–tumor-specific deaths (`survival_status_with_cause == "deceased not tumor specific"`), then encodes `living` → `0`, `deceased` → `1`.

## ICD / CD3–CD8 handling

Default (multimodal-friendly): structured ICD and CD3/CD8 features are merged but dropped from the final CSV so the raw ICD text and CD3/CD8 images can be used as separate modalities in Multimodal Learner.

Paper-like mode (`--paper_like`): ICD / CD3–CD8 tables are not read; the output contains only classical tabular covariates.

The script contains a list of ICD-related and CD3/CD8-derived column names that are removed in the multimodal-friendly outputs to avoid leakage.

## Examples

- Default (recurrence, in-distribution):
  ```bash
  python preprocessing_data.py --output_directory hancock_datasets
  ```
- Paper-like recurrence (in-distribution):
  ```bash
  python preprocessing_data.py --output_directory hancock_datasets --paper_like
  ```
- All targets and splits:
  ```bash
  python preprocessing_data.py --output_directory hancock_datasets --all
  ```

## Notes and troubleshooting

- Ensure `paper_dataset/` contains all listed CSVs and JSON split files.
- The script uses outer joins so missing values are represented as `NaN`; consider your downstream imputation/encoding strategy when training models.
- If you want to keep structured ICD or CD3/CD8 features in the final CSV, run without the default drop logic (modify the script accordingly).

## Contact

For questions about the preprocessing or HANCOCK dataset usage, open an issue in this repository or contact the maintainers.
