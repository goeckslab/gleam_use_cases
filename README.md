# GLEAM Use Cases

## Purpose

This repository contains **data preprocessing scripts** that generate datasets used to demonstrate and validate the **GLEAM tools** (Galaxy Learning Enabled Analysis Modules) for machine learning:

- **Image Learner** — For image classification tasks
- **Multimodal Learner** — For multimodal learning combining images, tabular data, and structured features
- **Tabular Learner** — For tabular/structured data analysis

Each subdirectory contains preprocessing scripts that download, clean, balance, and prepare public datasets for use with the corresponding GLEAM tool.

---

## Repository Structure

### [`image_learner/`](./image_learner/)

**Dataset:** HAM10000 (Skin Lesion Classification)

**Script:** `preprocessing_data.py`

Prepares a class-balanced, lesion-aware subset of the HAM10000 dermatoscopic dataset:
- Downloads 10,000 dermatoscopic images
- Creates class-balanced samples (100 images per diagnosis class)
- Resizes images to 96×96 with augmentations (original + horizontal flip)
- Outputs: `selected_images_96.zip` and `selected_image_metadata.csv`

**Publication:**
> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018).

---

### [`multimodal_learner/`](./multimodal_learner/)

**Dataset:** HANCOCK (Head and Neck Cancer)

**Script:** `preprocessing_data.py`

Prepares tabular supervision files from the HANCOCK clinical, pathological, and imaging dataset:
- Merges structured tables (clinical, pathological, blood, TMA cell density)
- Optionally includes ICD codes and image-derived features
- Applies split labels (training/test) for multiple cohorts
- Supports two modes:
  - **Multimodal-friendly:** Drops structured ICD/CD3–CD8 columns for separate modality handling
  - **Paper-like:** Tabular-only features for classical structured learning
- Outputs: Patient-level CSV files with splits and binary targets (recurrence, survival status)

**Publication:**
> Dörrich, M., Balk, M., Heusinger, T. et al. A multimodal dataset for precision oncology in head and neck cancer. *Nat Commun* 16, 7163 (2025).

---

### [`tabular_learner/`](./tabular_learner/)

**Dataset:** LORIS (Immunotherapy Cohort)

**Script:** `preprocessing_data.py`

Prepares the LORIS pan-cancer immunotherapy dataset for tabular learning:
- Downloads the public LORIS Excel file (`AllData.xlsx`)
- Extracts clinical, pathological, and genomic features matching the **LLR6** pan-cancer model
- Includes 16 cancer type one-hot encodings
- Applies data quality filters (outlier clipping for TMB, Age, NLR)
- Outputs: Clean TSV files (Chowell_train, Chowell_test, MSK1) with `Response` target

**Publication:**
> Chang, T.G., Cao, Y., Sfreddo, H.J. et al. LORIS robustly predicts patient outcomes with immune checkpoint blockade therapy using common clinical, pathologic and genomic features. *Nat Cancer* 5, 1158–1175 (2024).

---

## Quick Start

Each subdirectory is self-contained with its own `preprocessing_data.py` script and detailed `README.md`.

### Image Learner
```bash
cd image_learner
pip install pandas numpy pillow
python preprocessing_data.py
```

### Multimodal Learner
```bash
cd multimodal_learner
pip install pandas
python preprocessing_data.py --output_directory hancock_datasets
```

### Tabular Learner
```bash
cd tabular_learner
pip install pandas openpyxl requests
python preprocessing_data.py --output-dir loris_datasets
```

---

## Key Features

- ✅ **Fully self-contained scripts** — No manual setup required; data is downloaded and processed automatically
- ✅ **Reproducible** — Fixed random seeds and documented preprocessing pipelines
- ✅ **Galaxy-ready** — Output formats designed for direct import into GLEAM Galaxy tools
- ✅ **Publicly available data** — All datasets are published and citable
- ✅ **Minimal dependencies** — Uses only standard Python libraries (pandas, numpy, pillow, openpyxl, requests)

---

## Use Cases

These preprocessed datasets enable users to:

1. **Learn by example** — Understand how to use GLEAM tools with real, publication-backed data
2. **Reproduce published results** — Validate GLEAM tool implementations against known benchmarks
3. **Extend research** — Build upon standardized datasets for new machine learning experiments
4. **Educational demonstrations** — Teach concepts of multimodal learning, image classification, and tabular analysis

---

## License

See the [LICENSE](./LICENSE) file for repository-level terms. Individual datasets retain their original licenses and publication terms.
