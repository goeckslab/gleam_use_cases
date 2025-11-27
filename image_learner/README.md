# HAM10000 Preprocessing for GLEAM-Image Learner

## Overview

This repository contains a preprocessing script that prepares the **HAM10000** dermatoscopic dataset for use with the **GLEAM-Image Learner** tool in Galaxy.

The preprocessing step creates a balanced, leakage-aware metadata CSV file that can be used together with the original HAM10000 image archive when configuring the Image Learner.

## Dataset and Citation

The HAM10000 dataset was created and made publicly available by:

> Peter Tschandl, Cliff Rosendahl, and Harald Kittler

If you use this data, you **must** cite the original publication:

> Tschandl, P., Rosendahl, C. & Kittler, H.
> The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
> *Sci Data* 5, 180161 (2018).
> https://doi.org/10.1038/sdata.2018.161

This preprocessing logic follows the sampling and leakage considerations described in:

> Shetty, B., Fernandes, R., Rodrigues, A.P. et al.
> Skin lesion classification of dermoscopic images using machine learning and convolutional neural network.
> *Sci Rep* 12, 18134 (2022).  
> https://doi.org/10.1038/s41598-022-22644-9

## Required Files

### HAM10000 Metadata (CSV)
The script automatically downloads:
- https://zenodo.org/records/17702692/files/HAM10000_metadata.csv?download=1

### HAM10000 Images (ZIP)
Download the full image archive:
- https://zenodo.org/records/17702692/files/HAM10000_all_images.zip?download=1

This ZIP can be uploaded directly to a Galaxy history and used by the GLEAM-Image Learner tool as the image dataset.

## Requirements

Install the required dependencies:

```bash
pip install pandas numpy requests
```

## Quick Start

### Run the Script

```bash
python preprocessing_data.py
```

### Expected Output

```
HAM10000_dataset/
├── HAM10000_metadata.csv (downloaded raw metadata)
└── HAM10000_metadata_preprocessed.csv (balanced, leakage-aware metadata)
```

You can then upload `HAM10000_metadata_preprocessed.csv` to Galaxy and pair it with the `HAM10000_all_images.zip` archive for use with GLEAM-Image Learner.

## What the Preprocessing Script Does

### 1. Folder Setup

The script ensures a folder named `HAM10000_dataset` exists in the current working directory and stores:
- `HAM10000_metadata.csv` (downloaded raw metadata)
- `HAM10000_metadata_preprocessed.csv` (preprocessed output)

### 2. Download Metadata

The script downloads the original metadata file if it does not already exist:

- **URL**: https://zenodo.org/records/17702692/files/HAM10000_metadata.csv?download=1
- **Saved as**: `HAM10000_dataset/HAM10000_metadata.csv`

The raw CSV contains the following columns:
- `lesion_id`
- `image_id`
- `dx` (diagnosis)
- `dx_type`
- `age`
- `sex`
- `localization`

Only `lesion_id`, `image_id`, and `dx` are required for preprocessing.

### 3. Class-Balanced Sampling

The script creates a **balanced subset** by sampling up to **200 samples per class** based on the `dx` column:

- The `dx` column is treated as the **class label** (diagnosis)
- For each unique class, the script:
  1. Filters metadata to that class
  2. Ensures at most one row per `lesion_id` (see leakage control below)
  3. Randomly samples up to **200 lesions** (or fewer if unavailable)
- Sampling uses a fixed random seed for reproducibility

This follows the preprocessing strategy described by Shetty et al. (2022), where a fixed number of samples per class balances the dataset.

### 4. Data Leakage Control

To prevent **data leakage**, the script enforces **lesion-level uniqueness**:

#### Within Each Class
- Before sampling, metadata for each class is shuffled and deduplicated on `lesion_id`
- This ensures **only one image per lesion** is considered
- If a lesion appears multiple times (e.g., multiple dermatoscopic images), only one is kept

#### Across the Final Dataset
- After concatenating all class samples, the script shuffles again and removes duplicate `lesion_id` rows globally
- This is a safety measure ensuring a lesion belongs to only one diagnosis class

This lesion-aware sampling prevents the same lesion from appearing in both training and testing splits in downstream experiments, which would artificially inflate performance.

### 5. Output Columns

The script constructs a minimal CSV tailored for GLEAM-Image Learner:

- **`label`**: Copied from the `dx` column (diagnosis)
- **`image_path`**: Copied from the `image_id` column (e.g., `ISIC_0027419` → `ISIC_0027419.jpg`)
- **`lesion_id`**: Original lesion identifier

The final file contains only three columns:

```csv
lesion_id,image_path,label
HAM_0000118,ISIC_0027419,bkl
HAM_0002730,ISIC_0025661,bkl
```

### Output File

- **Location**: `HAM10000_dataset/HAM10000_metadata_preprocessed.csv`
