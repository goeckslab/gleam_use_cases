```markdown
# HAM10000 Preprocessing for GLEAM-Image Learner

## Overview

This preprocessing script prepares a class-balanced, lesion-aware subset of the **HAM10000** dermatoscopic dataset for use with GLEAM-Image Learner (Galaxy).

The script is fully self-contained and requires **no input arguments**:
- Automatically downloads the HAM10000 images and metadata
- Produces class-balanced samples (no split assignment)
- Generates optimized 96×96 resized images (original + flipped)

**Output files:**
- `selected_images_96.zip` — ZIP containing 96×96 JPEGs for each selected image (original + horizontally flipped)
- `selected_image_metadata.csv` — Minimal metadata file with `image_path` and `label` only

---

## Quick Start

```bash
pip install pandas numpy pillow
python preprocessing_data.py
```

After running, you will find:
```
downloads/
├── HAM10000_all_images.zip
└── HAM10000_metadata.csv

processed_data_no_leak_70_10_20/
├── selected_images_96.zip
└── selected_image_metadata.csv
```

Upload `selected_images_96.zip` and `selected_image_metadata.csv` to Galaxy for use with GLEAM-Image Learner.

---

## Requirements

- **Python 3.x**
- **Dependencies:** `pandas`, `numpy`, `pillow`
- Other modules used by the script are from the Python standard library

---

## What the Script Does

### 1. Downloads (if missing)
The script downloads and caches the following files:
- `./downloads/HAM10000_all_images.zip`
- `./downloads/HAM10000_metadata.csv`

If files already exist, they are reused (no re-download).

### 2. Loads metadata and sets labels
- Reads `HAM10000_metadata.csv`
- Renames the `dx` column to `label` (diagnosis values unchanged)
- Requires only: `lesion_id`, `image_id`, `label`

### 3. Class-balanced sampling (100 per class, lesion-aware)
The script builds a balanced subset using `sample_balanced_no_leak()`:
- For each diagnosis class, samples **100 images**
- Prefers one image per unique lesion to avoid data leakage
- If fewer than 100 unique lesions exist, tops up with additional images from the same lesions
- Uses a fixed random seed for reproducibility

### 4. Creates 96×96 ZIP (original + flipped)
For each selected image:
- `{image_id}_orig.jpg` — 96×96 resized original
- `{image_id}_flip.jpg` — 96×96 horizontally flipped

Images are read directly from the source ZIP without extracting the entire archive (memory efficient).

### 5. Writes minimal output metadata
Output CSV contains:
- `image_path` — Filename inside `selected_images_96.zip`
- `label` — Diagnosis category

**Example:**
```csv
image_path,label
ISIC_0027419_orig.jpg,bkl
ISIC_0027419_flip.jpg,bkl
```

---

## Dataset Information

### Citation
If you use this dataset, please cite:

**Tschandl et al. (2018)** — The HAM10000 dataset:
> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018).
> https://doi.org/10.1038/sdata.2018.161

**Shetty et al. (2022)** — Lesion-aware sampling methodology:
> Shetty, B., Fernandes, R., Rodrigues, A.P. et al. Skin lesion classification of dermoscopic images using machine learning and convolutional neural network. *Sci Rep* 12, 18134 (2022).
> https://doi.org/10.1038/s41598-022-22644-9

### Data Sources
Files are automatically downloaded from [Zenodo](https://zenodo.org/records/17702692):
- **Metadata CSV:** `HAM10000_metadata.csv`
- **Images ZIP:** `HAM10000_all_images.zip`

### Metadata Columns
The original metadata CSV contains:
- `lesion_id` — Unique lesion identifier
- `image_id` — Unique image identifier
- `dx` — Diagnosis (renamed to `label`)
- `dx_type` — Type of diagnostic method
- `age` — Patient age
- `sex` — Patient sex
- `localization` — Body site location

---

## Output Files

processed_data_no_leak_70_10_20/selected_images_96.zip
processed_data_no_leak_70_10_20/selected_image_metadata.csv
``` 