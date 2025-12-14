#!/usr/bin/env python3
import os
import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------------------------------------------------------
# Fixed sources (no user inputs required)
# -----------------------------------------------------------------------------
IMAGES_ZIP_URL = "https://zenodo.org/records/17702692/files/HAM10000_all_images.zip"
METADATA_CSV_URL = "https://zenodo.org/records/17702692/files/HAM10000_metadata.csv"

# -----------------------------------------------------------------------------
# Constants / Outputs
# -----------------------------------------------------------------------------
SAMPLES_PER_CLASS = 100
IMG_SIZE_96 = (96, 96)

DOWNLOAD_DIR = Path("./downloads")
OUTPUT_DIR = Path("./processed_data_no_leak_70_10_20")

OUT_ZIP_NAME = "selected_images_96.zip"
OUT_CSV_NAME = "selected_image_metadata.csv"  # <-- renamed

RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Download helpers
# -----------------------------------------------------------------------------
def download_if_needed(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)

    tmp.replace(dst)


# -----------------------------------------------------------------------------
# Load metadata
# -----------------------------------------------------------------------------
def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Ensure dx values are kept as-is, but use column name "label"
    if "dx" in df.columns:
        df = df.rename(columns={"dx": "label"})

    required = {"lesion_id", "image_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in metadata: {sorted(missing)}")

    return df.sort_values(by=["image_id", "label"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Class-balanced sampling (no-leak, lesion-aware with fallback)
# -----------------------------------------------------------------------------
def sample_balanced_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    sampled = []
    rng = np.random.default_rng(RANDOM_SEED)

    for label in sorted(df["label"].unique()):
        c = df[df["label"] == label].copy()
        lids = c["lesion_id"].unique()
        rng.shuffle(lids)

        picks = []
        # 1 image per lesion first
        for lid in lids:
            picks.append(c[c["lesion_id"] == lid].sample(n=1, random_state=RANDOM_SEED))
            if len(picks) >= SAMPLES_PER_CLASS:
                break

        picked = pd.concat(picks, ignore_index=True)

        # top-up if needed (split remains lesion-level, so no leakage across splits)
        if len(picked) < SAMPLES_PER_CLASS:
            pool = c[~c["image_id"].isin(picked["image_id"])]
            if not pool.empty:
                need = SAMPLES_PER_CLASS - len(picked)
                picked = pd.concat(
                    [picked, pool.sample(n=min(need, len(pool)), random_state=RANDOM_SEED)],
                    ignore_index=True,
                )

        sampled.append(picked)

    return pd.concat(sampled, ignore_index=True)


# -----------------------------------------------------------------------------
# Lesion-level split: 70% train / 10% val / 20% test (no leakage)
# -----------------------------------------------------------------------------
def assign_split_70_10_20_by_lesion(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    out_parts = []

    for label in sorted(df["label"].unique()):
        c = df[df["label"] == label].copy()
        lids = np.array(c["lesion_id"].unique(), dtype=object)
        rng.shuffle(lids)

        n = len(lids)
        if n == 0:
            continue

        # Aim for 70/10/20 by lesion count, remainder goes to test
        n_train = int(round(0.70 * n))
        n_val = int(round(0.10 * n))
        n_test = n - n_train - n_val

        # Guardrails for small n
        if n >= 3:
            if n_val == 0:
                n_val = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_test = max(1, n_test - 1)
            if n_test == 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(1, n_val - 1)

        # Final clamp
        n_train = max(1, min(n_train, n))
        n_val = max(0, min(n_val, n - n_train))
        n_test = n - n_train - n_val

        tr_lids = set(lids[:n_train])
        va_lids = set(lids[n_train : n_train + n_val])
        te_lids = set(lids[n_train + n_val :])

        c["split"] = np.where(
            c["lesion_id"].isin(tr_lids),
            "train",
            np.where(c["lesion_id"].isin(va_lids), "val", "test"),
        )
        out_parts.append(c)

    return pd.concat(out_parts, ignore_index=True)


def verify_no_lesion_overlap(df_with_split: pd.DataFrame) -> None:
    def lids(split_name: str) -> set:
        return set(df_with_split.loc[df_with_split["split"] == split_name, "lesion_id"].unique())

    tr, va, te = lids("train"), lids("val"), lids("test")
    overlaps = {
        "TRAIN/VAL": tr & va,
        "TRAIN/TEST": tr & te,
        "VAL/TEST": va & te,
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        for k, v in bad.items():
            print(f"❌ Leakage detected {k}: {len(v)} lesions")
    else:
        print("✅ No lesion_id overlap between train/val/test")


# -----------------------------------------------------------------------------
# Image IO from source zip -> write resized images into output zip
# -----------------------------------------------------------------------------
def build_image_member_map(zf: zipfile.ZipFile) -> dict:
    m = {}
    for name in zf.namelist():
        if name.lower().endswith(".jpg"):
            m[Path(name).stem] = name
    return m


def add_96_images_to_zip(
    df_rows: pd.DataFrame,
    src_zip: zipfile.ZipFile,
    member_map: dict,
    out_zip: zipfile.ZipFile,
) -> list:
    rows_out = []
    written = set()

    for _, row in df_rows.iterrows():
        iid = row["image_id"]
        lbl = row["label"]
        split = row["split"]

        member = member_map.get(iid) or (f"{iid}.jpg" if f"{iid}.jpg" in src_zip.namelist() else None)
        if member is None:
            print(f"Skipping {iid}: not found in source images zip")
            continue

        try:
            with src_zip.open(member) as fp:
                data = fp.read()
            im = Image.open(io.BytesIO(data)).convert("RGB")

            for tag, img in [("orig", im), ("flip", im.transpose(Image.FLIP_LEFT_RIGHT))]:
                fn = f"{iid}_{tag}.jpg"
                if fn in written:
                    continue
                written.add(fn)

                buf = io.BytesIO()
                img.resize(IMG_SIZE_96, Image.Resampling.LANCZOS).save(
                    buf, format="JPEG", quality=95
                )
                out_zip.writestr(fn, buf.getvalue())

                rows_out.append({"image_path": fn, "label": lbl, "split": split})

        except Exception as e:
            print(f"Skipping {iid}: {e}")

    return rows_out


# -----------------------------------------------------------------------------
# Main pipeline (no CLI args)
# -----------------------------------------------------------------------------
def main() -> None:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images_zip_path = DOWNLOAD_DIR / "HAM10000_all_images.zip"
    metadata_csv_path = DOWNLOAD_DIR / "HAM10000_metadata.csv"

    download_if_needed(IMAGES_ZIP_URL, images_zip_path)
    download_if_needed(METADATA_CSV_URL, metadata_csv_path)

    df = load_metadata(metadata_csv_path)

    bal = sample_balanced_no_leak(df)
    bal = assign_split_70_10_20_by_lesion(bal)
    verify_no_lesion_overlap(bal)

    print("\nSplit counts:")
    print(bal["split"].value_counts(dropna=False).to_string())
    print("\nLabel counts by split:")
    print(pd.crosstab(bal["label"], bal["split"]).to_string())

    out_zip_path = OUTPUT_DIR / OUT_ZIP_NAME
    out_csv_path = OUTPUT_DIR / OUT_CSV_NAME

    with zipfile.ZipFile(images_zip_path, "r") as src_zf:
        member_map = build_image_member_map(src_zf)
        with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as out_zf:
            rows_out = add_96_images_to_zip(
                df_rows=bal[["image_id", "label", "split"]],
                src_zip=src_zf,
                member_map=member_map,
                out_zip=out_zf,
            )

    pd.DataFrame(rows_out, columns=["image_path", "label", "split"]).to_csv(out_csv_path, index=False)

    print(f"\n✅ Output written to: {OUTPUT_DIR}")
    print(f"  {OUT_ZIP_NAME}")
    print(f"  {OUT_CSV_NAME} (rows: {len(rows_out)})")


if __name__ == "__main__":
    main()
