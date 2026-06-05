#!/usr/bin/env python3
"""
Convert a HANCOCK CD3/CD8 image ZIP from PNG to JPEG and keep metadata tables aligned.

Main features:
- Converts image files in a ZIP archive to JPEG (lossy), keeping original dimensions.
- Updates train/test (or other) CSV tables so image path columns reference .jpg files.
- Verifies updated table paths still match images in the converted ZIP.
- Optionally keeps only rows with fully matched image paths.
"""

from __future__ import annotations

import argparse
import io
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from PIL import Image


IMAGE_PATH_PATTERN = r"(?i)\.(png|jpg|jpeg|tif|tiff|bmp|webp)$"
DEFAULT_SOURCE_EXTS = "png"
DEFAULT_QUALITY = 67
DEFAULT_TARGET_GB = 3.0


@dataclass
class JpegSettings:
    quality: int
    optimize: bool
    progressive: bool
    subsampling: int


@dataclass
class ConversionResult:
    converted_images: int
    copied_members: int
    output_names: set[str]
    old_to_new: Dict[str, str]


@dataclass
class MetadataReport:
    source_csv: Path
    output_csv: Path
    image_columns: List[str]
    rows_in: int
    rows_out: int
    rows_all_present: int
    rows_all_matched: int
    rows_missing_any_image_path: int
    rows_unmatched_any_image_path: int


def parse_extensions(text: str) -> set[str]:
    exts = set()
    for part in text.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if not p.startswith("."):
            p = f".{p}"
        exts.add(p)
    if not exts:
        raise ValueError("No valid source extensions provided.")
    return exts


def replace_with_jpg(path_text: str) -> str:
    normalized = path_text.replace("\\", "/")
    if re.search(IMAGE_PATH_PATTERN, normalized):
        return re.sub(IMAGE_PATH_PATTERN, ".jpg", normalized)
    return f"{normalized}.jpg"


def path_basename(path_text: str) -> str:
    return PurePosixPath(path_text.replace("\\", "/")).name


def read_member_lists(src_zip: Path, source_exts: set[str]) -> Tuple[List[str], List[str]]:
    convert_members: List[str] = []
    passthrough_members: List[str] = []
    with zipfile.ZipFile(src_zip, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext in source_exts:
                convert_members.append(info.filename)
            else:
                passthrough_members.append(info.filename)
    return convert_members, passthrough_members


def build_old_to_new_map(convert_members: Sequence[str]) -> Dict[str, str]:
    old_to_new: Dict[str, str] = {}
    new_names: set[str] = set()
    for old_name in convert_members:
        new_name = replace_with_jpg(old_name)
        if new_name in new_names and old_to_new.get(old_name) != new_name:
            raise ValueError(f"Name collision after extension rewrite: {new_name}")
        old_to_new[old_name] = new_name
        new_names.add(new_name)
    return old_to_new


def make_unique_basename_map(old_to_new: Dict[str, str]) -> Dict[str, str]:
    base_map: Dict[str, str] = {}
    ambiguous: set[str] = set()

    for old_name, new_name in old_to_new.items():
        old_base = path_basename(old_name)
        new_base = path_basename(new_name)
        if old_base in base_map and base_map[old_base] != new_base:
            ambiguous.add(old_base)
        else:
            base_map[old_base] = new_base

    for base in ambiguous:
        base_map.pop(base, None)
    return base_map


def image_to_jpeg_bytes(raw: bytes, settings: JpegSettings) -> bytes:
    with Image.open(io.BytesIO(raw)) as im:
        im.load()

        # Flatten transparency on white background before JPEG encoding.
        if im.mode in {"RGBA", "LA"} or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.split()[-1])
            out_im = rgb
        else:
            out_im = im.convert("RGB")

        out = io.BytesIO()
        out_im.save(
            out,
            format="JPEG",
            quality=settings.quality,
            optimize=settings.optimize,
            progressive=settings.progressive,
            subsampling=settings.subsampling,
        )
        return out.getvalue()


def estimate_quality_from_sample(
    src_zip: Path,
    convert_members: Sequence[str],
    passthrough_members: Sequence[str],
    target_size_bytes: int,
    min_q: int,
    max_q: int,
    sample_size: int,
    random_seed: int,
    optimize: bool,
    progressive: bool,
    subsampling: int,
) -> Tuple[int, int]:
    if min_q > max_q:
        raise ValueError("--auto-quality-min must be <= --auto-quality-max")
    if not convert_members:
        raise ValueError("No convertible image members found in source ZIP.")

    rng = random.Random(random_seed)
    sample_members = list(convert_members)
    rng.shuffle(sample_members)
    sample_members = sample_members[: min(sample_size, len(sample_members))]

    with zipfile.ZipFile(src_zip, "r") as zf:
        sample_raws = [zf.read(name) for name in sample_members]
        passthrough_bytes = sum(zf.getinfo(name).file_size for name in passthrough_members)

    best_quality = min_q
    best_estimate = 0
    best_delta = float("inf")

    for quality in range(min_q, max_q + 1):
        settings = JpegSettings(
            quality=quality,
            optimize=optimize,
            progressive=progressive,
            subsampling=subsampling,
        )
        total_sample_jpeg_bytes = 0
        for raw in sample_raws:
            total_sample_jpeg_bytes += len(image_to_jpeg_bytes(raw, settings))

        avg_image_bytes = total_sample_jpeg_bytes / max(1, len(sample_raws))
        estimated_total = int(avg_image_bytes * len(convert_members) + passthrough_bytes)
        delta = abs(estimated_total - target_size_bytes)
        if delta < best_delta:
            best_delta = delta
            best_quality = quality
            best_estimate = estimated_total

    return best_quality, best_estimate


def convert_archive(
    src_zip: Path,
    dst_zip: Path,
    convert_members: Sequence[str],
    old_to_new: Dict[str, str],
    settings: JpegSettings,
    keep_non_images: bool,
    zip_compression: str,
) -> ConversionResult:
    compression = (
        zipfile.ZIP_STORED if zip_compression == "stored" else zipfile.ZIP_DEFLATED
    )
    output_names: set[str] = set()
    converted_images = 0
    copied_members = 0
    convert_set = set(convert_members)

    dst_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(src_zip, "r") as src, zipfile.ZipFile(dst_zip, "w") as dst:
        total_to_convert = len(convert_members)
        done = 0

        for info in src.infolist():
            if info.is_dir():
                continue
            name = info.filename

            if name in convert_set:
                jpeg_bytes = image_to_jpeg_bytes(src.read(name), settings)
                new_name = old_to_new[name]
                if new_name in output_names:
                    raise ValueError(f"Duplicate output member name detected: {new_name}")
                dst.writestr(new_name, jpeg_bytes, compress_type=compression)
                output_names.add(new_name)
                converted_images += 1
                done += 1
                if done % 250 == 0 or done == total_to_convert:
                    print(f"Converted {done}/{total_to_convert} images...")
            else:
                if keep_non_images:
                    if name in output_names:
                        raise ValueError(f"Duplicate passthrough member name detected: {name}")
                    dst.writestr(name, src.read(name), compress_type=compression)
                    output_names.add(name)
                    copied_members += 1

    return ConversionResult(
        converted_images=converted_images,
        copied_members=copied_members,
        output_names=output_names,
        old_to_new=old_to_new,
    )


def infer_image_columns(df: pd.DataFrame) -> List[str]:
    preferred = [c for c in df.columns if "image" in c.lower() and "path" in c.lower()]
    if preferred:
        return preferred

    guessed: List[str] = []
    for col in df.columns:
        s = df[col].astype(str)
        if s.str.contains(IMAGE_PATH_PATTERN, regex=True, na=False).any():
            guessed.append(col)
    return guessed


def rewrite_table_path(
    value: str,
    old_to_new: Dict[str, str],
    basename_map: Dict[str, str],
) -> str:
    text = str(value).strip()
    if not text:
        return text
    normalized = text.replace("\\", "/")

    if normalized in old_to_new:
        return old_to_new[normalized]

    base = path_basename(normalized)
    if base in basename_map:
        parent = str(PurePosixPath(normalized).parent)
        new_base = basename_map[base]
        if parent == ".":
            return new_base
        return f"{parent}/{new_base}"

    return replace_with_jpg(normalized)


def path_exists_in_output(path_text: str, output_names: set[str], output_bases: set[str]) -> bool:
    normalized = path_text.replace("\\", "/")
    if normalized in output_names:
        return True
    return path_basename(normalized) in output_bases


def build_output_csv_paths(
    input_csvs: Sequence[Path],
    explicit_output_csvs: Sequence[Path] | None,
    output_dir: Path | None,
    suffix: str,
) -> List[Path]:
    if explicit_output_csvs is not None and len(explicit_output_csvs) > 0:
        if len(explicit_output_csvs) != len(input_csvs):
            raise ValueError(
                "--metadata-output-csvs must have the same number of entries as --metadata-csvs"
            )
        return list(explicit_output_csvs)

    out_paths = []
    for csv_path in input_csvs:
        base_dir = output_dir if output_dir is not None else csv_path.parent
        out_paths.append(base_dir / f"{csv_path.stem}{suffix}{csv_path.suffix}")
    return out_paths


def update_metadata_table(
    src_csv: Path,
    dst_csv: Path,
    image_columns: Sequence[str] | None,
    conversion: ConversionResult,
    keep_all_rows: bool,
) -> MetadataReport:
    df = pd.read_csv(src_csv, dtype=str, keep_default_na=False)
    cols = list(image_columns) if image_columns else infer_image_columns(df)
    if not cols:
        raise ValueError(f"Could not infer image path columns in {src_csv}")

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing requested image columns in {src_csv}: {missing_cols}")

    basename_map = make_unique_basename_map(conversion.old_to_new)
    output_bases = {path_basename(n) for n in conversion.output_names}

    all_present = pd.Series(True, index=df.index)
    all_matched = pd.Series(True, index=df.index)

    for col in cols:
        rewritten = df[col].map(
            lambda v: rewrite_table_path(v, conversion.old_to_new, basename_map)
        )
        df[col] = rewritten

        present_col = rewritten.astype(str).str.strip() != ""
        matched_col = rewritten.map(
            lambda v: path_exists_in_output(v, conversion.output_names, output_bases)
            if str(v).strip()
            else False
        )
        all_present &= present_col
        all_matched &= matched_col

    rows_in = len(df)
    rows_all_present = int(all_present.sum())
    rows_all_matched = int((all_present & all_matched).sum())
    rows_missing_any = int((~all_present).sum())
    rows_unmatched_any = int((all_present & ~all_matched).sum())

    if keep_all_rows:
        df_out = df
    else:
        df_out = df[all_present & all_matched].copy()

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(dst_csv, index=False)

    return MetadataReport(
        source_csv=src_csv,
        output_csv=dst_csv,
        image_columns=list(cols),
        rows_in=rows_in,
        rows_out=len(df_out),
        rows_all_present=rows_all_present,
        rows_all_matched=rows_all_matched,
        rows_missing_any_image_path=rows_missing_any,
        rows_unmatched_any_image_path=rows_unmatched_any,
    )


def bytes_to_gb(size_bytes: int) -> float:
    return size_bytes / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HANCOCK-style PNG image ZIP to JPEG and update metadata CSV path columns."
        )
    )
    parser.add_argument(
        "--input-zip",
        type=Path,
        required=True,
        help="Source ZIP containing image members (typically PNG files).",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=None,
        help=(
            "Destination ZIP path. If omitted, derived as '<input_stem>_jpeg_q<quality>.zip'."
        ),
    )
    parser.add_argument(
        "--source-exts",
        type=str,
        default=DEFAULT_SOURCE_EXTS,
        help="Comma-separated source extensions to convert (default: png).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_QUALITY,
        help="JPEG quality (1-95). Default: 67.",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_false",
        dest="optimize",
        help="Disable JPEG optimize flag (enabled by default).",
    )
    parser.set_defaults(optimize=True)
    parser.add_argument(
        "--no-progressive",
        action="store_false",
        dest="progressive",
        help="Disable progressive JPEG encoding (enabled by default).",
    )
    parser.set_defaults(progressive=True)
    parser.add_argument(
        "--subsampling",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="JPEG chroma subsampling: 0=4:4:4, 1=4:2:2, 2=4:2:0 (default).",
    )
    parser.add_argument(
        "--zip-compression",
        choices=["stored", "deflated"],
        default="stored",
        help="Compression method for output ZIP entries (default: stored).",
    )
    parser.add_argument(
        "--drop-non-image-members",
        action="store_true",
        help="If set, do not copy non-converted members into output ZIP.",
    )
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=DEFAULT_TARGET_GB,
        help="Target output ZIP size in GB for reporting and optional auto-quality.",
    )
    parser.add_argument(
        "--auto-quality",
        action="store_true",
        help=(
            "Select JPEG quality from sample conversion to best match --target-size-gb "
            "before full conversion."
        ),
    )
    parser.add_argument(
        "--auto-quality-min",
        type=int,
        default=55,
        help="Min quality for auto search (inclusive).",
    )
    parser.add_argument(
        "--auto-quality-max",
        type=int,
        default=90,
        help="Max quality for auto search (inclusive).",
    )
    parser.add_argument(
        "--auto-quality-sample-size",
        type=int,
        default=120,
        help="Number of images used for auto-quality estimation.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for auto-quality sampling.",
    )
    parser.add_argument(
        "--metadata-csvs",
        type=Path,
        nargs="*",
        default=[],
        help="Metadata CSV files to rewrite (e.g., train/test split tables).",
    )
    parser.add_argument(
        "--metadata-output-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit output CSV paths (must match --metadata-csvs count).",
    )
    parser.add_argument(
        "--metadata-output-dir",
        type=Path,
        default=None,
        help="Directory for rewritten metadata CSVs when explicit outputs are not provided.",
    )
    parser.add_argument(
        "--metadata-suffix",
        type=str,
        default="_3GB_jpeg",
        help=(
            "Suffix appended to auto-derived metadata CSV names. "
            "Supports '{quality}' placeholder."
        ),
    )
    parser.add_argument(
        "--image-columns",
        nargs="*",
        default=None,
        help=(
            "Image path columns to rewrite in metadata tables. "
            "If omitted, columns are inferred."
        ),
    )
    parser.add_argument(
        "--keep-all-rows",
        action="store_true",
        help=(
            "Keep all rows in rewritten metadata. "
            "Default behavior keeps only rows with all image columns present and matched."
        ),
    )

    args = parser.parse_args()

    if not args.input_zip.exists():
        raise FileNotFoundError(f"Input ZIP not found: {args.input_zip}")

    if not (1 <= args.quality <= 95):
        raise ValueError("--quality must be between 1 and 95")

    source_exts = parse_extensions(args.source_exts)
    convert_members, passthrough_members = read_member_lists(args.input_zip, source_exts)
    if not convert_members:
        raise ValueError(
            f"No members found with extensions {sorted(source_exts)} in {args.input_zip}"
        )

    chosen_quality = args.quality
    target_size_bytes = int(args.target_size_gb * (1024 ** 3))

    if args.auto_quality:
        chosen_quality, estimated_bytes = estimate_quality_from_sample(
            src_zip=args.input_zip,
            convert_members=convert_members,
            passthrough_members=passthrough_members,
            target_size_bytes=target_size_bytes,
            min_q=args.auto_quality_min,
            max_q=args.auto_quality_max,
            sample_size=args.auto_quality_sample_size,
            random_seed=args.random_seed,
            optimize=args.optimize,
            progressive=args.progressive,
            subsampling=args.subsampling,
        )
        print(
            f"Auto-selected JPEG quality={chosen_quality} "
            f"(sample-estimated size: {bytes_to_gb(estimated_bytes):.2f} GB)"
        )

    output_zip = args.output_zip
    if output_zip is None:
        output_zip = args.input_zip.with_name(
            f"{args.input_zip.stem}_jpeg_q{chosen_quality}.zip"
        )

    old_to_new = build_old_to_new_map(convert_members)
    keep_non_images = not args.drop_non_image_members

    settings = JpegSettings(
        quality=chosen_quality,
        optimize=args.optimize,
        progressive=args.progressive,
        subsampling=args.subsampling,
    )

    print(f"Converting ZIP: {args.input_zip}")
    print(f"  Source image extensions: {sorted(source_exts)}")
    print(f"  Images to convert: {len(convert_members)}")
    print(f"  JPEG quality: {chosen_quality}")
    print(f"  JPEG optimize: {args.optimize}")
    print(f"  JPEG progressive: {args.progressive}")
    print(f"  JPEG subsampling: {args.subsampling}")
    print(f"  Output ZIP compression: {args.zip_compression}")

    conversion = convert_archive(
        src_zip=args.input_zip,
        dst_zip=output_zip,
        convert_members=convert_members,
        old_to_new=old_to_new,
        settings=settings,
        keep_non_images=keep_non_images,
        zip_compression=args.zip_compression,
    )

    out_size = output_zip.stat().st_size
    delta = out_size - target_size_bytes
    print("\nZIP conversion summary:")
    print(f"  Converted images: {conversion.converted_images}")
    print(f"  Copied non-image members: {conversion.copied_members}")
    print(f"  Output ZIP: {output_zip}")
    print(f"  Output size: {bytes_to_gb(out_size):.2f} GB")
    print(
        f"  Delta vs target {args.target_size_gb:.2f} GB: "
        f"{bytes_to_gb(abs(delta)):.2f} GB ({'above' if delta > 0 else 'below'})"
    )

    reports: List[MetadataReport] = []
    if args.metadata_csvs:
        for csv_path in args.metadata_csvs:
            if not csv_path.exists():
                raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

        suffix = args.metadata_suffix.format(quality=chosen_quality)
        out_csv_paths = build_output_csv_paths(
            input_csvs=args.metadata_csvs,
            explicit_output_csvs=args.metadata_output_csvs,
            output_dir=args.metadata_output_dir,
            suffix=suffix,
        )

        for src_csv, out_csv in zip(args.metadata_csvs, out_csv_paths):
            report = update_metadata_table(
                src_csv=src_csv,
                dst_csv=out_csv,
                image_columns=args.image_columns,
                conversion=conversion,
                keep_all_rows=args.keep_all_rows,
            )
            reports.append(report)

        print("\nMetadata rewrite summary:")
        for rep in reports:
            print(f"  Source: {rep.source_csv}")
            print(f"  Output: {rep.output_csv}")
            print(f"  Image columns: {rep.image_columns}")
            print(f"  Rows in/out: {rep.rows_in}/{rep.rows_out}")
            print(f"  Rows with all image paths present: {rep.rows_all_present}")
            print(f"  Rows with all image paths matched in ZIP: {rep.rows_all_matched}")
            print(
                f"  Rows missing any image path: {rep.rows_missing_any_image_path}; "
                f"rows with unmatched image path: {rep.rows_unmatched_any_image_path}"
            )


if __name__ == "__main__":
    main()
