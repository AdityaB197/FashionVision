"""Data preparation script for FashionVision v2.

Downloads (or accepts a source folder for) the agrigorev/clothing-dataset-full dataset,
inspects real layout, filters non-clothing and under-represented classes, splits into
train/val/test (preferring group-aware split by sender_id), resizes images to (img_size, img_size)
with LANCZOS, and saves them ready for training.

Also supports a --synthetic flag for local offline plumbing verification without downloading 7 GB.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import GroupShuffleSplit, train_test_split

NON_CLOTHING_CATEGORIES = {"not sure", "other", "skip", "notsure", "unknown", "none"}


def download_dataset():
    """Download the dataset via kagglehub lazily."""
    try:
        import kagglehub

        print("\nInitiating Kaggle download via kagglehub...")
        path = kagglehub.dataset_download("agrigorev/clothing-dataset-full")
        print("Path to dataset files:", path)
        return Path(path)
    except ImportError:
        print(
            "\n[Error] kagglehub is not installed in this environment.\n"
            "On Google Colab, install or import kagglehub. On local machines, use --source-dir or --synthetic.\n"
        )
        raise
    except Exception as e:
        print(
            "\n[Kaggle Authentication Required]\n"
            "If downloading asks for Kaggle credentials, run kagglehub.login() or configure\n"
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables (e.g. via Colab secrets).\n"
        )
        raise


def generate_synthetic_raw_dataset(temp_dir):
    """Generate a tiny synthetic raw dataset (3 classes, 20 images each) for plumbing verification."""
    temp_path = Path(temp_dir)
    images_dir = temp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    classes = ["dress", "shirt", "t-shirt"]
    records = []
    sender_ids = [f"user_{i}" for i in range(1, 8)]

    print(f"Generating synthetic raw dataset in '{temp_path}' (3 classes, 20 images each)...")
    img_counter = 1
    for cls in classes:
        for i in range(20):
            img_id = f"synth_{img_counter:04d}"
            img_filename = f"{img_id}.jpg"
            img_path = images_dir / img_filename

            # Create a small RGB image with distinctive random color pattern
            color = (
                int((hash(cls) * 37 + i * 11) % 255),
                int((hash(cls) * 73 + i * 17) % 255),
                int((hash(cls) * 109 + i * 23) % 255),
            )
            img = Image.new("RGB", (64, 64), color=color)
            img.save(img_path, format="JPEG", quality=90)

            sender = sender_ids[(img_counter - 1) % len(sender_ids)]
            records.append({
                "image": img_filename,
                "sender_id": sender,
                "label": cls,
                "kids": False,
            })
            img_counter += 1

    csv_path = temp_path / "images.csv"
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print("Synthetic raw images.csv and images created successfully.")
    return temp_path


def find_dataset_files(source_dir):
    """Find images.csv and the images directory recursively under source_dir."""
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory '{source_path}' does not exist.")

    csv_candidates = list(source_path.rglob("images.csv"))
    if not csv_candidates:
        csv_candidates = list(source_path.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"Could not find images.csv in '{source_path}'.")

    csv_path = csv_candidates[0]

    # Search for image directory candidates.
    # The Kaggle clothing-dataset-full dataset uses 'images_original' or 'images_compressed' or 'images'.
    images_dir = None
    search_dirs = [csv_path.parent, source_path]
    folder_names = ["images_original", "images_compressed", "images", "img"]

    for parent in search_dirs:
        for fname in folder_names:
            candidate = parent / fname
            if candidate.is_dir() and any(candidate.glob("*.jpg")):
                images_dir = candidate
                break
        if images_dir:
            break

    if not images_dir:
        # Search recursively for folder named images_original, images_compressed, or images
        for fname in folder_names:
            candidates = [d for d in source_path.rglob(fname) if d.is_dir() and any(d.glob("*.jpg"))]
            if candidates:
                images_dir = candidates[0]
                break

    if not images_dir:
        # Fallback: search for any directory containing .jpg files
        candidates = [d for d in source_path.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]
        if candidates:
            images_dir = candidates[0]
        else:
            images_dir = csv_path.parent

    return csv_path, images_dir


def split_dataset(df, seed=42):
    """Split DataFrame into train (70%), val (15%), test (15%), preferring group-aware split by sender_id."""
    has_sender = "sender_id" in df.columns and df["sender_id"].nunique() > 10

    split_method = "plain_stratified"
    if has_sender:
        try:
            # Stage 1: 70% train, 30% temp
            gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
            train_idx, temp_idx = next(gss1.split(df, groups=df["sender_id"]))

            train_df = df.iloc[train_idx].copy()
            temp_df = df.iloc[temp_idx].copy()

            # Stage 2: 15% val, 15% test (50% of temp)
            gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
            val_sub_idx, test_sub_idx = next(gss2.split(temp_df, groups=temp_df["sender_id"]))

            val_df = temp_df.iloc[val_sub_idx].copy()
            test_df = temp_df.iloc[test_sub_idx].copy()

            # Verify that all classes exist in all 3 splits
            unique_classes = set(df["label"].unique())
            if (
                set(train_df["label"].unique()) == unique_classes
                and set(val_df["label"].unique()) == unique_classes
                and set(test_df["label"].unique()) == unique_classes
            ):
                split_method = "group_aware_by_sender_id"
                print("[Split] Group-aware split by sender_id succeeded (no contributor leakage across splits).")
                return train_df, val_df, test_df, split_method
            else:
                print("[Split] Group-aware split left classes unbalanced/missing in splits; falling back to stratified split.")
        except Exception as e:
            print(f"[Split] Group split attempt encountered: {e}; falling back to stratified split.")

    # Fallback: Plain stratified split
    train_df, temp_df = train_test_split(
        df,
        train_size=0.70,
        stratify=df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.50,
        stratify=temp_df["label"],
        random_state=seed,
    )
    print("[Split] Plain stratified split applied (70% train, 15% val, 15% test).")
    return train_df, val_df, test_df, "plain_stratified"


def _process_single_image(file_path, dest_file, img_size):
    """Worker function for threaded image processing."""
    try:
        dest_file = Path(dest_file)
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(file_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
            img.save(dest_file, format="JPEG", quality=95)
        return dest_file.parent.name, True
    except Exception as e:
        return str(e), False


def process_and_save_images(split_df, split_name, images_dir, output_dir, img_size=224):
    """Open, transpose, convert to RGB, resize with LANCZOS, and save each image.

    Uses threaded processing for ~2-4x speedup. PIL releases the GIL during
    JPEG decode/encode, so threads give real parallelism for image I/O.
    """
    split_dir = Path(output_dir) / split_name
    counts = Counter()

    # Build work items list
    work_items = []
    for _, row in split_df.iterrows():
        img_id = str(row["image"]).strip()
        label = str(row["label"]).strip()

        # Resolve image file: check directly and with common extensions
        file_path = None
        for ext in ["", ".jpg", ".jpeg", ".png", ".JPG"]:
            cand = images_dir / f"{img_id}{ext}"
            if cand.is_file():
                file_path = cand
                break
        if file_path is None:
            continue

        dest_file = split_dir / label / f"{Path(img_id).stem}.jpg"
        work_items.append((str(file_path), str(dest_file), img_size))

    if not work_items:
        return dict(counts)

    # Scale threads: on Colab (2 vCPUs), 8 threads keep CPUs fully loaded during I/O and PIL ops
    num_workers = min(16, max(4, (os.cpu_count() or 2) * 4), len(work_items))
    done = 0
    total = len(work_items)
    failed = 0

    print(f"  [{split_name}] Processing {total} images using {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_image, fp, dp, sz): (fp, dp, sz)
            for fp, dp, sz in work_items
        }
        for future in as_completed(futures):
            label_or_err, success = future.result()
            if success:
                counts[label_or_err] += 1
            else:
                failed += 1
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  [{split_name}] {done}/{total} images processed ({failed} failed)")

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Prepare clothing dataset for FashionVision v2.")
    parser.add_argument("--output-dir", default="data/clothing_prepared", help="Destination folder.")
    parser.add_argument("--source-dir", default=None, help="Existing folder with downloaded dataset.")
    parser.add_argument("--min-images-per-class", type=int, default=100, help="Minimum images per class.")
    parser.add_argument("--img-size", type=int, default=224, help="Target image size (square).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--synthetic", action="store_true", help="Generate tiny synthetic data for plumbing tests.")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip preparation if output directory already contains valid split_summary.json.")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    summary_file = output_path / "split_summary.json"
    if args.skip_if_exists and summary_file.is_file():
        print(f"[SKIP] Prepared data already found at '{output_path.resolve()}'. Skipping re-preparation.")
        return

    temp_synthetic_dir = None
    if args.synthetic:
        temp_synthetic_dir = tempfile.mkdtemp(prefix="fashion_synth_")
        source_path = generate_synthetic_raw_dataset(temp_synthetic_dir)
        # In synthetic mode, adjust threshold if default 100 was kept
        min_images = 5 if args.min_images_per_class == 100 else args.min_images_per_class
    elif args.source_dir:
        source_path = Path(args.source_dir)
        min_images = args.min_images_per_class
    else:
        source_path = download_dataset()
        min_images = args.min_images_per_class

    try:
        csv_path, images_dir = find_dataset_files(source_path)
        print(f"\nFound dataset CSV: '{csv_path}'")
        print(f"Found images directory: '{images_dir}'")

        df = pd.read_csv(csv_path)
        print(f"CSV Columns: {list(df.columns)}")
        print(f"Total CSV Rows: {len(df)}")

        # Normalize column names
        col_map = {c.lower().strip(): c for c in df.columns}
        img_col = col_map.get("image") or col_map.get("image_id") or list(df.columns)[0]
        label_col = col_map.get("label") or col_map.get("category") or "label"
        sender_col = col_map.get("sender_id") or col_map.get("sender")

        df = df.rename(columns={img_col: "image", label_col: "label"})
        if sender_col:
            df = df.rename(columns={sender_col: "sender_id"})

        # Initial inspection of labels and counts
        raw_counts = df["label"].value_counts().to_dict()
        print("\nRaw class distribution in CSV:")
        for cls_name, cnt in sorted(raw_counts.items()):
            print(f"  {cls_name:<20}: {cnt:>5}")

        # Check for missing/unreadable files
        print("\nVerifying image file availability...")
        valid_rows = []
        skipped_count = 0
        for idx, row in df.iterrows():
            img_id = str(row["image"]).strip()
            target_path = None
            for ext in ["", ".jpg", ".jpeg", ".png", ".JPG"]:
                cand = images_dir / f"{img_id}{ext}"
                if cand.is_file():
                    target_path = cand
                    break
            if target_path is None:
                skipped_count += 1
                continue
            valid_rows.append(row)

        df = pd.DataFrame(valid_rows)
        print(f"Usable images found: {len(df)} (Skipped missing files: {skipped_count})")
        if len(df) == 0:
            raise RuntimeError(
                f"No matching image files found in '{images_dir}' corresponding to '{csv_path}'. "
                f"Skipped all {skipped_count} entries. Please verify the dataset structure."
            )

        # Class filtering
        print("\nFiltering classes...")
        curr_counts = df["label"].value_counts().to_dict()
        kept_classes = []
        for cls_name, cnt in curr_counts.items():
            normalized = str(cls_name).strip().lower()
            if normalized in NON_CLOTHING_CATEGORIES:
                print(f"  [DROPPED] '{cls_name}': Non-clothing category.")
            elif cnt < min_images:
                print(f"  [DROPPED] '{cls_name}' (count={cnt}): Below min-images threshold ({min_images}).")
            else:
                kept_classes.append(str(cls_name).strip())

        kept_classes = sorted(list(set(kept_classes)))
        print(f"\nFinal sorted class list ({len(kept_classes)} classes):")
        for cls in kept_classes:
            print(f"  - {cls}")

        df = df[df["label"].isin(kept_classes)].copy()
        print(f"Total images after filtering: {len(df)}")

        # Split
        train_df, val_df, test_df, split_method = split_dataset(df, seed=args.seed)

        # Process and save images
        output_path = Path(args.output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing and resizing images into '{output_path.resolve()}'...")
        train_counts = process_and_save_images(train_df, "train", images_dir, output_path, args.img_size)
        val_counts = process_and_save_images(val_df, "val", images_dir, output_path, args.img_size)
        test_counts = process_and_save_images(test_df, "test", images_dir, output_path, args.img_size)

        # Build split summary
        summary = {
            "dataset_source": "kaggle: agrigorev/clothing-dataset-full",
            "seed": args.seed,
            "split_method": split_method,
            "img_size": [args.img_size, args.img_size],
            "class_names": kept_classes,
            "counts_per_split": {
                "train": train_counts,
                "val": val_counts,
                "test": test_counts,
            },
            "total_images_processed": len(train_df) + len(val_df) + len(test_df),
            "skipped_files_count": skipped_count,
        }

        summary_file = output_path / "split_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n=== Split Summary ===")
        print(json.dumps(summary, indent=2))
        print(f"\nData preparation complete. Output saved to '{output_path.resolve()}'.")

    finally:
        if temp_synthetic_dir and Path(temp_synthetic_dir).exists():
            shutil.rmtree(temp_synthetic_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
