"""
Dataset download script for Cats vs Dogs classification.
Supports Kaggle API download with fallback to manual structure.
"""

import os
import zipfile
import argparse
import shutil
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

KAGGLE_DATASET = "bhavikjikadara/dog-and-cat-classification-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}/download"


def download_via_kaggle_api():
    """Download dataset using Kaggle API (requires kagglehub or kaggle CLI)."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"Downloaded via kagglehub to: {path}")
        return Path(path)
    except ImportError:
        print("kagglehub not installed, trying kaggle CLI...")

    import shutil
    import subprocess
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Neither kagglehub nor the kaggle CLI is installed.\n"
            "Install with: pip install kagglehub  (no Kaggle credentials needed)\n"
            "Or place your dataset manually in data/raw/ with 'cat' and 'dog' subfolders."
        )

    result = subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET, "--path", str(RAW_DATA_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle CLI failed. Install with: pip install kagglehub\n"
            "Or place your dataset manually in data/raw/ with 'cat' and 'dog' subfolders.\n"
            f"Error: {result.stderr}"
        )
    print("Downloaded via kaggle CLI.")
    return RAW_DATA_DIR


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract a zip file."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")
    zip_path.unlink()


def _merge_images(src_folder: Path, dest_folder: Path):
    """Copy all images from src_folder into dest_folder, skipping duplicates."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_folder.iterdir()):
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            target = dest_folder / f.name
            if not target.exists():
                shutil.copy2(f, target)
                count += 1
    return count


def organize_into_folders(source_dir: Path, dest_dir: Path):
    """
    Organize raw images into cat/ and dog/ subdirectories.

    The bhavikjikadara dataset is nested, e.g.:
        training_set/training_set/{cats,dogs}
        test_set/test_set/{cats,dogs}
    so we walk the tree recursively, find every folder named (cat|dog),
    and merge all their images into dest_dir/cat and dest_dir/dog.

    Falls back to filename-prefix categorization for flat structures
    ('cat.*' / 'dog.*').
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    cat_dir, dog_dir = dest_dir / "cat", dest_dir / "dog"

    label_folders = {"cat", "cats", "dog", "dogs", "petimages"}
    merged_cats = merged_dogs = 0
    found_nested = False

    for root, dirs, files in os.walk(source_dir):
        for d in dirs:
            # The Kaggle "PetImages" layout nests actual class dirs one level
            # deeper (PetImages/Cat, PetImages/Dog); recursing into it lets the
            # walk find the real cat/dog folders.
            if d.lower() == "petimages":
                continue
            if d.lower() in label_folders:
                src = Path(root) / d
                dst = cat_dir if d.lower().startswith("cat") else dog_dir
                n = _merge_images(src, dst)
                if d.lower().startswith("cat"):
                    merged_cats += n
                else:
                    merged_dogs += n
                found_nested = True

    if found_nested:
        print(f"Organized {merged_cats} cat images, {merged_dogs} dog images into {dest_dir}")
        return

    # Flat fallback: categorize by filename prefix
    cat_dir.mkdir(exist_ok=True)
    dog_dir.mkdir(exist_ok=True)
    for f in sorted(source_dir.iterdir()):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if f.name.lower().startswith("cat"):
                shutil.copy2(f, cat_dir / f.name)
            elif f.name.lower().startswith("dog"):
                shutil.copy2(f, dog_dir / f.name)

    n_cats = len(list(cat_dir.iterdir()))
    n_dogs = len(list(dog_dir.iterdir()))
    print(f"Organized {n_cats} cat images, {n_dogs} dog images into {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download Cats vs Dogs dataset")
    parser.add_argument("--force", action="store_true", help="Redownload even if data exists")
    args = parser.parse_args()

    if RAW_DATA_DIR.exists() and not args.force:
        print(f"Raw data already exists at {RAW_DATA_DIR}. Use --force to redownload.")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # download_via_kaggle_api returns the directory the files were actually
    # written to (kagglehub caches under ~/.cache/kagglehub, NOT data/raw/).
    download_path = download_via_kaggle_api()

    # Check for zip files and extract
    for z in download_path.glob("*.zip"):
        extract_zip(z, download_path)

    organize_into_folders(download_path, PROCESSED_DATA_DIR / "organized")

    print("Download complete. Run preprocess.py to prepare data for training.")


if __name__ == "__main__":
    main()
