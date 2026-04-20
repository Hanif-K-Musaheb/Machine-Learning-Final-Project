# dataPrep.py
import os
import math
import random
import shutil
import constants

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def split_number(total, percentages):
    """
    Convert percentages into integer counts that sum exactly to 'total',
    distributing rounding leftovers by largest fractional part.
    """
    exact_splits = [total * p for p in percentages]
    integer_splits = [math.floor(x) for x in exact_splits]
    remainder = total - sum(integer_splits)

    fractional_parts = [(exact_splits[i] - integer_splits[i], i) for i in range(len(exact_splits))]
    fractional_parts.sort(reverse=True, key=lambda x: x[0])

    for i in range(remainder):
        index_to_add = fractional_parts[i][1]
        integer_splits[index_to_add] += 1

    return integer_splits


def list_images(folder_path):
    """Return full paths to image files in folder_path."""
    files = []
    for f in os.listdir(folder_path):
        full = os.path.join(folder_path, f)
        if os.path.isfile(full) and f.lower().endswith(IMAGE_EXTS):
            files.append(full)
    return files


def split_data(city_folder_path, seed=42):
    """
    Split one city folder into (train_files, val_files, test_files).
    Split is reproducible with the same seed.
    """
    files = list_images(city_folder_path)
    rng = random.Random(seed)
    rng.shuffle(files)

    total = len(files)
    if total == 0:
        return [], [], []

    # We will use ordering: train, validation, test
    n_train, n_val, n_test = split_number(
        total,
        [constants.TRAIN_SPLIT, constants.VALIDATION_SPLIT, constants.TEST_SPLIT]
    )

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:n_train + n_val + n_test]

    return train_files, val_files, test_files


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_files_to_folder(file_list, out_dir):
    """
    Copy each file in file_list into out_dir.
    If a filename collision occurs, append a suffix.
    """
    ensure_dir(out_dir)

    for src in file_list:
        base = os.path.basename(src)
        dst = os.path.join(out_dir, base)

        if os.path.exists(dst):
            name, ext = os.path.splitext(base)
            k = 1
            while True:
                dst2 = os.path.join(out_dir, f"{name}__dup{k}{ext}")
                if not os.path.exists(dst2):
                    dst = dst2
                    break
                k += 1

        shutil.copy2(src, dst)


def split_all_cities_to_folders(
    raw_root="archive/images",
    output_root="data",
    seed=42
):
    
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"Raw root folder not found: {raw_root}")

    # Create top-level split dirs
    ensure_dir(os.path.join(output_root, "train"))
    ensure_dir(os.path.join(output_root, "validation"))
    ensure_dir(os.path.join(output_root, "test"))

    cities = [d for d in os.listdir(raw_root) if os.path.isdir(os.path.join(raw_root, d))]
    cities.sort()

    summary = {}

    for city in cities:
        city_dir = os.path.join(raw_root, city)

        train_files, val_files, test_files = split_data(city_dir, seed=seed)

        out_train = os.path.join(output_root, "train", city)
        out_val = os.path.join(output_root, "validation", city)
        out_test = os.path.join(output_root, "test", city)

        copy_files_to_folder(train_files, out_train)
        copy_files_to_folder(val_files, out_val)
        copy_files_to_folder(test_files, out_test)

        summary[city] = {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
            "total": len(train_files) + len(val_files) + len(test_files)
        }

    return summary
