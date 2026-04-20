import os
from dataPrep import split_data

CITY_FOLDER = "archive/images/Boston"  # CHANGE to a real path on your machine

def test_folder_exists():
    assert os.path.isdir(CITY_FOLDER), f"Folder not found: {CITY_FOLDER}"

def test_split_not_empty():
    train, val, test = split_data(CITY_FOLDER, seed=42)
    total = len(train) + len(val) + len(test)
    assert total > 0, "No images found (check folder or extensions)."

def test_split_disjoint():
    train, val, test = split_data(CITY_FOLDER, seed=42)
    s_train, s_val, s_test = set(train), set(val), set(test)
    assert len(s_train & s_val) == 0, "Train/Val overlap!"
    assert len(s_train & s_test) == 0, "Train/Test overlap!"
    assert len(s_val & s_test) == 0, "Val/Test overlap!"

def test_split_reproducible_same_seed():
    train1, val1, test1 = split_data(CITY_FOLDER, seed=42)
    train2, val2, test2 = split_data(CITY_FOLDER, seed=42)
    print(split_data(CITY_FOLDER, seed=42))
    assert train1 == train2 and val1 == val2 and test1 == test2, "Not reproducible with same seed."