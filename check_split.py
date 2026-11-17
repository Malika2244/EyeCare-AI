import os
import random

DATA_DIR = "dataset"
SEED = 42

def get_split():
    random.seed(SEED)

    train_paths = []
    val_paths = []
    test_paths = []

    class_names = sorted([
        folder for folder in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, folder))
    ])

    for class_name in class_names:
        folder_path = os.path.join(DATA_DIR, class_name)

        image_files = [
            file for file in os.listdir(folder_path)
            if file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        paths = [
            os.path.join(folder_path, file)
            for file in image_files
        ]

        random.shuffle(paths)

        total = len(paths)
        train_end = int(total * 0.70)
        val_end = int(total * 0.85)

        train_paths.extend(paths[:train_end])
        val_paths.extend(paths[train_end:val_end])
        test_paths.extend(paths[val_end:])

    return train_paths, val_paths, test_paths


train, val, test = get_split()

print("Training images:", len(train))
print("Validation images:", len(val))
print("Testing images:", len(test))

print("Train/Test overlap:", len(set(train) & set(test)))
print("Train/Validation overlap:", len(set(train) & set(val)))
print("Validation/Test overlap:", len(set(val) & set(test)))
import json

os.makedirs("models", exist_ok=True)

split_data = {
    "train": train,
    "validation": val,
    "test": test
}

with open("models/dataset_split.json", "w") as f:
    json.dump(split_data, f, indent=4)

print("\nDataset split saved successfully!")
print("File: models/dataset_split.json")