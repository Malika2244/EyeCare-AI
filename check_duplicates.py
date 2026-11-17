import os
import json
import hashlib
from collections import defaultdict

# ==========================================
# SETTINGS
# ==========================================

SPLIT_FILE = "models/dataset_split.json"

# ==========================================
# LOAD SAVED SPLIT
# ==========================================

with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

print("\n========== DATASET QUALITY CHECK ==========")

hash_groups = defaultdict(list)

# ==========================================
# CALCULATE IMAGE HASHES
# ==========================================

for split_name in ["train", "validation", "test"]:

    for path in splits[split_name]:

        if not os.path.isfile(path):
            print("Missing image:", path)
            continue

        with open(path, "rb") as image_file:
            image_hash = hashlib.sha256(
                image_file.read()
            ).hexdigest()

        hash_groups[image_hash].append({
            "split": split_name,
            "path": path
        })

# ==========================================
# FIND DUPLICATES
# ==========================================

duplicate_groups = [
    images
    for images in hash_groups.values()
    if len(images) > 1
]

cross_split_groups = [
    images
    for images in duplicate_groups
    if len(set(item["split"] for item in images)) > 1
]

print("\nTotal images:",
      sum(len(splits[k]) for k in ["train", "validation", "test"]))

print("Exact duplicate groups:", len(duplicate_groups))
print("Cross-split duplicate groups:", len(cross_split_groups))

# ==========================================
# PRINT CROSS-SPLIT DUPLICATES
# ==========================================

if cross_split_groups:

    print("\nWARNING: Cross-split duplicates found!")

    for group in cross_split_groups[:10]:
        print("\nDuplicate group:")
        for item in group:
            print(item["split"], ":", item["path"])

else:
    print("\nNo exact cross-split duplicates found.")

# ==========================================
# SAVE REPORT
# ==========================================

report = {
    "exact_duplicate_groups": duplicate_groups,
    "cross_split_duplicate_groups": cross_split_groups
}

os.makedirs("models", exist_ok=True)

with open(
    "models/duplicate_report.json",
    "w"
) as f:
    json.dump(report, f, indent=4)

print("\nReport saved: models/duplicate_report.json")
print("Dataset quality check completed!")