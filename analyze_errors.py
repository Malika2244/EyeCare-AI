import os
import json
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. SETTINGS
# ==========================================

MODEL_PATH = "models/eye_disease_model.keras"
SPLIT_FILE = "models/dataset_split.json"
OUTPUT_DIR = "models/error_analysis"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. LOAD SAVED VALIDATION SPLIT
# ==========================================

with open(SPLIT_FILE, "r") as file:
    split_data = json.load(file)

validation_paths = split_data["validation"]

class_names = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

validation_labels = [
    class_to_index[
        os.path.basename(os.path.dirname(path))
    ]
    for path in validation_paths
]

print("\n========== DATASET INFORMATION ==========")
print("Validation images:", len(validation_paths))
print("Classes:", class_names)

# ==========================================
# 3. IMAGE PREPROCESSING
# ==========================================

def load_image(path):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)

    return image


validation_ds = tf.data.Dataset.from_tensor_slices(
    validation_paths
)

validation_ds = validation_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_ds = validation_ds.batch(BATCH_SIZE)
validation_ds = validation_ds.prefetch(
    tf.data.AUTOTUNE
)

# ==========================================
# 4. LOAD ORIGINAL MODEL
# ==========================================

print("\nLoading original MobileNetV2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

# ==========================================
# 5. MAKE PREDICTIONS
# ==========================================

print("\nAnalyzing validation images...")

probabilities = model.predict(
    validation_ds,
    verbose=1
)

y_true = np.array(validation_labels)

y_pred = np.argmax(
    probabilities,
    axis=1
)

confidence_scores = np.max(
    probabilities,
    axis=1
) * 100

# ==========================================
# 6. CALCULATE ACCURACY
# ==========================================

correct = int(
    np.sum(y_true == y_pred)
)

incorrect = int(
    np.sum(y_true != y_pred)
)

accuracy = (
    correct / len(y_true)
) * 100

print("\n========== VALIDATION RESULTS ==========")

print("Total images:", len(y_true))
print("Correct predictions:", correct)
print("Incorrect predictions:", incorrect)
print("Validation accuracy:", round(accuracy, 2), "%")

# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )
)

# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title(
    "Original Model - Validation Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "validation_confusion_matrix.png"
    ),
    dpi=200
)

plt.close()

# ==========================================
# 9. SAVE ALL PREDICTIONS TO CSV
# ==========================================

all_predictions_file = os.path.join(
    OUTPUT_DIR,
    "all_validation_predictions.csv"
)

with open(
    all_predictions_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image Path",
        "Actual Disease",
        "Predicted Disease",
        "Confidence (%)",
        "Correct Prediction"
    ])

    for index, path in enumerate(validation_paths):

        writer.writerow([
            path,
            class_names[y_true[index]],
            class_names[y_pred[index]],
            round(float(confidence_scores[index]), 2),
            y_true[index] == y_pred[index]
        ])

# ==========================================
# 10. SAVE INCORRECT PREDICTIONS
# ==========================================

error_file = os.path.join(
    OUTPUT_DIR,
    "incorrect_predictions.csv"
)

error_counter = Counter()

with open(
    error_file,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Image Path",
        "Actual Disease",
        "Predicted Disease",
        "Confidence (%)"
    ])

    for index, path in enumerate(validation_paths):

        if y_true[index] != y_pred[index]:

            actual = class_names[y_true[index]]
            predicted = class_names[y_pred[index]]

            error_counter[(actual, predicted)] += 1

            writer.writerow([
                path,
                actual,
                predicted,
                round(float(confidence_scores[index]), 2)
            ])

# ==========================================
# 11. DISEASE-WISE ERROR ANALYSIS
# ==========================================

print("\n========== DISEASE-WISE ERRORS ==========")

for actual_class in class_names:

    total_errors = sum(
        count
        for (actual, predicted), count
        in error_counter.items()
        if actual == actual_class
    )

    print(
        actual_class,
        "- Incorrect predictions:",
        total_errors
    )

# ==========================================
# 12. MOST COMMON CLASSIFICATION ERRORS
# ==========================================

print("\n========== MOST COMMON ERRORS ==========")

for (actual, predicted), count in error_counter.most_common():

    print(
        f"{actual} predicted as {predicted}: {count} images"
    )

# ==========================================
# 13. GLAUCOMA MISCLASSIFICATION
# ==========================================

glaucoma_as_normal = sum(
    1
    for actual, predicted in zip(y_true, y_pred)
    if (
        class_names[actual] == "glaucoma"
        and class_names[predicted] == "normal"
    )
)

normal_as_glaucoma = sum(
    1
    for actual, predicted in zip(y_true, y_pred)
    if (
        class_names[actual] == "normal"
        and class_names[predicted] == "glaucoma"
    )
)

print("\n========== GLAUCOMA ANALYSIS ==========")

print(
    "Glaucoma predicted as Normal:",
    glaucoma_as_normal
)

print(
    "Normal predicted as Glaucoma:",
    normal_as_glaucoma
)

# ==========================================
# 14. SAVE SUMMARY
# ==========================================

summary = {
    "model": MODEL_PATH,
    "dataset_split": "validation",
    "total_images": len(y_true),
    "correct_predictions": correct,
    "incorrect_predictions": incorrect,
    "accuracy_percent": round(accuracy, 2),
    "glaucoma_predicted_as_normal": glaucoma_as_normal,
    "normal_predicted_as_glaucoma": normal_as_glaucoma,
    "error_counts": [
        {
            "actual": actual,
            "predicted": predicted,
            "count": count
        }
        for (actual, predicted), count
        in error_counter.most_common()
    ]
}

summary_file = os.path.join(
    OUTPUT_DIR,
    "error_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )

print("\n========== ANALYSIS COMPLETED ==========")

print("Results saved in:", OUTPUT_DIR)

print("Files generated:")
print("- all_validation_predictions.csv")
print("- incorrect_predictions.csv")
print("- validation_confusion_matrix.png")
print("- error_summary.json")

print("\nNo dataset images or model files were modified.")