
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# =====================================
# 1. Basic Settings
# =====================================

DATA_DIR = "dataset"
MODEL_PATH = "models/eye_disease_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================
# 2. Get Class Names
# =====================================

class_names = sorted([
    folder for folder in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, folder))
])

print("Classes:", class_names)

# =====================================
# 3. Recreate Same Test Split
# =====================================

test_paths = []
test_labels = []

for label, class_name in enumerate(class_names):

    folder_path = os.path.join(DATA_DIR, class_name)

    image_files = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_paths = [
        os.path.join(folder_path, file)
        for file in image_files
    ]

    random.shuffle(image_paths)

    total = len(image_paths)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    class_test_paths = image_paths[val_end:]

    test_paths.extend(class_test_paths)
    test_labels.extend([label] * len(class_test_paths))

print("\nTotal test images:", len(test_paths))

# =====================================
# 4. Load Test Images
# =====================================

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)

    return image, label


test_ds = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# =====================================
# 5. Load Trained Model
# =====================================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

# =====================================
# 6. Predict Test Images
# =====================================

y_true = []
y_pred = []

for images, labels in test_ds:

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =====================================
# 7. Calculate Accuracy
# =====================================

accuracy = accuracy_score(y_true, y_pred)

print("\n========== MODEL ACCURACY ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

# =====================================
# 8. Classification Report
# =====================================

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

# =====================================
# 9. Confusion Matrix
# =====================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

# =====================================
# 10. Plot Confusion Matrix
# =====================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Eye Disease Confusion Matrix")
plt.tight_layout()

plt.savefig("models/confusion_matrix.png")
plt.show()

print("\nEvaluation completed successfully!")
