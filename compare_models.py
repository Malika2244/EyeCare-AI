import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# SETTINGS
# =====================================

OLD_MODEL = "models/eye_disease_model.keras"
NEW_MODEL = "models/eye_disease_improved.keras"
SPLIT_FILE = "models/dataset_split.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

# =====================================
# LOAD FIXED TEST DATA
# =====================================

with open(SPLIT_FILE, "r") as f:
    split_data = json.load(f)

test_paths = split_data["test"]

class_names = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]

class_to_index = {
    name: i for i, name in enumerate(class_names)
}

test_labels = [
    class_to_index[
        os.path.basename(os.path.dirname(path))
    ]
    for path in test_paths
]

def load_image(path, label):
    image = tf.io.read_file(path)

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
# COMPARE MODELS
# =====================================

for name, path in [
    ("ORIGINAL MODEL", OLD_MODEL),
    ("IMPROVED MODEL", NEW_MODEL)
]:

    print("\n================================")
    print(name)
    print("================================")

    model = tf.keras.models.load_model(path)

    predictions = model.predict(
        test_ds,
        verbose=0
    )

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.array(test_labels)

    accuracy = accuracy_score(y_true, y_pred)

    print("Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(class_names)))
        )
    )

    del model

print("\nCOMPARISON COMPLETED")