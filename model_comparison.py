
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. BASIC SETTINGS

DATA_DIR = "dataset"
MODEL_DIR = "comparison_models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 5
FINE_TUNE_EPOCHS = 3

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)

# 2. LOAD DATASET AND SPLIT


class_names = sorted([
    folder for folder in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, folder))
])

print("Classes:", class_names)

train_paths = []
train_labels = []

val_paths = []
val_labels = []

test_paths = []
test_labels = []

for label, class_name in enumerate(class_names):

    folder_path = os.path.join(DATA_DIR, class_name)

    image_files = sorted([
        file for file in os.listdir(folder_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    image_paths = [
        os.path.join(folder_path, file)
        for file in image_files
    ]

    random.shuffle(image_paths)

    total = len(image_paths)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_class = image_paths[:train_end]
    val_class = image_paths[train_end:val_end]
    test_class = image_paths[val_end:]

    train_paths.extend(train_class)
    train_labels.extend([label] * len(train_class))

    val_paths.extend(val_class)
    val_labels.extend([label] * len(val_class))

    test_paths.extend(test_class)
    test_labels.extend([label] * len(test_class))

    print(
        class_name,
        "Total:", total,
        "Train:", len(train_class),
        "Validation:", len(val_class),
        "Test:", len(test_class)
    )

print("\nTotal Training:", len(train_paths))
print("Total Validation:", len(val_paths))
print("Total Testing:", len(test_paths))

# 3. IMAGE PREPROCESSING


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


AUTOTUNE = tf.data.AUTOTUNE


def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        dataset = dataset.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = create_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = create_dataset(
    val_paths,
    val_labels
)

test_ds = create_dataset(
    test_paths,
    test_labels
)

# 4. MODEL BUILDING FUNCTION


def build_model(model_name):

    if model_name == "MobileNetV2":

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            name="feature_extractor"
        )

        preprocess = (
            tf.keras.applications.mobilenet_v2.preprocess_input
        )

    elif model_name == "ResNet50":

        base_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            name="feature_extractor"
        )

        preprocess = (
            tf.keras.applications.resnet50.preprocess_input
        )

    else:
        raise ValueError("Unknown model name")

    base_model.trainable = False

    inputs = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = preprocess(inputs)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# 5. COMPARE TWO MODELS


model_names = [
    "MobileNetV2",
    "ResNet50"
]

results = {}

print("\nMODEL COMPARISON STARTED")

for model_name in model_names:

    print("\nTraining:", model_name)

    tf.keras.backend.clear_session()

    model = build_model(model_name)

    checkpoint_path = os.path.join(
        MODEL_DIR,
        model_name + "_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max"
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    results[model_name] = best_val_accuracy

    print(
        model_name,
        "Best Validation Accuracy:",
        round(best_val_accuracy * 100, 2),
        "%"
    )

    del model

# 6. MODEL COMPARISON RESULTS

print("\nCOMPARISON RESULTS")

for name, score in results.items():

    print(
        name,
        ":",
        round(score * 100, 2),
        "%"
    )

best_model_name = max(
    results,
    key=results.get
)

print("\nBest Model:", best_model_name)

best_model_path = os.path.join(
    MODEL_DIR,
    best_model_name + "_best.keras"
)


# 7. LOAD BEST MODEL


tf.keras.backend.clear_session()

best_model = tf.keras.models.load_model(
    best_model_path
)

print("\nBest model loaded:", best_model_name)


# 8. FINE TUNING


print("\nFINE TUNING STARTED")

base_model = best_model.get_layer(
    "feature_extractor"
)

base_model.trainable = True

# Freeze all except last 30 layers

for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

best_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_path = os.path.join(
    MODEL_DIR,
    "best_finetuned_model.keras"
)

fine_tune_callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        fine_tune_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
]

fine_history = best_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_tune_callbacks
)

fine_tune_accuracy = max(
    fine_history.history["val_accuracy"]
)

print(
    "Fine-tuning Validation Accuracy:",
    round(fine_tune_accuracy * 100, 2),
    "%"
)


# 9. SELECT BEST VERSION


original_val_accuracy = results[best_model_name]

if fine_tune_accuracy > original_val_accuracy:

    print("\nFine-tuned model performed better.")

    final_model = tf.keras.models.load_model(
        fine_tune_path
    )

else:

    print("\nOriginal model performed better.")

    final_model = tf.keras.models.load_model(
        best_model_path
    )

final_model_path = os.path.join(
    MODEL_DIR,
    "best_eye_model_comparison.keras"
)

final_model.save(final_model_path)

print("\nFinal selected model saved!")


# 10. TEST SET EVALUATION


print("\nFINAL TEST EVALUATION")

test_loss, test_accuracy = final_model.evaluate(
    test_ds
)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

# 11. CLASSIFICATION REPORT

test_probabilities = final_model.predict(test_ds)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

y_test = np.array(test_labels)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)

# 12. THRESHOLD TUNING


print("\nTHRESHOLD TESTING")

val_probabilities = final_model.predict(val_ds)

y_val = np.array(val_labels)

val_predictions = np.argmax(
    val_probabilities,
    axis=1
)

val_confidence = np.max(
    val_probabilities,
    axis=1
)

thresholds = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

best_threshold = 0.50
best_threshold_accuracy = 0

print("\nValidation Threshold Results:")

for threshold in thresholds:

    accepted = val_confidence >= threshold

    coverage = np.mean(accepted)

    if np.sum(accepted) > 0:

        accepted_accuracy = accuracy_score(
            y_val[accepted],
            val_predictions[accepted]
        )

    else:

        accepted_accuracy = 0

    print(
        "Threshold:", threshold,
        "| Coverage:", round(coverage * 100, 2), "%",
        "| Accepted Accuracy:",
        round(accepted_accuracy * 100, 2), "%"
    )

    # Keep at least 50% predictions accepted

    if coverage >= 0.50:

        if accepted_accuracy > best_threshold_accuracy:

            best_threshold_accuracy = accepted_accuracy

            best_threshold = threshold

print("\nBest Threshold:", best_threshold)


# 13. APPLY THRESHOLD ON TEST DATA


test_confidence = np.max(
    test_probabilities,
    axis=1
)

accepted_test = (
    test_confidence >= best_threshold
)

test_coverage = np.mean(accepted_test)

if np.sum(accepted_test) > 0:

    accepted_test_accuracy = accuracy_score(
        y_test[accepted_test],
        test_predictions[accepted_test]
    )

else:

    accepted_test_accuracy = 0

print("\nTHRESHOLD TEST RESULTS")

print(
    "Test Coverage:",
    round(test_coverage * 100, 2),
    "%"
)

print(
    "Accuracy of Accepted Predictions:",
    round(accepted_test_accuracy * 100, 2),
    "%"
)

print(
    "Rejected Predictions:",
    np.sum(~accepted_test)
)


# 14. SAVE COMPARISON GRAPH


plt.figure(figsize=(9, 5))

plt.bar(
    list(results.keys()),
    [score * 100 for score in results.values()]
)

plt.title("Model Comparison - Validation Accuracy")

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.savefig(
    os.path.join(
        MODEL_DIR,
        "model_comparison.png"
    )
)

plt.show()

print("\nALL TASKS COMPLETED")
    