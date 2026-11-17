import os
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# 1. BASIC SETTINGS


DATA_DIR = "dataset"
MODEL_DIR = "models"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LAYERS = 30

SEED = 42

# Example threshold, not clinically validated
CONFIDENCE_THRESHOLD = 0.80

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)


# 2. READ DATASET

class_names = sorted([
    folder for folder in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, folder))
])

print("Classes:", class_names)

train_all_paths = []
train_all_labels = []

val_all_paths = []
val_all_labels = []

test_all_paths = []
test_all_labels = []

for label, class_name in enumerate(class_names):

    folder_path = os.path.join(DATA_DIR, class_name)

    image_files = sorted([
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    image_paths = [
        os.path.join(folder_path, file)
        for file in image_files
    ]

    random.shuffle(image_paths)

    total = len(image_paths)

    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    train_paths = image_paths[:train_end]
    val_paths = image_paths[train_end:val_end]
    test_paths = image_paths[val_end:]

    train_all_paths.extend(train_paths)
    train_all_labels.extend([label] * len(train_paths))

    val_all_paths.extend(val_paths)
    val_all_labels.extend([label] * len(val_paths))

    test_all_paths.extend(test_paths)
    test_all_labels.extend([label] * len(test_paths))

    print(
        f"{class_name}: {total} images | "
        f"Train: {len(train_paths)} | "
        f"Validation: {len(val_paths)} | "
        f"Test: {len(test_paths)}"
    )

print("\n DATASET SPLIT")

print("Training:", len(train_all_paths))
print("Validation:", len(val_all_paths))
print("Testing:", len(test_all_paths))

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

    # MobileNetV2 preprocessing is inside the model
    return image, label

# 4. CREATE DATASETS


AUTOTUNE = tf.data.AUTOTUNE

def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
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
    train_all_paths, train_all_labels, training=True
)

val_ds = create_dataset(
    val_all_paths, val_all_labels
)

test_ds = create_dataset(
    test_all_paths, test_all_labels
)

# 5. LOAD MOBILENETV2


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Initially freeze all pretrained layers
base_model.trainable = False

# 6. BUILD MODEL


inputs = tf.keras.Input(shape=(224, 224, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(
    inputs
)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)

model.summary()


# 7. INITIAL TRAINING


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

initial_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_initial_best.keras"
)

initial_callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=initial_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min"
    )
]

print("\nINITIAL TRAINING")

history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=initial_callbacks
)

initial_model = tf.keras.models.load_model(
    initial_path
)


# 8. FINE-TUNING


print("\n FINE-TUNING STARTED")

# Unfreeze MobileNetV2
base_model.trainable = True

# Freeze all except last 30 layers
fine_tune_at = len(base_model.layers) - FINE_TUNE_LAYERS

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Freeze BatchNormalization layers
for layer in base_model.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

print("Total MobileNetV2 layers:", len(base_model.layers))
print("Fine-tuning starts from:", fine_tune_at)

# Recompile with a smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.00001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fine_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_finetuned_best.keras"
)

fine_callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=fine_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min"
    )
]

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_callbacks
)

fine_model = tf.keras.models.load_model(
    fine_path
)

# 9. COMPARE BOTH MODELS

print("\nMODEL COMPARISON")

initial_loss, initial_accuracy = initial_model.evaluate(
    val_ds,
    verbose=0
)

fine_loss, fine_accuracy = fine_model.evaluate(
    val_ds,
    verbose=0
)

print(
    "Initial Validation Accuracy:",
    round(initial_accuracy * 100, 2), "%"
)

print(
    "Fine-Tuned Validation Accuracy:",
    round(fine_accuracy * 100, 2), "%"
)

if fine_loss <= initial_loss:

    best_model = fine_model
    selected_model = "Fine-Tuned MobileNetV2"

else:

    best_model = initial_model
    selected_model = "Initial MobileNetV2"

print("Selected Model:", selected_model)

# 10. FINAL TESTING


print("\nFINAL TESTING")

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\nFINAL TEST RESULTS")

print("Selected Model:", selected_model)
print("Test Loss:", round(test_loss, 4))

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2), "%"
)

# 11. CONFIDENCE THRESHOLD EVALUATION

print("\nTHRESHOLD TEST")

predictions = best_model.predict(
    test_ds,
    verbose=0
)

predicted_labels = np.argmax(predictions, axis=1)
confidences = np.max(predictions, axis=1)

true_labels = np.array(test_all_labels)

accepted = confidences >= CONFIDENCE_THRESHOLD

accepted_count = int(np.sum(accepted))
uncertain_count = int(np.sum(~accepted))

print(
    "Example Confidence Threshold:",
    CONFIDENCE_THRESHOLD * 100, "%"
)

print("Accepted Predictions:", accepted_count)
print("Uncertain Predictions:", uncertain_count)

if accepted_count > 0:

    accepted_accuracy = np.mean(
        predicted_labels[accepted] == true_labels[accepted]
    )

    print(
        "Accuracy on Accepted Predictions:",
        round(float(accepted_accuracy) * 100, 2),
        "%"
    )

print(
    "Coverage:",
    round(float(np.mean(accepted)) * 100, 2),
    "%"
)

# 12. SAVE FINAL MODEL


# Save separately to protect the website model
final_path = os.path.join(
    MODEL_DIR,
    "mobilenetv2_new_finetuned.keras"
)

best_model.save(final_path)

with open(
    os.path.join(MODEL_DIR, "mobilenetv2_classes.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(class_names, file, indent=4)

print("\nNew Model Saved:", final_path)

# 13. ACCURACY GRAPH

train_accuracy = (
    history_initial.history["accuracy"]
    + history_fine.history["accuracy"]
)

val_accuracy = (
    history_initial.history["val_accuracy"]
    + history_fine.history["val_accuracy"]
)

plt.figure(figsize=(9, 5))

plt.plot(
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    x=len(history_initial.history["accuracy"]) - 0.5,
    linestyle="--",
    label="Fine-Tuning Started"
)

plt.title("MobileNetV2 Fine-Tuning Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(MODEL_DIR, "accuracy_plot_finetuned.png")
)

plt.close()

# 14. LOSS GRAPH


train_loss = (
    history_initial.history["loss"]
    + history_fine.history["loss"]
)

val_loss = (
    history_initial.history["val_loss"]
    + history_fine.history["val_loss"]
)

plt.figure(figsize=(9, 5))

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.axvline(
    x=len(history_initial.history["loss"]) - 0.5,
    linestyle="--",
    label="Fine-Tuning Started"
)

plt.title("MobileNetV2 Fine-Tuning Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(MODEL_DIR, "loss_plot_finetuned.png")
)

plt.close()

print("\nTRAINING COMPLETED")