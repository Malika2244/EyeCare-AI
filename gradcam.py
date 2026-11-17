import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


# =====================================
# 1. SETTINGS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2_new_finetuned.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "mobilenetv2_classes.json"
)

IMG_SIZE = (224, 224)

# Experimental confidence threshold
CONFIDENCE_THRESHOLD = 80.0

DEFAULT_CLASS_NAMES = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]


# =====================================
# 2. LOAD CLASS NAMES
# =====================================

if os.path.exists(CLASS_NAMES_PATH):
    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        class_names = json.load(file)
else:
    class_names = DEFAULT_CLASS_NAMES

print("Classes:", class_names)


# =====================================
# 3. LOAD TRAINED MODEL
# =====================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Fine-tuned model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        "Model output classes do not match class names."
    )

print("Fine-Tuned MobileNetV2 loaded successfully!")
print("Confidence Threshold:", CONFIDENCE_THRESHOLD, "%")


# =====================================
# 4. ACCESS MOBILENETV2 LAYERS
# =====================================

base_model = model.get_layer(
    "mobilenetv2_1.00_224"
)

last_conv_layer = base_model.get_layer(
    "out_relu"
)

feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

pool_layer = model.get_layer(
    "global_average_pooling2d"
)

dropout_layer = model.get_layer(
    "dropout"
)

dense_layer = model.get_layer(
    "dense"
)

print("Grad-CAM layers loaded successfully!")


# =====================================
# 5. GENERATE GRAD-CAM
# =====================================

def generate_gradcam(
    image_path,
    output_path="static/gradcam_heatmap.jpg"
):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not os.path.isabs(output_path):
        output_path = os.path.join(
            BASE_DIR,
            output_path
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    # =====================================
    # 6. PREPARE IMAGE
    # =====================================

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = tf.convert_to_tensor(
        np.asarray(
            original_image,
            dtype=np.float32
        )
    )

    image_tensor = tf.image.resize(
        image_tensor,
        IMG_SIZE
    )

    raw_image_batch = tf.expand_dims(
        image_tensor,
        axis=0
    )

    # The complete saved model already has
    # MobileNetV2 preprocessing inside it.
    #
    # For Grad-CAM, we call the backbone
    # directly, so preprocess once here.

    preprocessed_image = (
        tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.identity(raw_image_batch)
        )
    )

    # =====================================
    # 7. PREDICTION + GRADIENTS
    # =====================================

    with tf.GradientTape() as tape:

        conv_outputs, base_outputs = feature_model(
            preprocessed_image,
            training=False
        )

        x = pool_layer(base_outputs)

        x = dropout_layer(
            x,
            training=False
        )

        predictions = dense_layer(x)

        predicted_class = tf.argmax(
            predictions[0],
            output_type=tf.int32
        )

        class_score = tf.gather(
            predictions[0],
            predicted_class
        )

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    # =====================================
    # 8. GENERATE HEATMAP
    # =====================================

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_features = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_features * weights[0],
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    # =====================================
    # 9. GET PREDICTION
    # =====================================

    predicted_index = int(
        predicted_class.numpy()
    )

    predicted_disease = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[0, predicted_index].numpy()
        * 100
    )

    # =====================================
    # 10. APPLY 80% THRESHOLD
    # =====================================

    is_uncertain = (
        confidence < CONFIDENCE_THRESHOLD
    )

    if is_uncertain:
        prediction_status = "Uncertain"

        warning_message = (
            "Low-confidence prediction. "
            "Please consult an eye specialist."
        )
    else:
        prediction_status = "Accepted"
        warning_message = ""

    # =====================================
    # 11. CREATE COLOURED HEATMAP
    # =====================================

    heatmap_image = Image.fromarray(
        np.uint8(
            np.clip(heatmap, 0, 1) * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        ) / 255.0
    )

    colored_heatmap = plt.colormaps["jet"](
        heatmap_array
    )[:, :, :3]

    colored_heatmap = np.uint8(
        255 * colored_heatmap
    )

    colored_heatmap_image = Image.fromarray(
        colored_heatmap
    )

    # =====================================
    # 12. OVERLAY HEATMAP
    # =====================================

    overlay = Image.blend(
        original_image,
        colored_heatmap_image,
        alpha=0.4
    )

    overlay = overlay.convert("RGB")

    overlay.save(
        output_path,
        quality=95
    )

    # =====================================
    # 13. PRINT RESULTS
    # =====================================

    print("\n========== GRAD-CAM RESULT ==========")

    print("Disease:", predicted_disease)

    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )

    print(
        "Threshold:",
        CONFIDENCE_THRESHOLD,
        "%"
    )

    print("Status:", prediction_status)

    if warning_message:
        print("Warning:", warning_message)

    print("Heatmap saved at:", output_path)

    # =====================================
    # 14. RETURN TO FLASK
    # =====================================

    return {
        "disease": predicted_disease,
        "confidence": round(confidence, 2),
        "heatmap_path": output_path,
        "is_uncertain": bool(is_uncertain),
        "threshold": CONFIDENCE_THRESHOLD,
        "prediction_status": prediction_status,
        "warning_message": warning_message
    }


# =====================================
# 15. TEST INDEPENDENTLY
# =====================================

if __name__ == "__main__":

    test_image = os.path.join(
        BASE_DIR,
        "dataset",
        "cataract",
        "0_left.jpg"
    )

    if not os.path.exists(test_image):
        print("Test image not found:", test_image)
        print("Choose an existing retinal image.")

    else:
        result = generate_gradcam(
            test_image,
            "static/gradcam_heatmap.jpg"
        )

        print("\n========== FINAL RESULT ==========")

        print("Disease:", result["disease"])
        print("Confidence:", result["confidence"], "%")
        print("Threshold:", result["threshold"], "%")
        print("Status:", result["prediction_status"])
        print("Heatmap:", result["heatmap_path"])