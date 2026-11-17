
import tensorflow as tf
import numpy as np
from PIL import Image

# Import Disease Information Module
from disease_info import get_disease_info


# Load trained model
model = tf.keras.models.load_model(
    "models/eye_disease_model.keras"
)

# Disease classes
class_names = [
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal"
]


# Prediction Function
def predict_eye_disease(image_path):

    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    # Convert image to array
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    # Get predicted disease
    predicted_index = np.argmax(prediction[0])

    disease = class_names[predicted_index]

    confidence = prediction[0][predicted_index] * 100

    return disease, float(confidence)


# Image path
image_path = "dataset/cataract/0_left.jpg"


# Run Prediction
disease, confidence = predict_eye_disease(image_path)


# Display Prediction
print("\nEYE DISEASE PREDICTION")

print("Disease:", disease.replace("_", " ").title())

print("Confidence:", round(confidence, 2), "%")


# Get Disease Information Automatically
info = get_disease_info(disease)


# Display Disease Information
print("\nDISEASE INFORMATION")

print("\nDisease:", info["name"])

print("\nDescription:")
print(info["description"])

print("\nSymptoms:")

for symptom in info["symptoms"]:
    print("-", symptom)


print("\nCauses:")

for cause in info["causes"]:
    print("-", cause)


print("\nNote:")
print("This AI result is for educational purposes only.")
print("Please consult an eye specialist for medical advice.")
