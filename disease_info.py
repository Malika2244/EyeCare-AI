
disease_information = {
    "cataract": {
        "name": "Cataract",
        "description": "Cataract is a condition in which the natural lens of the eye becomes cloudy.",
        "symptoms": [
            "Blurred vision",
            "Difficulty seeing at night",
            "Sensitivity to light",
            "Faded or yellow colors"
        ],
        "causes": [
            "Age-related changes",
            "Diabetes",
            "Eye injury",
            "Long-term exposure to sunlight"
        ]
    },

    "diabetic_retinopathy": {
        "name": "Diabetic Retinopathy",
        "description": "Diabetic retinopathy is an eye condition that can develop in people with diabetes due to damage to retinal blood vessels.",
        "symptoms": [
            "Blurred vision",
            "Dark spots or floaters",
            "Difficulty seeing colors",
            "Vision changes"
        ],
        "causes": [
            "Long-term high blood sugar",
            "Diabetes",
            "High blood pressure"
        ]
    },

    "glaucoma": {
        "name": "Glaucoma",
        "description": "Glaucoma is a group of eye conditions that can damage the optic nerve and may lead to vision loss.",
        "symptoms": [
            "Loss of peripheral vision",
            "Eye pain in some types",
            "Blurred vision",
            "Halos around lights"
        ],
        "causes": [
            "Increased eye pressure is an important risk factor",
            "Family history",
            "Age-related risk"
        ]
    },

    "normal": {
        "name": "Normal Eye",
        "description": "The model classified the uploaded image as belonging to the normal category.",
        "symptoms": [
            "No disease detected by this model"
        ],
        "causes": [
            "This category represents normal images in the training dataset."
        ]
    }
}


def get_disease_info(disease_name):

    disease_name = disease_name.lower().replace(" ", "_")

    return disease_information.get(
        disease_name,
        {
            "name": "Information Not Available",
            "description": "No information found.",
            "symptoms": [],
            "causes": []
        }
    )


# Test the module
if __name__ == "__main__":

    info = get_disease_info("cataract")

    print("Disease:", info["name"])
    print("Description:", info["description"])

    print("\nSymptoms:")
    for symptom in info["symptoms"]:
        print("-", symptom)

    print("\nCauses:")
    for cause in info["causes"]:
        print("-", cause)
