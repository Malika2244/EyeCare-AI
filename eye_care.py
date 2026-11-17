
eye_care_recommendations = {

    "cataract": {
        "title": "Cataract Eye Care Recommendations",
        "recommendations": [
            "Wear sunglasses that provide 100% UV protection.",
            "Schedule regular eye examinations.",
            "Avoid smoking.",
            "Maintain a healthy and balanced diet.",
            "Consult an eye specialist if vision becomes blurry."
        ]
    },

    "diabetic_retinopathy": {
        "title": "Diabetic Retinopathy Eye Care Recommendations",
        "recommendations": [
            "Keep blood sugar levels under medical supervision.",
            "Monitor blood pressure regularly.",
            "Schedule regular dilated retinal examinations.",
            "Follow your diabetes treatment plan.",
            "Consult an eye specialist about any vision changes."
        ]
    },

    "glaucoma": {
        "title": "Glaucoma Eye Care Recommendations",
        "recommendations": [
            "Schedule regular eye pressure examinations.",
            "Follow prescribed eye medication instructions.",
            "Never stop glaucoma medication without consulting your doctor.",
            "Attend regular appointments with an eye specialist.",
            "Seek urgent medical attention for sudden severe eye pain or vision changes."
        ]
    },

    "normal": {
        "title": "Healthy Eye Care Recommendations",
        "recommendations": [
            "Eat a balanced diet containing green vegetables.",
            "Take regular breaks during screen use.",
            "Wear UV-protective sunglasses outdoors.",
            "Maintain good eye hygiene.",
            "Schedule routine eye examinations."
        ]
    }
}


def get_eye_care_recommendations(disease_name):

    disease_name = disease_name.lower().replace(" ", "_")

    return eye_care_recommendations.get(
        disease_name,
        {
            "title": "General Eye Care",
            "recommendations": [
                "Consult an eye specialist for personalized advice."
            ]
        }
    )


# Test the module
if __name__ == "__main__":

    info = get_eye_care_recommendations("cataract")

    print("\n========== EYE CARE RECOMMENDATIONS ==========")

    print("\n", info["title"])

    for recommendation in info["recommendations"]:
        print("-", recommendation)
