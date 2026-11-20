EyeCare AI - Eye Disease Classification System
This project is about detecting eye diseases from eye images using Deep Learning. I made this project as my final year project.
The main idea is very simple. User will upload an eye image and the system will tell which disease it is, like Cataract, Diabetic Retinopathy, Glaucoma or Normal. Along with prediction, it also gives some basic information about eye care.
I used MobileNetV2 model for this because it is lightweight and works well for image classification. I used Transfer Learning for training.
Modules
In this project there are total 9 modules:
Module 1: Eye Disease Classification
User uploads eye image and model predicts the disease category.
Module 2: Severity Check
After prediction, it gives basic severity related information so user can understand the result.
Module 3: Grad-CAM Heatmap
It shows heatmap on image so we can see which part the model focused on for prediction.
Module 4: Eye Surgery Guide
This module gives information about eye surgeries like LASIK, PRK and SMILE. It also has how surgery is performed and recovery guide.
Module 5: Disease Information
Here user can read basic details and symptoms of different eye diseases.
Module 6: Eye Care Recommendations
This gives simple eye care tips and recommendations to keep eyes healthy.
Module 7: Prediction History
User can see his/her previous predictions in this section.
Module 8: AI Chatbot
Chatbot is made with Gemini API. User can ask general eye health related questions.
Module 9: Nearby Eye Centres
This helps user to find nearby eye hospitals and eye care centres using location based search. It uses OpenStreetMap / Overpass API.
Model Details
Model: MobileNetV2
Framework: TensorFlow / Keras
Input size: 224 x 224
Classes: 4 (Cataract, Diabetic Retinopathy, Glaucoma, Normal)
Working is like this: User uploads image -> Image preprocessing -> Model extracts features -> Model predicts class -> Result is shown with confidence.
Results
I got these results after training and testing:
Validation Accuracy: 90.05%
Testing Accuracy: 88.03%
Test Loss: 0.3235
These results are from my experiments. Accuracy can be different if dataset or training changes.
Dataset
Total images in dataset are 4217.
Cataract: 1038
Diabetic Retinopathy: 1098
Glaucoma: 1007
Normal: 1074
I divided dataset like this:
Training: 2949
Validation: 633
Testing: 635
Technologies Used
I used Python, Flask for backend, TensorFlow and Keras for model, HTML, CSS, JavaScript for frontend, SQLite for login/signup database, Gemini API for chatbot, and OpenStreetMap for nearby centres.
Project StructurejavascriptEye_Diseases_Classification/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── gradcam.py
├── disease_info.py
├── eye_care.py
├── chatbot.py
├── templates/
│   ├── login.html
│   ├── signup.html
│   └── index.html
├── static/
├── models/
├── dataset/
├── requirements.txt
└── README.mdHow to Run
First clone the repo:javascriptgit https://github.com/Malika2244/EyeCare-AI/tree/main into project folder:javascriptcd Eye_Diseases_ClassificationMake virtual environment:javascriptpy -m venv venvActivate it:javascriptvenv\Scripts\activateInstall requirements:javascriptpip install -r requirements.txtRun the app:javascriptpy app.pyThen open browser and go to:javascripthttp://127.0.0.1:5000After that you can register and login to use the app. Make sure model files are in the models folder before running.
Note
This project is made for educational purpose only. It is not for medical diagnosis. For any real eye problem, please consult a proper eye specialist.
Conclusion
In this project I tried to combine Deep Learning and Web Development to make a simple system for eye disease classification and eye care awareness. I learned a lot while working on MobileNetV2, Flask and Grad-CAM.
