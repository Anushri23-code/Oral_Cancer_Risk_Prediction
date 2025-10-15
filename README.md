Project Overview

This project is a web-based application that predicts the risk of oral cancer based on user inputs including personal, lifestyle, and oral health-related factors. The application uses a machine learning model (Logistic Regression) trained on a text and tabular dataset to provide risk prediction as Low, Medium, or High.

The system also maintains a history of all predictions for review and analysis.

Features

Collects user details:

Name, Age, Gender

Smoking and Alcohol habits

White patches in the mouth

HPV infection

Genetic history of cancer

Chronic irritation (e.g., dentures, sharp teeth)

Oral condition (good, moderate, poor)

Symptoms description (text input)

Preprocesses data automatically using a scikit-learn pipeline.

Extracts features from numeric, categorical, and textual input.

Predicts risk level using Logistic Regression.

Stores all predictions in data/predictions.csv.

Displays prediction result with risk level and confidence.

View history of all predictions via a dedicated page.

Project Structure
oral_cancer_risk_prediction/
│
├─ app.py                  # Flask web application
├─ train_model.py          # Script to train the ML pipeline (optional)
├─ model/
│   └─ pipeline.joblib     # Pre-trained ML pipeline
├─ templates/
│   ├─ index.html          # Input form page
│   ├─ result.html         # Prediction result page
│   └─ history.html        # Prediction history page
├─ data/
│   └─ predictions.csv     # Stores prediction history
└─ README.md

Installation

Clone the repository:

git clone <repository_url>
cd oral_cancer_risk_prediction


Create a virtual environment (optional but recommended):

python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Linux/Mac


Install required packages:

pip install -r requirements.txt


If requirements.txt does not exist, install manually:

pip install flask pandas scikit-learn joblib

Usage

Run the Flask app:

python app.py


Open your browser and go to:

http://127.0.0.1:5000/


Steps in the application:

Enter your details in the form.

Click Predict Risk.

View your risk level and confidence.

Optionally, click View Prediction History to see all previous predictions.

How It Works

Process Data: The app collects user input and converts it into a structured format (DataFrame).

Extract Features: The ML pipeline preprocesses numeric, categorical, and text features.

Predict Risk: Logistic Regression predicts the probability for low, medium, or high risk.

Store Result: Each prediction is saved in predictions.csv with timestamp and user details.

Generate & Display Result: Flask renders result.html to show the prediction to the user.

ML Model Used

Model: Logistic Regression (sklearn.linear_model.LogisticRegression)

Pipeline: Includes preprocessing (encoding, vectorization) + classifier.

Input Features: Combination of numeric, categorical, and text features.

Output: Predicted risk label and probability/confidence score.

Future Enhancements

Allow downloading prediction history as CSV or PDF.

Add user authentication for personalized history.

Improve ML model with larger datasets and more features.

Visualize predictions with graphs and statistics.

Author

Anu Shri

Mini Project: Oral Cancer Risk Prediction using Text Dataset