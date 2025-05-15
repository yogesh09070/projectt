# AI-Powered Medical Diagnosis System

## Project Overview
This project is an AI-powered medical diagnosis system designed to predict the likelihood of three critical diseases:
- *Heart Disease*
- *Diabetes*
- *Parkinson's Disease*

The system is built using *Machine Learning models* trained on respective datasets and deployed as a *Streamlit web application*. The user provides necessary medical inputs, and the trained models predict the likelihood of the disease.

## Internship Information
This project is developed as part of an *internship at Edunet Foundation, under the mentorship of **Saomya Chaudhury sir*.

## Features
- *Heart Disease Prediction* (13 Features)
- *Diabetes Prediction* (8 Features)
- *Parkinson's Disease Prediction* (22 Features)
- *User-friendly Streamlit UI* for seamless interaction
- *Model Loading and Input Validation*
- *Real-time Prediction Display*

## Technologies Used
- *Python*
- *Streamlit* (for UI)
- *Scikit-learn* (for model training)
- *Pandas & NumPy* (for data handling)
- *Joblib* (for model serialization)

## Installation & Setup
### Prerequisites
Ensure you have Python installed (Recommended: Python 3.8+)

### Steps to Run the Application

Run the Streamlit app:

   streamlit run app.py
   

## Usage
1. Select the disease you want to predict.
2. Enter the required medical parameters.
3. Click on *Predict* to get the result.

## Model Details
### Heart Disease Model
- *Dataset:* Heart Disease Dataset
- *Features:* 13 clinical parameters
- *Algorithm Used:* Logistic Regression / Random Forest

### Diabetes Model
- *Dataset:* PIMA Indian Diabetes Dataset
- *Features:* 8 clinical parameters
- *Algorithm Used:* Decision Tree / SVM

### Parkinson’s Disease Model
- *Dataset:* UCI Parkinson’s Disease Dataset
- *Features:* 22 voice-related parameters
- *Algorithm Used:* Random Forest / SVM

## Future Improvements
- Expand to include more diseases.
- Improve model accuracy with additional datasets.
- Integrate a database for storing patient history.
- Deploy as a web application with Flask/Django.

## Contributors
- *Santhosh S*  (Meenakshi Sundararajan Engineering College)

## Acknowledgment
This project was completed under the *Edunet Foundation Internship Program, with guidance from **Saomya Chaudhury sir*.

## License
This project is open-source under the MIT License.
