import streamlit as st
import pandas as pd
import joblib
import numpy as np

import joblib
import os

# Load the trained model using a relative path
model_path = os.path.join(os.path.dirname(__file__), 'best_gpa_prediction_model.pkl')
model = joblib.load(model_path)

# Define the input form
st.title('GPA Prediction App')

st.write("Please enter the following details:")

# Input fields
age = st.number_input("Age", min_value=15, max_value=18, value=17)
gender = st.selectbox("Gender", ['Female', 'Male'])
ethnicity = st.selectbox("Ethnicity", ['Caucasian', 'Asian', 'African American', 'Other'])
parental_education = st.selectbox("Parental Education", ['Some College', 'High School', "Bachelor's", 'Higher', 'None'])
study_time_weekly = st.number_input("Study Time Weekly (hours)", min_value=0.0, max_value=40.0, value=15.0)
absences = st.number_input("Number of Absences", min_value=0, max_value=40, value=0)
tutoring = st.selectbox("Tutoring", ['Yes', 'No'])
parental_support = st.selectbox("Parental Support", ['Moderate', 'Low', 'High', 'Very High', 'None'])
extracurricular = st.selectbox("Extracurricular Activities", ['No', 'Yes'])
sports = st.selectbox("Sports Participation", ['No', 'Yes'])
music = st.selectbox("Music Participation", ['Yes', 'No'])
volunteering = st.selectbox("Volunteering", ['No', 'Yes'])

# Prepare input data for prediction
input_data = {
    'Age': age,
    'Gender': gender,
    'Ethnicity': ethnicity,
    'ParentalEducation': parental_education,
    'StudyTimeWeekly': study_time_weekly,
    'Absences': absences,
    'Tutoring': tutoring,
    'ParentalSupport': parental_support,
    'Extracurricular': extracurricular,
    'Sports': sports,
    'Music': music,
    'Volunteering': volunteering
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Make predictions
predicted_gpa = model.predict(input_df)[0]  # Extract the first prediction

# Function to determine grade based on GPA
def determine_grade(gpa):
    if gpa >= 3.5:
        return 'A'
    elif 3.0 <= gpa < 3.5:
        return 'B'
    elif 2.5 <= gpa < 3.0:
        return 'C'
    elif 2.0 <= gpa < 2.5:
        return 'D'
    else:
        return 'F'

# Display the prediction
if st.button('Predict GPA'):
    grade = determine_grade(predicted_gpa)  # Get the grade for the predicted GPA
    st.success(f"The predicted GPA for the entered details is: {predicted_gpa:.2f}, which corresponds to a grade of: {grade}.")
