import streamlit as st
import pandas as pd
import joblib

import joblib
import os

# Load the trained model using a relative path
model_path = os.path.join(os.path.dirname(__file__), 'ml_model.pkl')
best_model = joblib.load(model_path)


# Function to predict GPA and corresponding grade
def predict_gpa(input_data):
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Predict GPA using the loaded model
    predicted_gpa = best_model.predict(input_df)[0]

    # Determine the grade based on GPA
    if predicted_gpa >= 3.5:
        grade = 'A'
    elif predicted_gpa >= 3.0:
        grade = 'B'
    elif predicted_gpa >= 2.5:
        grade = 'C'
    elif predicted_gpa >= 2.0:
        grade = 'D'
    else:
        grade = 'F'

    return predicted_gpa, grade


# Streamlit UI
st.title("GPA Prediction Application")

# Input fields
age = st.number_input("Age:", min_value=0, max_value=100, value=18)
gender = st.selectbox("Gender:", options=["Male", "Female"], index=0)
ethnicity = st.selectbox("Ethnicity:", options=[0, 1, 2, 3, 4], index=0)
parental_education = st.selectbox("Parental Education Level:", options=[0, 1, 2, 3], index=1)
study_time_weekly = st.number_input("Study Time (Weekly, hours):", min_value=0, value=20)
absences = st.number_input("Number of Absences:", min_value=0, value=0)
tutoring = st.selectbox("Tutoring:", options=["No", "Yes"], index=0)
parental_support = st.selectbox("Parental Support Level:", options=[0, 1, 2, 3, 4], index=3)
club_involvement = st.selectbox("Club Involvement:", options=["No", "Yes"], index=0)
sports = st.selectbox("Involved in Sports:", options=["No", "Yes"], index=0)
music = st.selectbox("Involved in Music:", options=["No", "Yes"], index=0)
volunteering = st.selectbox("Involved in Volunteering:", options=["No", "Yes"], index=0)

# Map string inputs to numerical values
gender = 0 if gender == "Male" else 1
tutoring = 1 if tutoring == "Yes" else 0
club_involvement = 1 if club_involvement == "Yes" else 0
sports = 1 if sports == "Yes" else 0
music = 1 if music == "Yes" else 0
volunteering = 1 if volunteering == "Yes" else 0

# Button for prediction
if st.button("Predict GPA"):
    input_data = {
        'Age': age,
        'Gender': gender,
        'Ethnicity': ethnicity,
        'ParentalEducation': parental_education,
        'StudyTimeWeekly': study_time_weekly,
        'Absences': absences,
        'Tutoring': tutoring,
        'ParentalSupport': parental_support,
        'ClubInvolvement': club_involvement,
        'Sports': sports,
        'Music': music,
        'Volunteering': volunteering
    }

    predicted_gpa, grade = predict_gpa(input_data)
    st.success(f'Predicted GPA: {predicted_gpa:.2f}, Grade: {grade}')
