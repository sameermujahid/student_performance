import streamlit as st
import pandas as pd
import joblib
import os

# Load the best model from the .pkl file
# Load the trained model using a relative path
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
model = joblib.load(model_path)

# Function to predict GPA and corresponding grade
def predict_gpa(input_data):
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Predict GPA using the loaded model
    predicted_gpa = model.predict(input_df)[0]  # Use 'model' here instead of 'best_model'

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

# Streamlit app
# Input fields for user data
age = st.number_input("Age", min_value=0, max_value=100, value=18)
gender = st.selectbox("Gender", options=["Male", "Female"], index=0)
ethnicity = st.selectbox("Ethnicity", options=["African American", "Asian", "Caucasian", "Other"], index=0)
parental_education = st.selectbox("Parental Education", options=["High School", "Some College", "Bachelor's", "Higher"], index=0)
study_time_weekly = st.number_input("Study Time Weekly (hours)", min_value=0.0, value=15.0, step=0.1)
absences = st.number_input("Absences", min_value=0, value=0)
tutoring = st.selectbox("Tutoring", options=["Yes", "No"], index=1)
parental_support = st.selectbox("Parental Support", options=["Low", "Moderate", "High", "Very High"], index=1)
club_involvement = st.selectbox("Club Involvement", options=["Yes", "No"], index=1)
sports = st.selectbox("Sports", options=["Yes", "No"], index=1)
music = st.selectbox("Music", options=["Yes", "No"], index=1)
volunteering = st.selectbox("Volunteering", options=["Yes", "No"], index=1)

# Convert categorical inputs to numeric
gender_map = {'Male': 0, 'Female': 1}
tutoring_map = {'No': 0, 'Yes': 1}
club_involvement_map = {'No': 0, 'Yes': 1}
sports_map = {'No': 0, 'Yes': 1}
music_map = {'No': 0, 'Yes': 1}
volunteering_map = {'No': 0, 'Yes': 1}
parental_support_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Very High': 4}
education_map = {'High School': 1, 'Some College': 2, "Bachelor's": 3, 'Higher': 4}
ethnicity_map = {'African American': 0, 'Asian': 1, 'Caucasian': 2, 'Other': 3}

# Prepare the input data for prediction
input_data = {
    'Age': age,
    'Gender': gender_map[gender],
    'Ethnicity': ethnicity_map[ethnicity],
    'ParentalEducation': education_map[parental_education],
    'StudyTimeWeekly': study_time_weekly,
    'Absences': absences,
    'Tutoring': tutoring_map[tutoring],
    'ParentalSupport': parental_support_map[parental_support],
    'ClubInvolvement': club_involvement_map[club_involvement],
    'Sports': sports_map[sports],
    'Music': music_map[music],
    'Volunteering': volunteering_map[volunteering]
}

# Button to predict GPA
if st.button("Predict GPA"):
    predicted_gpa, grade = predict_gpa(input_data)
    st.success(f'Predicted GPA: {predicted_gpa:.2f}, Grade: {grade}')
