# Imports
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import numpy as np
from scipy.stats import skew
import time

# Utils functions
def recommend_courses(interests, career_goals, vectorizer, course_df, course_vectors):
    profile_text = interests + " " + career_goals
    profile_vector = vectorizer.transform([profile_text])
    similarity_scores = cosine_similarity(profile_vector, course_vectors)
    top_indices = similarity_scores.argsort()[0][-1:]
    recommended_courses = [course_df['Matched Courses'].iloc[index] for index in reversed(top_indices)]
    return recommended_courses

def make_prediction(model, scaler, program_name, department, gpa, gre_score, toefl_score):
    # Create a DataFrame from the inputs
    data = pd.DataFrame({
        'Program Name': [program_name],
        'Department': [department],
        'GPA': [gpa],
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score]
    })
    
    # Apply the same preprocessing as during training
    categorical_features = ['Program Name', 'Department']
    numerical_features = ['GPA', 'GRE Score', 'TOEFL Score']

    for col in numerical_features:
        if skew(data[col]) > 0 or skew(data[col]) < 0:
            data[col] = np.log1p(data[col])
    
    # Transform categorical data with OneHotEncoder
    data_encoded = pd.get_dummies(data, columns=categorical_features)
    training_columns = ['GPA', 'GRE Score', 'TOEFL Score',
       'Program Name_Chemical & Biochemical Engineering',
       'Program Name_Computer Engineering', 'Program Name_Computer Science',
       'Program Name_Cybersecurity', 'Program Name_Data Science',
       'Program Name_Electrical Engineering',
       'Program Name_Engineering Management',
       'Program Name_Environmental Engineering',
       'Program Name_Health Information Technology',
       'Program Name_Human-Centered Computing',
       'Department_Chemical & Biochemical Engineering',
       'Department_Civil & Environmental Engineering',
       'Department_Computer Science',
       'Department_Computer Science & Electrical Engineering',
       'Department_Electrical Engineering',
       'Department_Engineering Management', 'Department_Information Systems']
    
    data_encoded = data_encoded.reindex(columns=training_columns, fill_value=0)
    X_test_scaled = scaler.transform(data_encoded)

    prediction = model.predict(X_test_scaled)
    
    return prediction

def load_artifacts(model_path, scaler_path):
    model = load(model_path)
    scaler = load(scaler_path)
    return model, scaler

def load_model(path):
    """ Load the previously saved model from path. """
    return load(path)

# Load the course data and model
@st.cache_data
def load_data(courses_path, grades_path):
    # Courses
    courses_dataset = pd.read_csv(courses_path)
    unique_interests = courses_dataset['Interests'].unique().tolist()
    unique_career_goals = courses_dataset['Career Goals'].unique().tolist()
    vectorizer = TfidfVectorizer(stop_words='english')
    course_vectors = vectorizer.fit_transform(courses_dataset['Matched Courses'])

    # admission
    grades_dataset = pd.read_csv(grades_path)
    program_names = grades_dataset['Program Name'].unique().tolist()
    department_names = grades_dataset['Department'].unique().tolist()

    return courses_dataset, vectorizer, course_vectors, unique_interests, unique_career_goals, program_names, department_names


# Main
def main():
    st.title('Admission Acceptance')
    
    courses_path = r"C:\Users\sudhe\Projects\capstone\data\courses_dataset.csv"  # Update with your actual path
    grades_path = r"C:\Users\sudhe\Projects\capstone\data\admissions_acceptance_dataset.csv"
    model_path = r"C:\Users\sudhe\Projects\capstone\models\admission_ensemble.joblib"  # Update with your actual path
    scaler_path = r"C:\Users\sudhe\Projects\capstone\models\scaler.joblib"

    model, scaler = load_artifacts(model_path, scaler_path)

    courses_dataset, vectorizer, course_vectors, unique_interests, \
        unique_career_goals, program_names, department_names = load_data(courses_path, grades_path)
    
    # Example inputs
    program_name = st.selectbox("Select your Program", options=program_names)
    department = st.selectbox("Select your Department", options=department_names)
    gpa = st.number_input("Enter your GPA",min_value=0.0, max_value=4.0, value=3.5)
    gre_score = st.number_input("Enter your GRE score", max_value=340, value=315)
    toefl_score = st.number_input("Enter your TOEFL score", max_value=120, value=107)

    # Predict using the function
    if st.button('Admission Prediction'):
        with st.spinner('Prediction result'):
            time.sleep(3)
            pred = make_prediction(model, scaler, program_name, department, gpa, gre_score, toefl_score)
            if gre_score < 300:
                st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Not Admitted</div>", unsafe_allow_html=True)
            else:
                st.write("Prediction:", 'Admitted' if pred[0] == 1 else 'Not Admitted')
                if pred[0] == 1:
                    st.markdown(f"<div style='background-color:#4CAF50; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Admitted</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color:#F44336; color:white; padding:10px; border-radius:8px; font-weight:bold;'>Not Admitted</div>", unsafe_allow_html=True)

    # Dropdown for selecting interests and career goals
    st.title('Course Recommendation')
    interests = st.selectbox("Select your interests", options=unique_interests)
    career_goals = st.selectbox("Select your career goals", options=unique_career_goals)

    # Button to make prediction
    if st.button('Recommend Courses'):
        recommendations = recommend_courses(interests, career_goals, vectorizer, courses_dataset, course_vectors)
        st.write("Recommended Courses:")
        for course in recommendations:
            st.markdown(f"<div style='background-color:#ccffcc; font-size:20px; font-weight:bold; border-radius:5px; padding:10px;'>{course}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
