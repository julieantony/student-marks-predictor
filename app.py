import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="centered")

# App Header
st.title("🎓 Student Marks Prediction App")
st.write("Input your study hours per day to predict your exam marks using trained machine learning models.")

# Load the saved models
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('linear_model.pkl')
        knn_model = joblib.load('knn_model.pkl')
        return lr_model, knn_model
    except FileNotFoundError:
        return None, None

lr_model, knn_model = load_models()

if lr_model is None or knn_model is None:
    st.error("Model files ('linear_model.pkl' or 'knn_model.pkl') not found. Please upload them to the repository.")
else:
    # Sidebar / Selection Option
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox("Choose Model", ["Linear Regression", "KNN Regressor"])

    # Main User Input
    st.subheader("Enter Details:")
    study_hours = st.number_input(
        "Study Hours per day:", 
        min_value=0.0, 
        max_value=24.0, 
        value=5.0, 
        step=0.5
    )

    # Prediction Button
    if st.button("Predict Marks"):
        # Reshape to a 2D array since scikit-learn models expect [[feature]]
        input_features = np.array([[study_hours]])
        
        # Make predictions
        if model_choice == "Linear Regression":
            prediction = lr_model.predict(input_features)[0]
        else:
            prediction = knn_model.predict(input_features)[0]
        
        # Ensure marks don't go outside realistic boundaries (e.g., 0 to 100) if required
        prediction = max(0.0, min(100.0, float(prediction)))

        # Display Result
        st.success(f"### Predicted Marks: {prediction:.2f} / 100")
        st.info(f"Prediction generated using: **{model_choice}**")