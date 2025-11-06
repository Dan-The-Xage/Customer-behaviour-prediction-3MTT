import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
try:
    knn_model = joblib.load('knn_purchase_model.joblib')
except FileNotFoundError:
    st.error("Error: 'knn_purchase_model.joblib' not found. Ensure it's in the correct directory.")
    st.stop()

# Load the scaler
try:
    scaler = joblib.load('feature_scaler.joblib')
except FileNotFoundError:
    st.error("Error: 'feature_scaler.joblib' not found. Ensure it's in the correct directory.")
    st.stop()

# Load the LabelEncoders
try:
    label_encoders = joblib.load('label_encoders.joblib')  # Ensure correct filename
    if not isinstance(label_encoders, dict):  # Check if it's properly stored as a dictionary
        raise ValueError("Error: 'label_encoders.joblib' is not a dictionary. Please check the saved file.")
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

# Streamlit UI
st.title("Purchase Prediction App")
st.write("Enter the features below to predict whether a purchase will occur.")

# Define feature categories
categorical_features = ["Gender"]  # Categorical input
numerical_features = ["Age", "Estimated Salary"]  # Numerical inputs

# Collect user inputs
input_data = {}

# Gender selection
gender_options = ["Male", "Female"]
input_data["Gender"] = st.selectbox("Select Gender", gender_options)

# Numerical inputs
input_data["Age"] = st.number_input("Enter Age", min_value=0, max_value=100, value=25)
input_data["Estimated Salary"] = st.number_input("Enter Estimated Salary", min_value=0, value=50000)

if st.button("Predict Purchase"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert categorical input explicitly to string before transformation
    input_df["Gender"] = input_df["Gender"].astype(str)

    # Encode categorical feature (Gender)
    input_df["Gender"] = input_df["Gender"].astype(str)

# Transform using LabelEncoder (after verifying it expects strings)
if "Gender" in label_encoders:
    input_df["Gender"] = label_encoders["Gender"].transform([input_df["Gender"].iloc[0]])
else:
    st.error("LabelEncoder for 'Gender' not found!")
    st.stop()
    # if "Gender" in label_encoders:
        # input_df["Gender"] = label_encoders["Gender"].transform([input_df["Gender"][0]])
    # else:
        # st.error("LabelEncoder for 'Gender' not found!")
        # st.stop()

    # Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Make prediction
    prediction = knn_model.predict(input_df)

    # Display result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("The model predicts a purchase will likely occur.")
    else:
        st.warning("The model predicts a purchase is unlikely.")
