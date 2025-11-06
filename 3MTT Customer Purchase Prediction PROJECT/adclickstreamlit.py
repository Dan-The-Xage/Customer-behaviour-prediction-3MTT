import streamlit as st
import joblib
import pandas as pd

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
    le = joblib.load('label_encoders1.joblib')
except FileNotFoundError:
    st.error("Error: 'label_encoders.joblib' not found. Ensure it's in the correct directory.")
    st.stop()

st.title("Purchase Prediction App")
st.write("Enter the features below to predict whether a purchase will occur.")

# Define feature categories
categorical_features = list(le.keys())
numerical_features = ['Age', 'Estimated Salary']  # Update with actual numerical features

# Collect user inputs
input_data = {}

# Categorical inputs
for feature in categorical_features:
    unique_values = list(le[feature].classes_)
    input_data[feature] = st.selectbox(f"Select {feature}", unique_values)

# Numerical inputs
for feature in numerical_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict Purchase"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for feature in categorical_features:
        input_df[feature] = le[feature].transform([input_df[feature][0]])

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