import streamlit as st
import joblib
import pandas as pd

# Load trained model
try:
    knn_model = joblib.load('knn_purchase_model.joblib')
except FileNotFoundError:
    st.error("Error: 'knn_purchase_model.joblib' not found.")
    st.stop()

# Load the scaler
try:
    scaler = joblib.load('feature_scaler.joblib')
except FileNotFoundError:
    st.error("Error: 'feature_scaler.joblib' not found.")
    st.stop()

# Load LabelEncoders
try:
    label_encoders = joblib.load('label_encoders.joblib')
except FileNotFoundError:
    st.error("Error: 'label_encoders.joblib' not found.")
    st.stop()

st.title("Purchase Prediction App")
st.write("Enter the details below:")

# Define feature categories
categorical_features = ["Gender"]
numerical_features = ["Age", "EstimatedSalary"]  # ✅ Ensure names match training data

# User input dictionary
input_data = {}

# Categorical inputs
for feature in categorical_features:
    unique_values = list(label_encoders[feature].classes_)  # Get categories from encoder
    input_data[feature] = st.selectbox(f"Select {feature}", unique_values)

# Numerical inputs
for feature in numerical_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict Purchase"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # 🔹 Ensure column names are correct
    input_df.rename(columns={"Estimated Salary": "EstimatedSalary"}, inplace=True)

    # 🔹 Encode categorical features (Fix Gender issue)
    for feature in categorical_features:
        if feature in input_df.columns:
            input_df[feature] = label_encoders[feature].transform(input_df[feature])
        else:
            st.error(f"Error: '{feature}' column is missing from input data.")
            st.stop()

    # 🔹 Ensure Gender is included in the DataFrame
    st.write("Final input DataFrame before scaling:", input_df)

    # 🔹 Scale numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # 🔹 Ensure all features used in training are present before prediction
    expected_features = categorical_features + numerical_features
    missing_features = [feat for feat in expected_features if feat not in input_df.columns]

    if missing_features:
        st.error(f"Missing features in input data: {missing_features}")
        st.stop()

    # 🔹 Make prediction
    prediction = knn_model.predict(input_df)

    # Display result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("Purchase is likely.")
    else:
        st.warning("Purchase is unlikely.")
