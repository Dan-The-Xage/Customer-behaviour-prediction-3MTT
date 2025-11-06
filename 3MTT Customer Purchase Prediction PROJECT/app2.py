import streamlit as st
import joblib
import pandas as pd

# Load trained model and preprocessors
knn_model = joblib.load('knn_purchase_model.joblib')
scaler = joblib.load('feature_scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

st.title("Purchase Prediction App")

# Define feature names
categorical_features = ["Gender"]
numerical_features = ["Age", "EstimatedSalary"]

# Collect user input
input_data = {}

# Categorical input
for feature in categorical_features:
    unique_values = list(label_encoders[feature].classes_)  # Get categories from encoder
    input_data[feature] = st.selectbox(f"Select {feature}", unique_values)

# Numerical inputs
for feature in numerical_features:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button("Predict Purchase"):
    # Convert input into DataFrame
    input_df = pd.DataFrame([input_data])

    # 🚀 Debugging step
    st.write("🚀 Debug: Columns before encoding:", input_df.columns)

    # Encode categorical features
    for feature in categorical_features:
        if feature in input_df.columns:
            input_df[feature] = label_encoders[feature].transform(input_df[feature])
        else:
            st.error(f"🚨 Error: Missing column '{feature}' before encoding.")
            st.stop()

    # 🚀 Debugging step
    st.write("🚀 Debug: Columns after encoding:", input_df.columns)

    # Ensure all features exist before scaling
    expected_features = categorical_features + numerical_features
    missing_features = [feat for feat in expected_features if feat not in input_df.columns]

    if missing_features:
        st.error(f"🚨 Missing features before scaling: {missing_features}")
        st.stop()

    # Apply feature scaling
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # 🚀 Debugging step
    st.write("🚀 Debug: Final input before prediction:", input_df)

    # Ensure correct column order before prediction
    input_df = input_df[expected_features]

    # Make prediction
    prediction = knn_model.predict(input_df)

    # Display result
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("Purchase is likely.")
    else:
        st.warning("Purchase is unlikely.")
