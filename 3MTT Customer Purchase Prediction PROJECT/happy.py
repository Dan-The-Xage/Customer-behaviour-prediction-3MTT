import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Import just in case

# --- Load Artifacts ---
try:
    best_model = joblib.load('best_purchase_model6.joblib')
    scaler = joblib.load('feature_scaler6.joblib')
    label_encoders = joblib.load('label_encoders6.joblib')
except FileNotFoundError as e:
    st.error(f"Error loading file: {e.filename} not found. Ensure all necessary files are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred during loading: {e}")
    st.stop()

st.title("Purchase Prediction App")
st.write("Enter the features to predict the purchase outcome.")

# --- Create Input Fields Based on Features ---
input_data = {}

# Categorical features input
for feature, encoder in label_encoders.items():
    unique_categories = list(encoder.classes_)
    input_data[feature] = st.selectbox(f"Select {feature}", unique_categories)

# Numerical features input (assuming you know these based on your data)
numerical_features_trained = [col for col in scaler.feature_names_in_] # Get numerical features scaled
for feature in numerical_features_trained:
    if feature not in input_data:  # Avoid duplicates if a numerical feature was also object (unlikely after encoding)
        input_data[feature] = st.number_input(f"Enter {feature}")

if st.button("Predict Purchase"):
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # --- Preprocess Input Data ---
    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = encoder.transform(input_df[col])
            except Exception as e:
                st.error(f"Error during encoding of '{col}': {e}")
                st.stop()

    # Scale numerical features
    numerical_cols_to_scale = [col for col in input_df.columns if col in numerical_features_trained]
    if numerical_cols_to_scale:
        try:
            input_df[numerical_cols_to_scale] = scaler.transform(input_df[numerical_cols_to_scale])
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()

# --- Reorder Columns ---
    # Get the original column order from your training script's output
    original_column_order = ['Age', 'EstimatedSalary', 'Gender'] # 
    input_df = input_df[original_column_order]

    # --- Make Prediction ---
    try:
        prediction = best_model.predict(input_df)
        st.subheader("Prediction:")
        if prediction[0] == 1:  # Assuming 1 represents 'Purchase'
            st.success("The model predicts a purchase.")
        else:
            st.warning("The model predicts no purchase.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()