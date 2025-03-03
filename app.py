import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Set paths
MODEL_PATH = os.path.join('models', 'tree_model.pkl')
ENCODERS_PATH = os.path.join('models', 'label_encoders.pkl')

# Set page configuration
st.set_page_config(page_title="Fastag Fraud Detection", layout="wide")

# Add CSS styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .fraud { color: red; font-weight: bold; }
    .not-fraud { color: green; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Load the trained model and label encoders
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return model, encoders
    except FileNotFoundError:
        st.error("Model files not found. Please ensure the models are saved in the 'models' directory.")
        return None, None

model, encoders = load_models()

# Title and description
st.title("🚗 Fastag Fraud Detection System")
st.markdown("### Enter transaction details to detect potential fraud")

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    vehicle_type = st.selectbox("Vehicle Type", ['Bus', 'Car', 'Motorcycle', 'Truck', 'Van', 'Sedan', 'SUV'])
    lane_type = st.selectbox("Lane Type", ['Express', 'Regular'])
    vehicle_dimensions = st.selectbox("Vehicle Dimensions", ['Large', 'Small', 'Medium'])

with col2:
    transaction_amount = st.number_input("Transaction Amount (₹)", min_value=0, value=100)
    amount_paid = st.number_input("Amount Paid (₹)", min_value=0, value=100)
    state_code = st.selectbox("State Code", ['KA', 'MH', 'TN', 'AP', 'DL', 'UP', 'GA'])

# Create input DataFrame
def create_input_data():
    current_time = datetime.now()
    return pd.DataFrame({
        'Vehicle_Type': [vehicle_type],
        'Lane_Type': [lane_type],
        'Vehicle_Dimensions': [vehicle_dimensions],
        'Transaction_Amount': [transaction_amount],
        'Amount_paid': [amount_paid],
        'Fraud_indicator': [0],
        'state_code': [state_code],
    })

# Predict button
if st.button("Detect Fraud", type="primary"):
    if model is not None and encoders is not None:
        with st.spinner("Analyzing transaction..."):
            try:
                # Prepare input data
                input_data = create_input_data()
                
                # Encode categorical variables
                for column in input_data.columns:
                    if column in encoders:
                        input_data[column] = encoders[column].transform(input_data[column])

                # Make prediction
                prediction = model.predict(input_data)[0]
                
                # Display result
                st.markdown("### Analysis Result")
                if prediction == 1:  # Fraud
                    st.error("🚨 Potential Fraud Detected!")
                    st.markdown("""
                        #### Suspicious patterns identified:
                        - Mismatch between transaction amount and amount paid
                        - Unusual transaction pattern for this vehicle type
                    """)
                else:
                    st.success("✅ Transaction appears legitimate")
                    st.markdown("#### No suspicious patterns detected")

                # Display transaction summary
                st.markdown("### Transaction Summary")
                summary_df = pd.DataFrame({
                    'Feature': ['Vehicle Type', 'Transaction Amount', 'Amount Paid', 'State'],
                    'Value': [vehicle_type, f'₹{transaction_amount}', f'₹{amount_paid}', state_code]
                })
                st.table(summary_df)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model or encoders not loaded properly. Please check the model files.")

# Footer
st.markdown("---")
st.markdown("### About")
st.markdown("""
    This fraud detection system uses machine learning to identify potentially fraudulent Fastag transactions.
    The model analyzes various transaction parameters to make predictions.
""")