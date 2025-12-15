# app.py
import streamlit as st
import pandas as pd
import pickle

#  Load the trained pipeline
with open('churn_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction")
st.write("Fill in the customer details below to see if they are likely to leave the platform.")

#  Input form for features
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
country = st.text_input("Country", "USA")
city = st.text_input("City", "New York")
customer_segment = st.selectbox("Customer Segment", ["Individual", "SME", "Enterprise"])
tenure_months = st.number_input("Tenure Months", min_value=0, max_value=200, value=12)
signup_channel = st.selectbox("Signup Channel", ["Web", "Mobile", "Referral"])
contract_type = st.selectbox("Contract Type", ["Monthly", "Quarterly", "Yearly"])
monthly_logins = st.number_input("Monthly Logins", min_value=0, max_value=100, value=20)
weekly_active_days = st.number_input("Weekly Active Days", min_value=0, max_value=7, value=3)
avg_session_time = st.number_input("Average Session Time (minutes)", min_value=0.0, max_value=100.0, value=10.0)

#  Create DataFrame from user input
input_df = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'country': [country],
    'city': [city],
    'customer_segment': [customer_segment],
    'tenure_months': [tenure_months],
    'signup_channel': [signup_channel],
    'contract_type': [contract_type],
    'monthly_logins': [monthly_logins],
    'weekly_active_days': [weekly_active_days],
    'avg_session_time': [avg_session_time]
})

#  Prediction button
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]  # probability of churn

    if prediction == 1:
        st.error(f" This customer is likely to leave the platform! (Churn Probability: {probability:.2f})")
    else:
        st.success(f" This customer is likely to stay! (Churn Probability: {probability:.2f})")
