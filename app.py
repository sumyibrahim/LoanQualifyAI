import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
with open('loanqualifyai.pkl', 'rb') as file:
    best_pipeline = pickle.load(file)

# Define expected feature names (must match training data)
feature_names = ['Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
 'Gender_Female', 'Gender_Male', 'Property_Area_Rural',
 'Property_Area_Semiurban', 'Property_Area_Urban']

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")

# User Inputs
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.number_input("Dependents", min_value=0, max_value=5, step=1)
education = st.selectbox("Education", ["Not Graduate", "Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount ", min_value=0)
loan_term = st.selectbox("Loan Term (in months)", [360, 240, 180, 120, 60])
credit_history = st.selectbox("Credit History", [0, 1])

# One-Hot Encoding for Gender
gender = st.radio("Gender", ["Male", "Female"])
gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

# One-Hot Encoding for Property Area
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
area_rural = 1 if property_area == "Rural" else 0
area_semiurban = 1 if property_area == "Semiurban" else 0
area_urban = 1 if property_area == "Urban" else 0

# Convert user inputs into model-friendly format
test_input = np.array([[
    1 if married == "Yes" else 0,  # Married
    dependents,  # Dependents
    1 if education == "Graduate" else 0,  # Education
    1 if self_employed == "Yes" else 0,  # Self Employed
    applicant_income,  # Applicant Income
    coapplicant_income,  # Coapplicant Income
    loan_amount,  # Loan Amount
    loan_term,  # Loan Term
    credit_history,  # Credit History
    gender_female,  # Gender_Female
    gender_male,  # Gender_Male
    area_rural,  # Area_Rural
    area_semiurban,  # Area_Semiurban
    area_urban  # Area_Urban
]])

# Convert to DataFrame
test_df = pd.DataFrame(test_input, columns=feature_names)



# Predict Button
if st.button("🔍 Check Loan Eligibility"):
    prediction = best_pipeline.predict(test_df)
    prediction_proba = best_pipeline.predict_proba(test_df)

    approval_prob = prediction_proba[0][1] * 100
    rejection_prob = prediction_proba[0][0] * 100

    if prediction[0] == 1:
        st.success(f"✅ Loan is Approved")
    else:
        st.error(f"❌ Loan is Rejected")

    st.write(f"📊 **Approval Probability:** {approval_prob:.2f}%")
    st.write(f"📉 **Rejection Probability:** {rejection_prob:.2f}%")

