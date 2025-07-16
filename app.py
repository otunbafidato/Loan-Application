import streamlit as st
from backend import preprocessing as pp


st.title("Loan Predictor App")


#Widget Features
Gender = st.selectbox("Gender",["Male","Female"])
Married = st.selectbox("Marriage Status", ["No", "Yes"])
Dependents = st.selectbox("Number of Dependent", ["0","1","2","+3"])
Education = st.selectbox("Education Status", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Employment Status", ["True", "Flase"])
Credit_History = st.selectbox("Credict History", ["True", "False"])
Property_Area = st.selectbox("Property_Area", ["Rural","Semiurban","Urban"])
applicant = st.number_input("ApplicantIncome",help="Enter your annual income in dollars",format="%.f")
coapplicant = st.number_input("CoapplicantIncome",help="Enter your annual income in dollars",format="%.f")
Loan_amount = st.number_input("LoanAmount",help="Enter your annual income in dollars",format="%.f")
Loan_amount_term = st.number_input("Loan_Amount_term",help="Enter your annual income in dollars",format="%.f")

Features = [Gender , Married, Dependents, Education, Self_Employed, Credit_History, Property_Area, applicant, coapplicant, Loan_amount, Loan_amount_term]

# Data Processing
dataset = pp.list_to_dataframe(Features)
print(dataset)

stage2 = pp.categorical_to_numerical(dataset)

temporary_fix = stage2.fillna(0)

stage3 = pp.column_addition(temporary_fix)

stage4 = pp.data_standardization(stage3)

stage5 = pp.model_deserialization("predictor.pkl",stage4)
print(stage5)

#button

if st.button("Predict loan Eligibility"):

    if stage5 == 1:
        st.success("Loan Approved")

    else:
        st.error("Loan Disapproved")
            