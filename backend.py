import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
pd.set_option('future.no_silent_downcasting',True)


class preprocessing():

    def __init__(self):
        pass
    
    def list_to_dataframe(features_list):
        values = features_list
        keys = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
       'ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
       'Property_Area', 'CoapplicantIncome']
        
        dictionary = {}

        for i in range(len(values)):
            dictionary[keys[i]] = values[i]
        df = pd.DataFrame(dictionary,index=[0])
        return df

    def categorical_to_numerical(dataset):

        streamlit_input = replace_dict = {
            "Married": {"No":0,"Yes":1},
            "Gender": {"Male":1, "Female":0},
            "Dependents": {"0":0, "1":1, "2":2, "3+":3},
            "Education": {"Graduate":1, "Not Graduate":0},
            "Self_Employed": {"No":0 , "Yes":1},
            "Property_Area": {"Rural":0, "Semiurban":1, "Urban":2},
            "Credit_History": {"True":1, "False":0}
        }    
        for column, replacements in streamlit_input.items():
             dataset[column] = dataset[column].replace(replacements)


        dataset = dataset.apply(pd.to_numeric, errors = "coerce")
        return dataset 
    
    def column_addition(dataset):

        try:

            dataset["ApplicantIncome"] = dataset["ApplicantIncome"] + dataset["CoapplicantIncome"]
            dataset = dataset.drop("CoapplicantIncome",axis=1) 

            return dataset
        except Exception as e:
            print(f"error:{e}")

    def data_standardization(data):
        if data.isnull().values.any():
            raise ValueError("Data contains missing values")      

        if not np.issubdtype(data.values.dtype,np.number):
            raise ValueError("Wrong Data Type")
        
        scaler = StandardScaler()
        result = scaler.fit_transform(data.values.reshape(-1,1))
        
        return result
    
    def model_deserialization(model, dataset):

        checker = dataset.reshape(1,-1)

        with open(model,"rb") as f:
            predictor = pickle.load(f)


        result = predictor.predict(checker)
        return result
    
    

