import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('./Customer_Churn_Model.sav', 'rb'))

def churn_prediction(input_data, lr_model, rf_model, xgb_model):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    lr_prediction = lr_model.predict(input_data_reshaped)
    rf_prediction = rf_model.predict(input_data_reshaped)
    xgb_prediction = xgb_model.predict(input_data_reshaped)

    return lr_prediction, rf_prediction, xgb_prediction

def main():
    # giving a title
    st.title('Customer Churn Prediction App')
    
    # Getting the input data from the user 
    Credit_Score = st.text_input('Credit Score')
    Geography = st.text_input('Geography [France - 0, Germany - 1, Spain - 2]')
    Gender = st.text_input('Gender [Female - 0, Male - 1]')
    Age = st.text_input('Age')
    Tenure = st.text_input('Tenure')
    Balance = st.text_input('Balance')
    Number_Of_Products = st.text_input('Number Of Products')
    Has_Credit_Card = st.text_input('Has Credit Card')
    Is_Active_Member = st.text_input('Is Active Member')
    Estimated_Salary = st.text_input('Estimated Salary')

    # Convert input to appropriate data types
    def convert_to_float(val):
        try:
            return float(val)
        except ValueError:
            return None

    def convert_to_int(val):
        try:
            return int(val)
        except ValueError:
            return None

    # Validate and convert input data types
    Credit_Score = convert_to_int(Credit_Score)
    Geography = convert_to_int(Geography)
    Gender = convert_to_int(Gender)
    Age = convert_to_int(Age)
    Tenure = convert_to_int(Tenure)
    Balance = convert_to_float(Balance)
    Number_Of_Products = convert_to_int(Number_Of_Products)
    Has_Credit_Card = convert_to_int(Has_Credit_Card)
    Is_Active_Member = convert_to_int(Is_Active_Member)
    Estimated_Salary = convert_to_float(Estimated_Salary)

    # creating a button for Prediction
    if st.button('Predict'):
        lr_prediction, rf_prediction, xgb_prediction = churn_prediction(
            [Credit_Score, Geography, Gender, Age, Tenure, Balance, Number_Of_Products, Has_Credit_Card, Is_Active_Member, Estimated_Salary],
            loaded_model[0], loaded_model[1], loaded_model[2]
        )

        st.subheader('Customer Churn Predictions')

        if(lr_prediction[0] == 0):
            st.success(f"Logistic Regression Prediction: Customer will not leave!")
        else:
            st.success(f"Logistic Regression Prediction: Customer will leave!")

        if(rf_prediction[0] == 0):
            st.success(f"Random Forest Prediction: Customer will not leave!")
        else:
            st.success(f"Random Forest Prediction: Customer will leave!")

        if(xgb_prediction[0] == 0):
            st.success(f"Xtreme Gradient Boosting Prediction: Customer not will leave!")
        else:
            st.success(f"Xtreme Gradient Boosting Prediction: Customer will leave!")
                

if __name__ == '__main__':
    main()
