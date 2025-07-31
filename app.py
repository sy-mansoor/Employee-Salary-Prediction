# Filename: app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load The Trained Model and Data ---

# Load the trained Random Forest model
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'random_forest_model.pkl' is in the same directory.")
    st.stop()

# Load the columns from the training phase
try:
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("Model columns file not found. Please ensure 'model_columns.pkl' is in the same directory.")
    st.stop()


# --- 2. Define the Application Interface ---

st.set_page_config(page_title="Employee Salary Prediction", layout="wide")

# App title
st.title('💰 Employee Salary Prediction')
st.markdown("""
This app predicts whether an employee's salary is greater than $50K or not, based on their demographic and employment data.
Please provide the employee's details in the sidebar.
""")

# --- 3. Create Input Fields in the Sidebar ---

st.sidebar.header('Employee Input Features')

# Create a function to gather user input
def user_input_features():
    age = st.sidebar.slider('Age', 17, 90, 35)
    workclass = st.sidebar.selectbox('Work Class', ('Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay'))
    fnlwgt = st.sidebar.number_input('Final Weight (fnlwgt)', 12285, 1484705, 178356)
    education_num = st.sidebar.slider('Education Level (Num)', 1, 16, 10)
    marital_status = st.sidebar.selectbox('Marital Status', ('Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'))
    occupation = st.sidebar.selectbox('Occupation', ('Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces'))
    relationship = st.sidebar.selectbox('Relationship', ('Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'))
    race = st.sidebar.selectbox('Race', ('White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'))
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    capital_gain = st.sidebar.number_input('Capital Gain', 0, 99999, 0)
    capital_loss = st.sidebar.number_input('Capital Loss', 0, 4356, 0)
    hours_per_week = st.sidebar.slider('Hours per Week', 1, 99, 40)
    native_country = st.sidebar.selectbox('Native Country', ('United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Japan', 'Poland', 'Columbia', 'Taiwan', 'Haiti', 'Iran', 'Portugal', 'Nicaragua', 'Peru', 'France', 'Greece', 'Ecuador', 'Ireland', 'Hong', 'Trinadad&Tobago', 'Cambodia', 'Laos', 'Thailand', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland'))

    # Create a dictionary of the input data
    data = {
        'age': age,
        'fnlwgt': fnlwgt,
        'education-num': education_num,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'workclass': workclass,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'sex': sex,
        'native-country': native_country
    }
    
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()


# --- 4. Preprocess User Input and Make Prediction ---

# Display the user input
st.subheader('User Input')
st.write(input_df)

# One-hot encode the user input
# We need to make sure the input has the same columns as the training data
input_processed = pd.get_dummies(input_df)

# Align the columns of the input with the model's columns
# This adds missing columns with a value of 0
input_aligned = input_processed.reindex(columns=model_columns, fill_value=0)

# Ensure the order of columns is the same as during training
input_aligned = input_aligned[model_columns]

# Create a predict button
if st.button('**Predict Salary**'):
    # Make a prediction
    prediction = model.predict(input_aligned)
    
    # Display the prediction
    st.subheader('Prediction Result')
    
    if prediction[0] > 0.5: # We use 0.5 as the threshold
        st.success('**The predicted salary is MORE than $50K.** 🎉')
    else:
        st.info('**The predicted salary is LESS than or equal to $50K.** 💼')


st.markdown("""
---
### How does this work?
This application uses a **Random Forest** machine learning model that was trained on the [Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult). 
The model learned patterns from nearly 30,000 employee records to make its prediction.
""") 

st.markdown("""
---
### Author.
Syed Mansoor Ahmed.[Github](https://github.com/sy-mansoor). 

""")