import streamlit as st
import streamlit_utils.predict as sp 
import numpy as np

st.title ("Immo Eliza project")

st.markdown(
    """
    IMMO ELIZA is a project as part of the **AI Bootcamp** in [BeCode.org](https://becode.org/) Gent.

    The purpose of the project is to predict real estate prices in Belgium using machine learning models.
    To obtain a prediction for a specific property provide following information:
        "type of property" (House or Apartment)
        "floor" (if apartment)
        "amount of bedrooms"
        "habitable surface"
        "amount of bathrooms"
        "condition" (from 1 to 5) 
        
    You can access the project's repository [here](https://github.com/MykolaSenko/Emmo_Eliza_Airflow_pipeline).
        
    The project was carried out by Junior Data Engineer from [BeCode](https://becode.org) 
    Mykola Senko: [LinkedIn](https://www.linkedin.com/in/mykola-senko-683510a4), [GitHub](https://github.com/MykolaSenko)
    between August 28 and 31, 2023.

    Gent, Belgium
    
    September 21, 2023
"""
)

type = st.radio('Choose the type of your property',['House','Apartment']).lower() 
if type == 'apartment':
    floor = st.number_input('Pick a floor', 0)
else:
    pass
bedrooms = st.number_input('Pick a number of bedrooms', 0)
bathrooms = st.number_input('Pick a number of bathrooms', 0)
surface = st.slider('Input a habitable surface in m2', 0, 1000)
condition = st.select_slider('Select condition of a property', ['1', '2', '3', '4', '5'])
button = st.button('Predict your price')
if button == True:
    reg = sp.open_reg(type)
    if type == 'apartment':
        X = np.array([floor, bedrooms, surface, bathrooms, condition])
    else:
        X = np.array([bedrooms, surface, bathrooms, condition])
    
    y_pred = sp.predict_new(X, reg)
    
    st.write(f'Your predicted price is: {y_pred} euro for your {type}.')	
else:
    pass
