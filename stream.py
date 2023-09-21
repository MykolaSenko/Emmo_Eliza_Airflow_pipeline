import streamlit as st
import streamlit_utils.predict as sp
import numpy as np

st.title("Immo Eliza project")

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

# The line `type = st.radio("Choose the type of your property", ["House", "Apartment"]).lower()` is
# creating a radio button input for the user to choose the type of their property. The options are
# "House" and "Apartment". The `.lower()` method is used to convert the user's input to lowercase.
# This is done to ensure consistency in the input, as the rest of the code expects the type to be in
# lowercase.
type = st.radio("Choose the type of your property", ["House", "Apartment"]).lower()
# The line `if type == "apartment":` is checking if the user has selected "Apartment" as the type of
# their property. If the condition is true, it means that the user has selected "Apartment", and the
# code inside the if block will be executed. If the condition is false, it means that the user has
# selected "House" or any other option, and the code inside the if block will be skipped.
if type == "apartment":
    floor = st.number_input("Pick a floor", 0)
else:
    pass
# `bedrooms = st.number_input("Pick a number of bedrooms", 0)` is creating a number input field for
# the user to enter the number of bedrooms in their property.
bedrooms = st.number_input("Pick a number of bedrooms", 0)
# `bathrooms = st.number_input("Pick a number of bathrooms", 0)` is creating a number input field for
# the user to enter the number of bathrooms in their property. The input field allows the user to
# select a number from 0 to any positive integer. The default value is set to 0. The value entered by
# the user will be stored in the variable `bathrooms` for further use in the code.
bathrooms = st.number_input("Pick a number of bathrooms", 0)
# The line `surface = st.slider("Input a habitable surface in m2", 0, 1000)` is creating a slider
# input field for the user to enter the habitable surface area of their property in square meters.
surface = st.slider("Input a habitable surface in m2", 0, 1000)
# The line `condition = st.select_slider("Select condition of a property", ["1", "2", "3", "4", "5"])`
# is creating a slider input field for the user to select the condition of their property. The slider
# has 5 options: "1", "2", "3", "4", and "5". The selected value will be stored in the variable
# `condition` for further use in the code.
condition = st.select_slider(
    "Select condition of a property", ["1", "2", "3", "4", "5"]
)
button = st.button("Predict your price")
# This code block is responsible for predicting the price of the user's property based on the input
# provided.
if button == True:
    # The line `reg = sp.open_reg(type)` is calling the `open_reg()` function from the
    # `streamlit_utils.predict` module and assigning the returned value to the variable `reg`.
    reg = sp.open_reg(type)
    # This code block is creating an input array `X` based on the user's input values.
    if type == "apartment":
        X = np.array([floor, bedrooms, surface, bathrooms, condition])
    else:
        X = np.array([bedrooms, surface, bathrooms, condition])

    # `y_pred = sp.predict_new(X, reg)` is calling the `predict_new()` function from the
    # `streamlit_utils.predict` module and passing the input array `X` and the trained machine
    # learning model `reg` as arguments.
    y_pred = sp.predict_new(X, reg)

    st.write(f"Your predicted price is: {y_pred} euro for your {type}.")
else:
    pass
