# Immo_Eliza_Airflow_pipeline
## Overview

This project builds a pipeline for scraping and predicting real estate data for Immo Eliza, a real estate agency in Belgium. The pipeline uses Apache Airflow to automate the collection, cleaning, transformation, and analysis of real estate data. The pipeline also trains a machine learning model to predict real estate prices, and serves the predictions to a Streamlit app for user interaction.
This is a [BeCode.org](https://becode.org/) project as part of the **AI Bootcamp** in Gent.
## Features

### Streamlit application
The ["Immo Eliza" Streamlit application](https://mykolasenko-immo-eliza-airflow-pipeline-stream-ion0fz.streamlit.app/) provides the following features:
Property Information Input: Users can input the following information about the property they want to predict the price for;
Type of property (House or Apartment);
Floor (if Apartment);
Number of bedrooms;
Number of bathrooms;
Habitable surface area;
Condition;
Price Prediction: Users can click the "Predict your price" button to obtain a price prediction for the specified property based on the input features.
Results Display: After clicking the prediction button, the application displays the predicted price for the user's property in euros.
### Apache Airflow pipeline
The Apache Airflow pipeline provides the following features:
Data Scraping: Automates the extraction of property data and related information from online sources;
Data Cleaning: Cleans and prepares the collected data for analysis and model training;
Data Transformation: Converts data into different formats for further analysis and visualization;
Machine Learning: Trains machine learning models to provide insights and predictions about the real estate market;
Git Integration: Automates the version control of the project with Git, allowing easy tracking of changes.
## Installation

To set up this project locally, follow the following installation instructions:
1. Clone [Immo_Eliza_Airflow_pipeline](https://github.com/MykolaSenko/Immo_Eliza_Airflow_pipeline) repository.
2. Change directory to the root of the repository.
3. Install required libraries by running `pip install -r requirements.txt`.
## Usage

### "Immo Eliza" Streamlit application
To use the "Immo Eliza" Streamlit application:
Go to the ["Immo Eliza" Streamlit application](https://mykolasenko-immo-eliza-airflow-pipeline-stream-ion0fz.streamlit.app/) or run the streamlit app locally after cloning and repository and installing the required libraries by `streamlit run streamlit.py` in your terminal.
Select the type of property (House or Apartment).
If you choose "Apartment," specify the floor.
Enter the number of bedrooms, bathrooms, habitable surface area, and rate the condition.
Click the "Predict your price" button to obtain the predicted price for your property.
### Apache Airflow pipeline
To use the Apache Airflow pipeline:
Configure the Airflow DAG to specify data sources, transformation steps, and ML tasks.
Schedule the DAG to run at specified intervals to keep the data up to date.
Monitor the pipeline's progress and view results in the Airflow UI.
Access the cleaned data and machine learning model predictions for real estate analysis.
## Timeline

This stage of the project lasted 4 days in the week of September 18-22, 2023.

## The Author

The project was made by Juniot Data engineer Mykola Senko: [LinkedIn](https://www.linkedin.com/in/mykola-senko-683510a4), [GitHub](https://github.com/MykolaSenko)

## Instruction

The project was made under the supervision of [Vanessa Rivera Quiñones](https://www.linkedin.com/in/vriveraq/) and [Samuel Borms](https://www.linkedin.com/in/sam-borms/?originalSubdomain=be)

## License
This project is under [GNU General Piblic License](./LICENSE) which allows to make modification to the code and use it for commercial purposes.