# Immo_Eliza_Airflow_pipeline
Building pipeline for scraping and predicting data for Immo Eliza 

Overview
Immo Eliza Pipeline is an Apache Airflow-based data pipeline designed to automate the collection, cleaning, transformation, and analysis of real estate data. This pipeline fetches property information from various sources, processes the data, and performs machine learning tasks to provide insights and predictions related to the real estate market.

Features
Data Scraping: Automates the extraction of property data and related information from online sources.
Data Cleaning: Cleans and prepares the collected data for analysis and model training.
Data Transformation: Converts data into different formats for further analysis and visualization.
Machine Learning: Trains machine learning models to provide insights and predictions about the real estate market.
Git Integration: Automates the version control of the project with Git, allowing easy tracking of changes.
Installation
To set up this project locally, follow the installation instructions in the Installation Guide.

Usage
Configure the Airflow DAG to specify data sources, transformation steps, and ML tasks.
Schedule the DAG to run at specified intervals to keep the data up to date.
Monitor the pipeline's progress and view results in the Airflow UI.
Access the cleaned data and machine learning model predictions for real estate analysis.
Dependencies
Python
Apache Airflow
Additional Python libraries (specified in requirements.txt)
Git (for version control)
