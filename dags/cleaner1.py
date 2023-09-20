from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle


default_args = {
    "owner": "flyingpig",
    "depends_on_past": False,
    "start_date": datetime(2023, 9, 17),
    "email": ["mykola.senko@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 5,
    "retry_delay": timedelta(minutes=5),
}

def opening_csv_cleaning():
    path_to_data_csv = Path.cwd() / "data" / "properties_data.csv"
    data_prop = pd.read_csv(path_to_data_csv)
    data_prop = data_prop.replace(r"^s*$", float("NaN"), regex=True)
    # delete  rows where 'price' value are NaN
    data_prop.dropna(subset=["price"], inplace=True)
    # delete rows where 'region' value is NaN
    data_prop.dropna(subset=["region"], inplace=True)
    data_prop[
        [
            "transactionType",
            "transactionSubtype",
            "type",
            "subtype",
            "locality",
            "street",
            "condition",
            "kitchen",
        ]
    ] = data_prop[
        [
            "transactionType",
            "transactionSubtype",
            "type",
            "subtype",
            "locality",
            "street",
            "condition",
            "kitchen",
        ]
    ].applymap(
        lambda x: str(x).capitalize()
    )  # change string values which were written in upper case into capitilized.
    data_prop["type"] = data_prop["type"].replace(
        r"_group$", "", regex=True
    )  # remove all "_group" in the "type" column
    data_prop = data_prop.drop_duplicates(
        subset=[
            "latitude",
            "longitude",
            "type",
            "subtype",
            "price",
            "district",
            "locality",
            "street",
            "number",
            "box",
            "floor",
            "netHabitableSurface",
        ],
        keep="first",
    )  # Cleaning from duplicates
    data_prop["kitchen"] = data_prop.kitchen.replace(
        [
            "Usa_installed",
            "Installed",
            "Semi_equipped",
            "Hyper_equipped",
            "Usa_hyper_equipped",
            "Usa_semi_equipped",
        ],
        "Installed",
    )  # Clearing kitchen column
    data_prop["kitchen"] = data_prop.kitchen.replace(
        ["Usa_uninstalled", "Not_installed"], "Not_installed"
    )
    # Changing all 'NaN' values to 'No_information' values
    data_prop = data_prop.fillna(value="No_information")
    # Changing all str 'Nan', 'NaN' values to 'No_information'
    data_prop = data_prop.replace(["Nan", "NaN"], "No_information")
    # Changing type of postal code data into integer
    data_prop["postalCode"] = data_prop["postalCode"].astype("int64")
    # Cleaning Energy Class data
    data_prop["epcScore"] = data_prop.epcScore.replace(
        ["G_A++", "C_B", "G_F", "G_A", "E_A", "D_C"], "No_information"
    )
    # Save clean data to CSV file
    path_to_save_csv = Path.cwd() / "data" / "properties_data_clean.csv"
    data_prop.to_csv(path_to_save_csv)


    
with DAG(
    "cleaner1",
    default_args=default_args,
    description="Immo Eliza Pipeline",
    schedule=timedelta(days=1),
) as dag:

    cleaner = PythonOperator(task_id="cleaner", python_callable=opening_csv_cleaning)
