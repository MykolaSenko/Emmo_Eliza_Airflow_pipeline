from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


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

def split_types():
    '''
    Splits data frame with properties into two data frames: apartments and houses.
    @return data_apartments (pd.DataFrame): Pandas data frame of apartments.
    @return data_houses(pd.DataFrame): Pandas data frame of houses.
    '''
    path_to_data_csv = Path.cwd() / "data" / "properties_data_clean.csv"
    data_prop = pd.read_csv(path_to_data_csv)
    data_apartments = data_prop[data_prop['type'] == 'Apartment']
    data_houses = data_prop[data_prop['type'] == 'House']
    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean.csv"
    
    data_apartments.to_csv(path_to_apart_csv)
    data_houses.to_csv(path_to_houses_csv)
    

with DAG(
    "split",
    default_args=default_args,
    description="Immo Eliza Pipeline",
    schedule=timedelta(days=1),
) as dag:
    
    split = PythonOperator(task_id="split", python_callable=split_types)
