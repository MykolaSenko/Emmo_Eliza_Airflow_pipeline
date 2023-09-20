from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from utils.id_scraper import id_scraper
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xg

default_args = {
    "owner": "flyingpig",
    "depends_on_past": False,
    "start_date": datetime(2023, 9, 17),
    "email": ["mykola.senko@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 5,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "scraper_id",
    default_args=default_args,
    description="Immo Eliza Pipeline",
    schedule=timedelta(days=1),
) as dag:
    scrape_id = PythonOperator(task_id="scrape_id", python_callable=id_scraper, op_args=[333])