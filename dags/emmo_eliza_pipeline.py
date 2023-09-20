from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from utils.functions  import opening_csv_cleaning, split_types, clean2, training, id_scraper, property_scraper, json_to_csv

# The `default_args` dictionary is used to define the default configuration settings for the DAG
# (Directed Acyclic Graph). These settings will be applied to all tasks within the DAG unless
# overridden by task-specific arguments.
default_args = {
    "owner": "flyingpig",
    "depends_on_past": False,
    # The line `    "start_date": datetime(2023, 9, 17),` is setting the start date for the DAG. It
    # specifies the date and time from which the DAG will start running. In this case, the DAG will
    # start running on September 17, 2023.
    "start_date": datetime(2023, 9, 17),
    "email": ["mykola.senko@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 5,
    # The line `    "retry_delay": timedelta(minutes=5),` is setting the delay between retries for a
    # task in the DAG.
    "retry_delay": timedelta(minutes=30),
}


# The `with DAG(...)` block is creating a Directed Acyclic Graph (DAG) object named "immo_eliza".
with DAG(
    "immo_eliza",
    default_args=default_args,
    description="Immo Eliza Pipeline",
    # The `schedule=timedelta(days=1)` parameter in the DAG constructor is setting the schedule
    # interval for the DAG. In this case, it is set to run once every day. This means that the DAG
    # will be triggered and executed automatically every day at the specified start time (defined in
    # `start_date`).
    schedule=timedelta(days=1),
) as dag:
    
    scrape_id = PythonOperator(task_id="scrape_id", python_callable=id_scraper, op_args=[333])
    
    scrape_property = PythonOperator(task_id="property_scraper", python_callable=property_scraper)
    
    save_csv = PythonOperator(task_id="json_to_csv", python_callable=json_to_csv)    

    cleaner = PythonOperator(task_id="cleaner", python_callable=opening_csv_cleaning)
    
    split = PythonOperator(task_id="split", python_callable=split_types)
    
    cleaner2 = PythonOperator(task_id="apart_clean", python_callable=clean2)
    
    train = PythonOperator(task_id="train", python_callable=training)

    scrape_id >> scrape_property >> save_csv >> cleaner >> split >> cleaner2 >> train
