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

def clean2():
    '''
    Cleans apartments data: deelete rows with 'No_information values, replaces non numerical values into numerical, converts all columns into intergers, delete outlayers.
    @return data_apartments_clean (pd.DataFrame): Cleaned up Pandas data frame of apartments.
    '''
    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean.csv"
    
    data_apartments_clean = pd.read_csv(path_to_apart_csv)
    data_houses_clean = pd.read_csv(path_to_houses_csv)
    data_apartments_clean = data_apartments_clean[[
        'price', 'floor', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']]

    rows_with_no_information = data_apartments_clean.loc[(data_apartments_clean['floor'] == 'No_information') |
                                                         (data_apartments_clean['bedroomCount'] == 'No_information') |
                                                         (data_apartments_clean['bathroomCount'] == 'No_information') |
                                                         (data_apartments_clean['netHabitableSurface'] == 'No_information') |
                                                         (data_apartments_clean['condition'] == 'No_information')]

    rows_to_drop = data_apartments_clean.index.isin(
        rows_with_no_information.index)
    data_apartments_clean = data_apartments_clean[~rows_to_drop]

    data_apartments_clean['condition'].replace(
        {'To_be_done_up': 0, 'To_restore': 1, 'To_renovate': 2, 'Just_renovated': 3, 'Good': 4, "As_new": 5}, inplace=True)

    data_apartments_clean = data_apartments_clean.apply(
        pd.to_numeric, errors='coerce')
    data_apartments_clean[['price', 'floor', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']] = data_apartments_clean[[
        'price', 'floor', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']].astype('int64')

    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['floor'] < 17)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['bedroomCount'] < 7)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['bathroomCount'] <= 3)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['netHabitableSurface'] < 350)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['price'] < 2500000)]

    data_houses_clean = data_houses_clean[[
        'price', 'bedroomCount', 'bathroomCount', 'netHabitableSurface', 'condition']]

    rows_with_no_information_1 = data_houses_clean.loc[(data_houses_clean['bedroomCount'] == 'No_information') |
                                                       (data_houses_clean['bathroomCount'] == 'No_information') |
                                                       (data_houses_clean['netHabitableSurface'] == 'No_information') |
                                                       (data_houses_clean['condition'] == 'No_information')]

    rows_to_drop = data_houses_clean.index.isin(
        rows_with_no_information_1.index)
    data_houses_clean = data_houses_clean[~rows_to_drop]

    data_houses_clean['condition'].replace(
        {'To_be_done_up': 0, 'To_restore': 1, 'To_renovate': 2, 'Just_renovated': 3, 'Good': 4, "As_new": 5}, inplace=True)

    data_houses_clean = data_houses_clean.apply(pd.to_numeric, errors='coerce')
    data_houses_clean[['price', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']] = data_houses_clean[[
        'price', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']].astype('int64')

    data_houses_clean = data_houses_clean[(
        data_houses_clean['price'] < 3500000)]
    data_houses_clean = data_houses_clean[(
        data_houses_clean['bathroomCount'] <= 8)]
    data_houses_clean = data_houses_clean[(
        data_houses_clean['bedroomCount'] <= 12)]
    data_houses_clean = data_houses_clean[(
        data_houses_clean['netHabitableSurface'] < 700)]

    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean2.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean2.csv"
    
    data_apartments_clean.to_csv(path_to_apart_csv)
    data_houses_clean.to_csv(path_to_houses_csv)


with DAG(
    "cleaner2",
    default_args=default_args,
    description="Immo Eliza Pipeline",
    schedule=timedelta(days=1),
) as dag:
    
    
    cleaner2 = PythonOperator(task_id="apart_clean", python_callable=clean2)
    
    