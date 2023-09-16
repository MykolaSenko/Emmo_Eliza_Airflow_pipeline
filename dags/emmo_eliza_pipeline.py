from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from utils.id_scraper import id_scraper
from utils.property_scraper import property_scraper
from utils.json_to_csv import json_to_csv
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
    "start_date": datetime(2023, 9, 16),
    "email": ["mykola.senko@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 5,
    "retry_delay": timedelta(minutes=5),
}


def main():
    search_pages_num = 333
    id_scraper(search_pages_num)
    property_scraper()
    json_to_csv()


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
    return data_prop

def split_types(data_prop):
    '''
    Splits data frame with properties into two data frames: apartments and houses.
    @return data_apartments (pd.DataFrame): Pandas data frame of apartments.
    @return data_houses(pd.DataFrame): Pandas data frame of houses.
    '''
    data_apartments = data_prop[data_prop['type'] == 'Apartment']
    data_houses = data_prop[data_prop['type'] == 'House']
    return data_apartments, data_houses

def clean2(data_apartments, data_houses):
    '''
    Cleans apartments data: deelete rows with 'No_information values, replaces non numerical values into numerical, converts all columns into intergers, delete outlayers.
    @return data_apartments_clean (pd.DataFrame): Cleaned up Pandas data frame of apartments.
    '''

    data_apartments_clean = data_apartments[[
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

    data_houses_clean = data_houses[[
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

    return data_houses_clean, data_apartments_clean

def training(data_houses_clean, data_apartments_clean):
    y_ap = data_apartments_clean['price'].values
    X_ap = data_apartments_clean[['floor', 'bedroomCount',
                     'netHabitableSurface', 'bathroomCount', 'condition']].values
    X_train_apart, X_test_apart, y_train_apart, y_test_apart = train_test_split(
        X_ap, y_ap, test_size=0.25, random_state=0)
    xgbr_apart = xg.XGBRegressor(objective='reg:squarederror',
                            n_estimators=10)
    xgbr_apart.fit(X_train_apart, y_train_apart)
    y_pred_apart = xgbr_apart.predict(X_test_apart)
    path_save_model_apart = Path.cwd() / "models" / "xgbr_apart.pickle"
    with open(path_save_model_apart, 'wb') as file:
        pickle.dump(xgbr_apart, file)
    mse_apart_xgbr = mean_squared_error(y_test_apart, y_pred_apart)
    # Compute mean absolute error between target test data and target predicted data for apartments.
    mae_apart_xgbr = mean_absolute_error(y_test_apart, y_pred_apart)
    y_h = data_houses_clean['price'].values
    X_h = data_houses_clean[['bedroomCount', 'bathroomCount',
                     'netHabitableSurface', 'condition']].values
    X_train_houses, X_test_houses, y_train_houses, y_test_houses = train_test_split(
        X_h, y_h, test_size=0.25, random_state=0)
    xgbr_houses = xg.XGBRegressor(objective='reg:squarederror',
                            n_estimators=10)
    xgbr_houses.fit(X_train_houses, y_train_houses)
    y_pred_houses = xgbr_houses.predict(X_test_houses)
    path_save_model_houses = Path.cwd() / "models" / "xgbr_houses.pickle"
    with open(path_save_model_houses, 'wb') as file:
        pickle.dump(xgbr_houses, file)
    mse_houses_xgbr = mean_squared_error(y_test_houses, y_pred_houses)
    # Compute mean absolute error between target test data and target predicted data for houses.
    mae_houses_xgbr = mean_absolute_error(y_test_houses, y_pred_houses)
    # save mse & mae for houses and apartments
    list_metrics = [mse_apart_xgbr, mae_apart_xgbr, mse_houses_xgbr, mae_houses_xgbr]
    path_metrics = Path.cwd() / "models" / "xgbr_metrics.txt"
    list_text = [
        "Mean Squared Error (MSE) for apartments training set (XGB Regressor):",
        "Mean Absolute Error (MAE) for apartments testing set (XGB Regressor):",
        "Mean Squared Error (MSE) for houses training set (XGB Regressor):",
        "Mean Absolute Error (MAE) for houses testing set (XGB Regressor):",
    ]
    with open(path_metrics, "w") as file:
        for metric, text in zip(list_metrics, list_text):
            metric_str = str(metric)
            file.write(text + " " + metric_str + "\n") # Add a newline character to separate lines
    
with DAG(
    "immo_eliza",
    default_args=default_args,
    description="Immo Eliza Pipeline",
    schedule_interval=timedelta(days=1),
) as dag:
    scraper = PythonOperator(task_id="scraper", python_callable=main)

    cleaner = PythonOperator(task_id="cleaner", python_callable=opening_csv_cleaning, provide_context=True)
    
    split = PythonOperator(task_id="split", python_callable=split_types, op_args=[cleaner.output], provide_context=True)
    
    cleaner2 = PythonOperator(task_id="apart_clean", python_callable=clean2, op_args=[split.output], provide_context=True)
    
    train = PythonOperator(task_id="train", python_callable=training, op_args=[cleaner2.output], provide_context=True)

    scraper >> cleaner >> split >> cleaner2 >> train
