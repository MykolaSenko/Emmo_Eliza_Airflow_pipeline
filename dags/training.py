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

def training():
    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean2.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean2.csv"
    data_apartments_clean = pd.read_csv(path_to_apart_csv)
    data_houses_clean = pd.read_csv(path_to_houses_csv)
    
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
    "training",
    default_args=default_args,
    description="Immo Eliza Pipeline",
    schedule=timedelta(days=1),
) as dag:
        
    train = PythonOperator(task_id="train", python_callable=training)
    
train