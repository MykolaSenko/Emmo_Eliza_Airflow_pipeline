from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle
import requests
from concurrent.futures import ThreadPoolExecutor
import time
import itertools
from pathlib import Path
import json
from bs4 import BeautifulSoup
import re
import signal
import sys
import os

def opening_csv_cleaning():
    """
    The function `opening_csv_cleaning()` reads a CSV file, cleans the data by removing rows with
    missing values, capitalizes certain string values, removes specific patterns from a column, removes
    duplicate rows, replaces values in a column, fills missing values with a specific value, changes the
    data type of a column, and saves the cleaned data to a new CSV file.
    """
    path_to_data_csv = Path.cwd() / "data" / "properties_data.csv"
    data_prop = pd.read_csv(path_to_data_csv)
    # The above code is using the `replace` method to replace any empty or whitespace-only strings in
    # the `data_prop` variable with `NaN` (Not a Number). The regular expression `r"^s*$"` is used to
    # match empty or whitespace-only strings.
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


def split_types():
    """
    Splits data frame with properties into two data frames: apartments and houses.
    @return data_apartments (pd.DataFrame): Pandas data frame of apartments.
    @return data_houses(pd.DataFrame): Pandas data frame of houses.
    """
    path_to_data_csv = Path.cwd() / "data" / "properties_data_clean.csv"
    data_prop = pd.read_csv(path_to_data_csv)
    data_apartments = data_prop[data_prop["type"] == "Apartment"]
    data_houses = data_prop[data_prop["type"] == "House"]
    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean.csv"

    data_apartments.to_csv(path_to_apart_csv)
    data_houses.to_csv(path_to_houses_csv)


def clean2():
    """
    Cleans apartments data: deelete rows with 'No_information values, replaces non numerical values into numerical, converts all columns into intergers, delete outlayers.
    @return data_apartments_clean (pd.DataFrame): Cleaned up Pandas data frame of apartments.
    """
    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean.csv"

    data_apartments_clean = pd.read_csv(path_to_apart_csv)
    data_houses_clean = pd.read_csv(path_to_houses_csv)
    data_apartments_clean = data_apartments_clean[
        [
            "price",
            "floor",
            "bedroomCount",
            "netHabitableSurface",
            "bathroomCount",
            "condition",
        ]
    ]

    rows_with_no_information = data_apartments_clean.loc[
        (data_apartments_clean["floor"] == "No_information")
        | (data_apartments_clean["bedroomCount"] == "No_information")
        | (data_apartments_clean["bathroomCount"] == "No_information")
        | (data_apartments_clean["netHabitableSurface"] == "No_information")
        | (data_apartments_clean["condition"] == "No_information")
    ]

    rows_to_drop = data_apartments_clean.index.isin(rows_with_no_information.index)
    data_apartments_clean = data_apartments_clean[~rows_to_drop]

    data_apartments_clean["condition"].replace(
        {
            "To_be_done_up": 0,
            "To_restore": 1,
            "To_renovate": 2,
            "Just_renovated": 3,
            "Good": 4,
            "As_new": 5,
        },
        inplace=True,
    )

    data_apartments_clean = data_apartments_clean.apply(pd.to_numeric, errors="coerce")
    data_apartments_clean[
        [
            "price",
            "floor",
            "bedroomCount",
            "netHabitableSurface",
            "bathroomCount",
            "condition",
        ]
    ] = data_apartments_clean[
        [
            "price",
            "floor",
            "bedroomCount",
            "netHabitableSurface",
            "bathroomCount",
            "condition",
        ]
    ].astype(
        "int64"
    )

    data_apartments_clean = data_apartments_clean[(data_apartments_clean["floor"] < 17)]
    data_apartments_clean = data_apartments_clean[
        (data_apartments_clean["bedroomCount"] < 7)
    ]
    data_apartments_clean = data_apartments_clean[
        (data_apartments_clean["bathroomCount"] <= 3)
    ]
    data_apartments_clean = data_apartments_clean[
        (data_apartments_clean["netHabitableSurface"] < 350)
    ]
    data_apartments_clean = data_apartments_clean[
        (data_apartments_clean["price"] < 2500000)
    ]

    data_houses_clean = data_houses_clean[
        ["price", "bedroomCount", "bathroomCount", "netHabitableSurface", "condition"]
    ]

    rows_with_no_information_1 = data_houses_clean.loc[
        (data_houses_clean["bedroomCount"] == "No_information")
        | (data_houses_clean["bathroomCount"] == "No_information")
        | (data_houses_clean["netHabitableSurface"] == "No_information")
        | (data_houses_clean["condition"] == "No_information")
    ]

    rows_to_drop = data_houses_clean.index.isin(rows_with_no_information_1.index)
    data_houses_clean = data_houses_clean[~rows_to_drop]

    data_houses_clean["condition"].replace(
        {
            "To_be_done_up": 0,
            "To_restore": 1,
            "To_renovate": 2,
            "Just_renovated": 3,
            "Good": 4,
            "As_new": 5,
        },
        inplace=True,
    )

    data_houses_clean = data_houses_clean.apply(pd.to_numeric, errors="coerce")
    data_houses_clean[
        ["price", "bedroomCount", "netHabitableSurface", "bathroomCount", "condition"]
    ] = data_houses_clean[
        ["price", "bedroomCount", "netHabitableSurface", "bathroomCount", "condition"]
    ].astype(
        "int64"
    )

    data_houses_clean = data_houses_clean[(data_houses_clean["price"] < 3500000)]
    data_houses_clean = data_houses_clean[(data_houses_clean["bathroomCount"] <= 8)]
    data_houses_clean = data_houses_clean[(data_houses_clean["bedroomCount"] <= 12)]
    data_houses_clean = data_houses_clean[
        (data_houses_clean["netHabitableSurface"] < 700)
    ]

    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean2.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean2.csv"

    data_apartments_clean.to_csv(path_to_apart_csv)
    data_houses_clean.to_csv(path_to_houses_csv)


def training():
    """
    The `training` function trains XGBoost regression models on apartment and house data, saves the
    models, and computes and saves the mean squared error (MSE) and mean absolute error (MAE) metrics
    for both datasets.
    """
    path_to_apart_csv = Path.cwd() / "data" / "apart_data_clean2.csv"
    path_to_houses_csv = Path.cwd() / "data" / "houses_data_clean2.csv"
    data_apartments_clean = pd.read_csv(path_to_apart_csv)
    data_houses_clean = pd.read_csv(path_to_houses_csv)

    y_ap = data_apartments_clean["price"].values
    X_ap = data_apartments_clean[
        ["floor", "bedroomCount", "netHabitableSurface", "bathroomCount", "condition"]
    ].values
    X_train_apart, X_test_apart, y_train_apart, y_test_apart = train_test_split(
        X_ap, y_ap, test_size=0.25, random_state=0
    )
    xgbr_apart = xg.XGBRegressor(objective="reg:squarederror", n_estimators=10)
    xgbr_apart.fit(X_train_apart, y_train_apart)
    y_pred_apart = xgbr_apart.predict(X_test_apart)
    path_save_model_apart = Path.cwd() / "models" / "xgbr_apart.pickle"
    with open(path_save_model_apart, "wb") as file:
        pickle.dump(xgbr_apart, file)
    mse_apart_xgbr = mean_squared_error(y_test_apart, y_pred_apart)
    # Compute mean absolute error between target test data and target predicted data for apartments.
    mae_apart_xgbr = mean_absolute_error(y_test_apart, y_pred_apart)
    y_h = data_houses_clean["price"].values
    X_h = data_houses_clean[
        ["bedroomCount", "bathroomCount", "netHabitableSurface", "condition"]
    ].values
    X_train_houses, X_test_houses, y_train_houses, y_test_houses = train_test_split(
        X_h, y_h, test_size=0.25, random_state=0
    )
    xgbr_houses = xg.XGBRegressor(objective="reg:squarederror", n_estimators=10)
    xgbr_houses.fit(X_train_houses, y_train_houses)
    y_pred_houses = xgbr_houses.predict(X_test_houses)
    path_save_model_houses = Path.cwd() / "models" / "xgbr_houses.pickle"
    with open(path_save_model_houses, "wb") as file:
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
            # Add a newline character to separate lines
            file.write(text + " " + metric_str + "\n")


def get_ids_from_page(page, property_types, session):
    """
    Get property ids from the search-results endpoint for a specific page for the given page and property types.

    @param page (int): page number to retrieve ids from.
    @param property_types (list): list of property types to search for.
    @param session (requests.Session()): requests session object for making http requests.
    @return (list): list of property ids from the page.
    """
    ids = []
    for prop_type in property_types:
        url_lists = (
            "https://www.immoweb.be/en/search-results/%s/for-sale?countries=BE&page=%s&orderBy=newest"
            % (prop_type, page)
        )
        r = session.get(url_lists)
        for listing in r.json()["results"]:
            ids.append(listing["id"])
    return ids


def get_ids(pages):
    """
    Get property ids from multiple pages using multithreading.

    @param pages (int): number of pages to scrape property ids from.
    @return (set): set of unique property ids.
    """
    if pages > 333:
        pages = 333
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            # the lambda function passes the page number, the list of property type ['house', 'apartment'] and the session object as arguments
            # executor.map() applies the function in parallel for each page number in the range from 1 to pages
            result = executor.map(
                lambda page: get_ids_from_page(page, ["house", "apartment"], session),
                range(1, pages + 1),
            )
            # flatten the result, which is a generator of lists, into a single list
            # itertools.chain.from_iterable() is used to concatenate all the nested lists into one iterable
            flattened_result = list(itertools.chain.from_iterable(result))
            print(f"Number of ids: {len(flattened_result)}")
            return set(flattened_result)


def save_to_txt(ids):
    """
    Save property ids to a text file.

    @param ids (list): list of property ids to save.
    """
    file_name = "properties_ids.txt"
    file_path = Path.cwd() / "data" / file_name
    with open(file_path, "w") as f:
        for id in ids:
            f.write("%s\n" % id)


def id_scraper(pages):
    """
    Main function to scrape property ids and save them into a text file.

    @param pages (int): number of pages to scrape.
    """
    start = time.time()
    ids = get_ids(pages)
    save_to_txt(ids)
    end = time.time()
    print("Time taken to scrape ids: {:.6f}s".format(end - start))


def json_to_csv():
    """
    Convert json file into a csv file.
    """
    path_to_open = Path.cwd() / "data" / "properties_data.json"
    path_to_save = Path.cwd() / "data" / "properties_data.csv"

    with open(path_to_open, "r", encoding="utf-8") as file:
        data = json.load(file)

    df = pd.DataFrame.from_dict(data, orient="index")
    df.to_csv(path_to_save, index_label="id", encoding="utf-8")
    print(df.head())


def get_js_data(js_data, property_data):
    """
    Extracts relevant property information from the javascript data.

    @param js_data (dict): javascript data containing property information.
    @param property_data (dict): dictionary with initial data to which the property information will be added.
    @return (dict): dictionary containing property information.
    """
    # get price
    property_data["transactionType"] = js_data["transaction"]["type"]
    property_data["transactionSubtype"] = js_data["transaction"]["subtype"]
    if js_data["transaction"]["sale"] != None:
        property_data["price"] = js_data["transaction"]["sale"]["price"]
    elif js_data["transaction"]["rental"] != None:
        property_data["price"] = js_data["transaction"]["rental"]["price"]
    else:
        property_data["price"] = None
    # get property data
    property = [
        "type",
        "subtype",
        "location",
        "bedroomCount",
        "netHabitableSurface",
        "building",
        "hasLift",
        "kitchen",
        "hasGarden",
        "gardenSurface",
        "hasTerrace",
        "terraceSurface",
        "land",
        "fireplaceExists",
        "hasSwimmingPool",
        "hasAirConditioning",
        "bathroomCount",
        "showerRoomCount",
        "toiletCount",
        "parkingCountIndoor",
        "parkingCountOutdoor",
        "parkingCountClosedBox",
    ]
    for prop in property:
        if prop == "location":
            loc = [
                "country",
                "region",
                "province",
                "district",
                "locality",
                "postalCode",
                "street",
                "number",
                "box",
                "floor",
                "latitude",
                "longitude",
            ]
            for l in loc:
                property_data[l] = js_data["property"][prop][l]
        elif prop == "building":
            sub = ["constructionYear", "facadeCount", "floorCount", "condition"]
            for s in sub:
                if js_data["property"][prop] != None:
                    property_data[s] = js_data["property"][prop][s]
                else:
                    property_data[s] = None
        elif prop == "kitchen":
            if js_data["property"][prop] != None:
                property_data[prop] = js_data["property"][prop]["type"]
            else:
                property_data[prop] = None
        elif prop == "land":
            if js_data["property"][prop] != None:
                property_data[prop] = js_data["property"][prop]["surface"]
            else:
                property_data[prop] = None
        else:
            property_data[prop] = js_data["property"][prop]
    # get energy consumption data
    if js_data["transaction"]["certificates"] != None:
        property_data["primaryEnergyConsumptionPerSqm"] = js_data["transaction"][
            "certificates"
        ]["primaryEnergyConsumptionPerSqm"]
        property_data["epcScore"] = js_data["transaction"]["certificates"]["epcScore"]
    else:
        property_data["primaryEnergyConsumptionPerSqm"] = None
        property_data["epcScore"] = None
    if js_data["property"]["energy"] != None:
        property_data["hasDoubleGlazing"] = js_data["property"]["energy"][
            "hasDoubleGlazing"
        ]
    else:
        property_data["hasDoubleGlazing"] = None
    # get sale type
    sale_type = None
    if js_data["flags"]["isPublicSale"]:
        sale_type = "PublicSale"
    elif js_data["flags"]["isNotarySale"]:
        sale_type = "NotarySale"
    elif js_data["flags"]["isLifeAnnuitySale"]:
        sale_type = "LifeAnnuitySale"
    elif js_data["flags"]["isAnInteractiveSale"]:
        sale_type = "AnInteractiveSale"
    elif js_data["flags"]["isInvestmentProject"]:
        sale_type = "InvestmentProject"
    elif js_data["flags"]["isNewRealEstateProject"]:
        sale_type = "NewRealEstateProject"
    property_data["saleType"] = sale_type
    # publication date
    property_data["creationDate"] = None
    property_data["lastModificationDate"] = None
    if js_data["publication"] != None:
        property_data["creationDate"] = js_data["publication"]["creationDate"]
        property_data["lastModificationDate"] = js_data["publication"][
            "lastModificationDate"
        ]

    return property_data


def get_page_data(id, session):
    """
    Scrape property information from a specific property listing page.

    @param id (str): id of the property.
    @param session (requests.Session()): requests session object for making http requests.
    @return (dict): dictionary containing property information.
    """
    url = "https://www.immoweb.be/en/classified/" + id
    property_data = {id: {}}
    property_data[id]["URL"] = url

    req = session.get(url)
    status = req.status_code

    # The above code is checking the status of a request. If the status is not equal to 200, it
    # updates the "Status" value in the property_data dictionary for a specific id. If the status is
    # 200, it also updates the "Status" value and then proceeds to extract data from the HTML content
    # of the request. It uses BeautifulSoup to parse the HTML and finds all script tags with the type
    # "text/javascript". It then searches for a specific JavaScript variable called
    # "window.classified" within each script tag. If found, it extracts the value of this variable,
    # which is in JSON
    if status != 200:
        property_data[id]["Status"] = status
    else:
        property_data[id]["Status"] = status
        content = req.content
        s = BeautifulSoup(content, "html.parser")

        script_tags = s.find_all("script", {"type": "text/javascript"})
        for st in script_tags:
            if st.text.find("window.classified") != -1:
                js_var = re.search(r"window\.classified = (\{.*\});", st.text)
                js_var_value = js_var.group(1)
                js_data = json.loads(js_var_value)
                property_data[id] = get_js_data(js_data, property_data[id])
                break

    return property_data


def scrape_from_txt():
    """
    Scrape property data from multiple pages ids listed in a text file using multithreading.

    @return (dict): dictionary containing property data scraped from multiple property listings.
    """
    file_name = "properties_ids.txt"
    file_path = Path.cwd() / "data" / file_name
    property_data = {}
    # The above code is reading a file and using the contents of the file to make parallel requests to
    # retrieve data from multiple web pages.
    with open(file_path, "r") as file:
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=10) as executor:
                # the lambda function passes the id and the session as arguments to the get_page_data() function; the returned dict is then added to the property_data dict
                # executor.map() applies the function in parallel for each id in the file
                executor.map(lambda id: property_data.update(get_page_data(id, session)),
                    (id.strip() for id in file))
    return property_data


def save_to_json(data):
    """
    Save scraped property data into a json file.

    @param data (dict): dictionary containing property data.
    """
    file_name = "properties_data.json"
    file_path = Path.cwd() / "data" / file_name
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def sigterm_handler(signum, frame):
    """
    The function `sigterm_handler` is a signal handler that is responsible for cleaning up the
    task's resources before exiting.
        
    :param signum: The signum parameter represents the signal number that caused the handler to be
    called. In this case, it is used to handle the SIGTERM signal, which is typically sent to a
    process to request its termination
    :param frame: The frame parameter is a reference to the current execution frame. It provides
    information about the current state of the program, such as the current line of code being
    executed and the values of local variables
    """
    # Clean up the task's resources here
    sys.exit(0)


def property_scraper():
    """
    Main function to scrape property data and save it into a json file.
    """
    signal.signal(signal.SIGTERM, sigterm_handler)
    
    start = time.time()
    immo_data = scrape_from_txt()
    save_to_json(immo_data)
    end = time.time()
    print("Time taken to scrape listings: {:.6f}s".format(end - start))


def run_git_command(command):
    """
    The function `run_git_command` changes the working directory to a specified Git repository and runs
    a given Git command.
    
    :param command: The `command` parameter is a string that represents the Git command you want to run.
    For example, it could be "git add ." to add all changes to the staging area, or "git commit -m
    'Initial commit'" to make a commit with a specific message
    """
    git_repo_directory = '/home/flyingpig/codes/becode_projects/Emmo_Eliza_Airflow_pipeline'
    os.chdir(git_repo_directory)

    # Run the git add command
    os.system(command)