import json
import logging
import numpy as np
import os
import pandas as pd
import requests
import sys
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

CONF_FILE = ".vscode/settings.json"
logger.info("Getting few important dependencies...")
with open(CONF_FILE, "r") as file:
    configur = json.load(file)

logger.info("Defining paths...")

DATA_DIR = get_project_dir(configur["general"]["data_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, configur["train"]["table_name"])
INFERENCE_PATH = os.path.join(DATA_DIR, configur["inference"]["inf_table_name"])

RANDOM_STATE = configur["general"]["random_state"]
TEST_SIZE = configur["train"]["test_size"]

class DataGetting():
    def __init__(self):
        self.df = None

    def get_iris(self):
        logger.info("Stealing dataset from wikipedia...")
        url = 'https://en.wikipedia.org/wiki/Iris_flower_data_set'
        response = requests.get(url)
        if response.status_code == 200:
          soup = BeautifulSoup(response.content, 'html.parser')
          table = soup.find('table', {'class':'wikitable'})
          data = []
          for row in table.find_all('tr')[1:]:
            columns = row.find_all(['th', 'td'])
            data.append([column.get_text(strip = True) for column in columns])
          df = pd.DataFrame(data, columns = ["id","Sepal length", "Sepal width", "Petal length", "Petal width", "Species"])
          df.set_index('id', inplace=True)
        else:
          print(f"Failed to get data from page. Status code: {response.status_code}")
        raw_file_path = os.path.join(DATA_DIR, 'raw_data.csv')
        logger.info("Hiding in discrete place...")
        df.to_csv(raw_file_path, index = False, encoding="utf-16")
        self.df = df
        return self.df

    def split_and_save(self, train_dir, inference_dir):
        logger.info("Separating dataset...")
        data = self.df
        target_col = "Species"
        X = data.drop(columns = target_col)
        y = pd.DataFrame(data[target_col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, stratify = y, random_state = RANDOM_STATE)
        train_df = pd.concat([X_train, y_train], axis = 1)
        inf_df = pd.concat([X_test, y_test], axis = 1)
        train_df.to_csv(train_dir, index = False, encoding="utf-16")
        inf_df.to_csv(inference_dir, index = False, encoding="utf-16")

if __name__ == "__main__":
    configure_logging()
    data_getter = DataGetting()
    data_getter.get_iris()
    data_getter.split_and_save(TRAIN_PATH, INFERENCE_PATH)
    logger.info("Great success! Script finished.")
    
