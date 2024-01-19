
import hashlib
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle
import requests
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from datetime import datetime
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

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
MODEL_PATH = os.path.join(DATA_DIR, configur["general"]["models_dir"])

RANDOM_STATE = configur["general"]["random_state"]
DATETIME_FORMAT = configur["general"]["datetime_format"]
TARGET_COL = configur["general"]["target_col"]
HIDDEN_SIZE = configur["train"]["hiiden_size"]
MAX_EPOCHS = configur["train"]["max_epochs"]
BATCH_SIZE = configur["train"]["batch_size"]
DICT_NAME = configur["general"]["dict_name"]
TEST_SIZE = configur["train"]["test_size"]


class TrainProcessor():
    def __init__(self):
        pass

    def prepare_data(self, train_path):
        logging.info("Loading data...")
        df = pd.read_csv(train_path, encoding="utf-16")
        df = self.encode_target(df)
        X_train, X_test, y_train, y_test, input_size = self.split_train(df)
        X_train_tens, y_train_tens, X_val_tens, y_val_tens = self.convert_to_tensors(X_train, X_test, y_train, y_test)
        train_loader, val_loader = self.prepare_dataloader(X_train_tens, y_train_tens, X_val_tens, y_val_tens, BATCH_SIZE)
        return train_loader, val_loader, input_size

    def encode_target(self, df):
        logging.info("Preparing data...")
        labels = list(range(len(df[TARGET_COL].unique())))
        #label encoding target column
        mapping = dict(zip(df[TARGET_COL].unique(), labels))
        inv_map = {v: k for k, v in mapping.items()}
        logging.info("Saving decoder...")
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        dict_path = os.path.join(MODEL_PATH, DICT_NAME)
        np.save(dict_path, inv_map)
        df[TARGET_COL] = pd.Series(df[TARGET_COL]).map(mapping)
        return df

    def split_train(self, data : pd.DataFrame, test_size: float = TEST_SIZE, target_col = 'Species'):
        logging.info("Splitting data...")
        target_col = target_col
        X = data.drop(columns = target_col).values
        y = (data[target_col]).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = y, random_state = RANDOM_STATE)
        input_size = X_train.shape[1]
        return X_train, X_test, y_train, y_test, input_size

    def convert_to_tensors(self, X_train, X_test, y_train, y_test):
        train_tensors = (torch.tensor(X_train, dtype = torch.float32),
                        torch.tensor(y_train, dtype = torch.long))
        val_tensors = (torch.tensor(X_test, dtype = torch.float32),
                        torch.tensor(y_test, dtype = torch.long))
        return *train_tensors, *val_tensors

    def prepare_dataloader(self, X_train_tens, y_train_tens, X_val_tens, y_val_tens, batch_size):
        train_dataset = TensorDataset(X_train_tens, y_train_tens)
        val_dataset = TensorDataset(X_val_tens, y_val_tens)
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, num_workers=1)
        logging.info("Preparation successful.")
        return train_loader, val_loader


class IrisNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 3)
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.F1Score(task='multiclass', num_classes = 3)
        self.val_metrics = torchmetrics.F1Score(task='multiclass', num_classes = 3)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
       inputs, target = batch
       outputs = self(inputs)
       loss = self.criterion(outputs, target)

       f1 = self.train_metrics(outputs, target)
       self.log('train_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
       self.log('train_f1_score', f1, on_step = True, on_epoch = True, prog_bar = True, logger = True)
       return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, target)

        f1 = self.train_metrics(outputs, target)
        self.log('val_loss', loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log('val_f1_score', f1, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = 0.01)
        return optimizer

def train_iris_model(train_loader, val_loader, input_size, hidden_size, max_epochs, model_path = None):
    logging.info("Starting training...")
    start_time = time.time()
    model = IrisNN(input_size = input_size, hidden_size = hidden_size)

    early_stop_callback = EarlyStopping(
        monitor = 'val_loss',
        patience = 3,
        verbose = True,
        mode = 'min'
    )
  
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_loss',
        dirpath = MODEL_PATH,
        filename = 'best_model',
        save_top_k = 1,
        mode = 'min'
    )
  
    trainer = pl.Trainer(
        max_epochs = max_epochs,
        callbacks = [early_stop_callback, checkpoint_callback]
    )
  
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    end_time = time.time()
    logging.info(f"Training finished in{end_time - start_time} seconds")
    
    logging.info("Saving the model...")
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
    
    if not model_path:
        model_path = os.path.join(MODEL_PATH, datetime.now().strftime(configur['general']['datetime_format']) + '.pickle')
    else:
        model_path = os.path.join(MODEL_PATH, model_path)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        logging.info("Model saved successfully.")

def get_preds_for_eval(model, dataloader):
    logging.info("Evaluating results...")
    model.eval()
    prediction_list = []
    target_list = []
    for batch in dataloader:
      inputs, targets = batch
      outputs = model(inputs)
      _, predictions = torch.max(F.softmax(outputs, dim = 1), 1)
      prediction_list.extend(predictions.detach().numpy())
      target_list.extend(targets.detach().numpy())
      res = f1_score(np.array(target_list) ,np.array(prediction_list))
      logging.info("Model achieved f1_score of {res:.2f}")
    return np.array(prediction_list), np.array(target_list)

def main():
    tr = TrainProcessor()
    train_loader, val_loader, input_size = tr.prepare_data(train_path = TRAIN_PATH)

    train_iris_model(
        train_loader = train_loader, 
        val_loader = val_loader, 
        input_size = input_size, 
        hidden_size = HIDDEN_SIZE,
        max_epochs = MAX_EPOCHS, 
                    )

if __name__ == '__main__':
    main()