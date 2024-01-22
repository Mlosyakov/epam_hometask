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
    raise RuntimeError("No data found. Please run data_prep.py first")

if not os.listdir(DATA_DIR):
    raise RuntimeError(f"The directory {DATA_DIR} is empty. Please populate it with data first.")

CONF_FILE = "settings.json"
logger.info("Getting few important dependencies...")
with open(CONF_FILE, "r") as file:
    configur = json.load(file)

logger.info("Defining paths...")
DATA_DIR = get_project_dir(configur["general"]["data_dir"])
MODEL_DIR = get_project_dir(configur["general"]["models_dir"])

TRAIN_PATH = os.path.join(DATA_DIR, configur["train"]["table_name"])
MODEL_PATH = os.path.join(MODEL_DIR, configur["inference"]["model_name"])

DATETIME_FORMAT = configur["general"]["datetime_format"]
RANDOM_STATE = configur["general"]["random_state"]
TARGET_COL = configur["general"]["target_col"]
DICT_NAME = configur["general"]["dict_name"]
HIDDEN_SIZE = configur["train"]["hiiden_size"]
MAX_EPOCHS = configur["train"]["max_epochs"]
TEST_SIZE = configur["train"]["test_size"]
BATCH_SIZE = configur["train"]["batch_size"]
if BATCH_SIZE > 32:
    raise RuntimeError("Pick smaller batch size. To specify batch size go to settings.json")

class TrainProcessor():
    """
    This class prepares training part of Iris dataset.
    Function encode_target saves decoder into models directory to provide object output like in initial datasey
    instead of returning labels of classes. 
    Methods do not save training and validation datasets.
    Class is intended to be used as a whole, calls for specific methods were not tested.
    This class also defines input and output size of layers used in model.
    """
    
    def __init__(self):
        pass

    def prepare_data(self, train_path):
        logging.info("Loading data...")
        df = pd.read_csv(train_path, encoding="utf-16")
        df = self.encode_target(df)
        X_train, X_test, y_train, y_test, input_size, output_size = self.split_train(df)
        X_train_tens, y_train_tens, X_val_tens, y_val_tens = self.convert_to_tensors(X_train, X_test, y_train, y_test)
        train_loader, val_loader = self.prepare_dataloader(X_train_tens, y_train_tens, X_val_tens, y_val_tens, BATCH_SIZE)
        return train_loader, val_loader, input_size, output_size

    def encode_target(self, df):
        logging.info("Preparing data...")
        labels = list(range(len(df[TARGET_COL].unique())))
        #label encoding target column
        mapping = dict(zip(df[TARGET_COL].unique(), labels))
        inv_map = {v: k for k, v in mapping.items()} #decoder for later use in output 
        
        logging.info("Saving decoder...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        dict_path = os.path.join(MODEL_DIR, DICT_NAME)
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
        output_size = len(np.unique(y))
        logging.info(f"Data is prepared with {len(data)} samples. {len(X_train)} of them used for training and rest for validation")
        return X_train, X_test, y_train, y_test, input_size, output_size

    def convert_to_tensors(self, X_train, X_test, y_train, y_test):
        train_tensors = (torch.tensor(X_train, dtype = torch.float32),
                        torch.tensor(y_train, dtype = torch.long))
        val_tensors = (torch.tensor(X_test, dtype = torch.float32),
                        torch.tensor(y_test, dtype = torch.long))
        return *train_tensors, *val_tensors

    def prepare_dataloader(self, X_train_tens, y_train_tens, X_val_tens, y_val_tens, batch_size):
        train_dataset = TensorDataset(X_train_tens, y_train_tens)
        val_dataset = TensorDataset(X_val_tens, y_val_tens)
        train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True, num_workers=3, persistent_workers= True)
        val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, num_workers=3, persistent_workers= True)
        logging.info("Preparation successful.")
        return train_loader, val_loader


class IrisNN(pl.LightningModule):
    def __init__(self, input_size, output_size, hidden_size):
        super(IrisNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.train_metrics = torchmetrics.F1Score(task='multiclass', num_classes = output_size)
        self.val_metrics = torchmetrics.F1Score(task='multiclass', num_classes = output_size)
    
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

def train_iris_model(train_loader, val_loader, input_size, output_size, hidden_size, max_epochs, model_path = None):
    logging.info("Starting training...")
    start_time = time.time()
    model = IrisNN(input_size = input_size, output_size = output_size, hidden_size = hidden_size)

    early_stop_callback = EarlyStopping(
        monitor = 'val_loss',
        patience = 5,
        verbose = True,
        mode = 'min'
    )
  
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_loss',
        dirpath = MODEL_DIR,
        filename = 'best_model',
        save_top_k = 1,
        mode = 'min'
    )
  
    trainer = pl.Trainer(
        max_epochs = max_epochs,
        callbacks = [early_stop_callback, checkpoint_callback],
        log_every_n_steps = 4
    )
  
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = val_loader)
    end_time = time.time()
    logging.info(f"Training finished in {(end_time - start_time):.2f} seconds")
    
    logging.info("Saving the model...")
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    torch.save(model, MODEL_PATH)
    logging.info("Model saved successfully.")

    return model

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
    res = f1_score(np.array(target_list) ,np.array(prediction_list), average = 'micro')
    logging.info(f"Training Finished! Model achieved f1_score of {res:.2f}")
    return np.array(prediction_list), np.array(target_list)

def main():
    tr = TrainProcessor()
    train_loader, val_loader, input_size, output_size = tr.prepare_data(train_path = TRAIN_PATH)
    model = train_iris_model(
        train_loader = train_loader, 
        val_loader = val_loader, 
        input_size = input_size,
        output_size = output_size,
        hidden_size = HIDDEN_SIZE,
        max_epochs = MAX_EPOCHS, 
    )
    get_preds_for_eval(model, val_loader)
if __name__ == '__main__':
    main()
