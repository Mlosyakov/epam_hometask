import json
import logging
import io
import os
import pandas as pd
import pickle
import pytorch_lightning as pl
import sys
import time
import torch
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import get_project_dir, configure_logging

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    raise RuntimeError("No data found. Please run data_prep.py first")

CONF_FILE = ".vscode/settings.json"
logger.info("Getting few important dependencies...")
with open(CONF_FILE, "r") as file:
    configur = json.load(file)

logger.info("Defining paths...")
MODEL_DIR = os.path.join(DATA_DIR, configur["general"]["models_dir"])
#if not os.path.exists(MODEL_DIR):
    #raise RuntimeError("No model directory found. Please run train.py first.")

RESULTS_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../results'))
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DATA_DIR = get_project_dir(configur["general"]["data_dir"])
INFERENCE_PATH = os.path.join(DATA_DIR, configur["inference"]["inf_table_name"])
TARGET_COL = configur["general"]["target_col"]
DICT_NAME = configur["general"]["dict_name"]
DICT_PATH = os.path.join(MODEL_DIR, DICT_NAME+".npy")
RESULTS_PATH = os.path.join(RESULTS_DIR, configur["inference"]["res_table_name"])
MODEL_PATH = os.path.join(MODEL_DIR, configur["inference"]["model_name"])
INF_MODEL_PATH = os.path.join(RESULTS_DIR, configur["inference"]["model_name"])

def preprocess_inf(path):
    logging.info("Loading data...")
    df = pd.read_csv(path, encoding="utf-16")
    X = df.drop(columns = TARGET_COL).values
    X = torch.tensor(X, dtype = torch.float32)
    X = DataLoader(dataset = X, batch_size = len(X), num_workers=1)
    logging.info(f"Loaded {len(df)} samples for inference.")
    return X

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


def load_model(folder_path = None):
    logging.info("Loading trained model...")
    model_files = list(Path(folder_path).rglob('*.pth'))
    checkpoint_files = list(Path(folder_path).rglob('*ckpt'))
    
    print(model_files)
    print(checkpoint_files)
    
    if not model_files or not checkpoint_files:
        raise FileNotFoundError('No models or checkpoints found. Train model first')
  
    latest_model_path = max(model_files, key = os.path.getctime)
    latest_checkpoint_path = max(checkpoint_files, key = os.path.getctime)
    
    print(latest_model_path)
    print(latest_checkpoint_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if latest_model_path.parts[0].startswith('http'):
        raise NotImplementedError('Please use only local models')
    else:
        model = torch.load(latest_model_path, map_location = torch.device(device))
    if latest_checkpoint_path.parts[0].startswith('http'):
        raise NotImplementedError('Please use only local models')
    else:
        checkpoint = torch.load(latest_checkpoint_path, map_location = torch.device(device))
    
    model.load_state_dict(checkpoint['state_dict'])

    logging.info("Saving latest model into results folder...")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if not folder_path:
        model_path = MODEL_PATH
    else:
        model_path = os.path.join(RESULTS_DIR, folder_path)
    
    torch.save(model, INF_MODEL_PATH)
    logging.info("Model saved successfully.")
    return model

def get_preds(model, dataloader):
    logging.info("Inference in progress...")
    start_time = time.time()
    model.eval()
    prediction_list = []
    for batch in dataloader:
        inputs = batch
        outputs = model(inputs)
        _, predictions = torch.max(F.softmax(outputs, dim = 1), 1)
        prediction_list.extend(predictions.detach().numpy())
    end_time = time.time()
    label_decode = np.load(DICT_PATH, allow_pickle='TRUE').item()
    results = np.vectorize(label_decode.get)(prediction_list)
    pd.Series(results).to_csv(RESULTS_PATH, index = False, encoding="utf-16")
    logging.info(f"Model finished predicting in {end_time - start_time} seconds. Output is stored in .csv file in results folder. Have a great day!")

def main():
    model = load_model(MODEL_DIR)
    data = preprocess_inf(INFERENCE_PATH)
    get_preds(model, data)
    

if __name__ == '__main__':
    main()

