import json
import logging
import io
import os
import pandas as pd
import sys
import torch
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

MODEL_PATH = os.path.join(DATA_DIR, configur["general"]["models_dir"])
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("No models found. Please run train.py first")

DATA_DIR = get_project_dir(configur["general"]["data_dir"])
INFERENCE_PATH = os.path.join(DATA_DIR, configur["inference"]["inf_table_name"])
TARGET_COL = configur["general"]["target_col"]
DICT_NAME = configur["train"]["dict_name"]
DICT_LOC = os.path.join(MODEL_PATH, DICT_NAME)

def preprocess_inf(path):
    df = pd.read_csv(path)
    X = df.drop(columns = TARGET_COL).values
    X = torch.tensor(X, dtype = torch.float32)
    X = DataLoader(dataset = X, batch_size = len(X), num_workers=1)
    return X

def load_model(folder_path):
    model_files = list(Path(folder_path).rglob('*.pth'))
    checkpoint_files = list(Path(folder_path).rglob('*ckpt'))
  
    if not model_files or not checkpoint_files:
      raise FileNotFoundError('No models or checkpoints found. Train model first')
  
    latest_model_path = max(model_files, key = os.path.getctime)
    latest_checkpoint_path = max(checkpoint_files, key = os.path.getctime)
  
    if latest_model_path.parts[0].startswith('http'):
      raise NotImplementedError('Please use only local models')
    else:
      model = torch.load(latest_model_path, map_location = torch.device('cpu'))
  
    if latest_checkpoint_path.parts[0].startswith('http'):
      raise NotImplementedError('Please use only local models')
    else:
      checkpoint = torch.load(latest_checkpoint_path, map_location = torch.device('cpu'))
  
    model.load_state_dict(checkpoint['state_dict'])
    return model

def get_preds(model, dataloader):
    model.eval()
    prediction_list = []
    for batch in dataloader:
      inputs = batch
      outputs = model(inputs)
      _, predictions = torch.max(F.softmax(outputs, dim = 1), 1)
      prediction_list.extend(predictions.detach().numpy())
    label_decode = np.load(DICT_LOC, allow_pickle='TRUE').item()
    results = np.vectorize(label_decode.get)(prediction_list)
    

def main():
    model = load_model(MODEL_PATH)
    data = preprocess_inf(INFERENCE_PATH)
    get_preds(model, data)
    

if __name__ == '__main__':
    main()

