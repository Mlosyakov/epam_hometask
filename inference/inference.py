import json
import logging
import io
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from pathlib import Path

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
MODEL_PATH = os.path.join(DATA_DIR, configur["general"]["models_dir"])

def preprocess_inf(url : str, target_col = 'Species'):
  df = pd.read_csv(url)
  X = df.drop(columns = target_col).values
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
    label_unencode = np.load('/content/Value dictionary.npy',allow_pickle='TRUE').item()
  return np.vectorize(label_unencode.get)(prediction_list)


model = load_model(model_dir)

data = preprocess_inf('/content/inference/inference.csv')

preds = get_preds(model, data)
