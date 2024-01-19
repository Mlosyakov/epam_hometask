# Basics MLE Module Homework
This is a project for Basics MLE module of a course. All scripts were tested on MacOS, if there are any performance issues on different OS please let me know.
## Project Structure

~~~
epam_hometask
├── data                      # Data files used for training and inference (can be generated with data_generation.py script) and file containing complete dataset
│   ├── raw_data.csv
│   ├── inference_iris.csv
│   └── train_iris.csv
├── data_prep                # Scripts used for data uploading and splitting into training and inference parts
│   ├── data_prep.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── inference.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── requirements.txt          # All requirements for the project
├── settings.json             # All configurable parameters and settings
└── README.md
~~~

## Data prep
Running <code>data_prep.py</code> script performs the following:  
1. Downloads data from the webpage;
2. Saves full dataset into <code>data</code> directory as a .csv file. If <code>data</code> directory does not exist, directory is created;
3. Splits dataset into training and inference parts according to <code>test_size</code> parameter in settings.json;
4. Saves training and inference dataset into <code>data</code> directory as a .csv files with names specified in settings.json;

## Training
Running <code>train.py</code> script performs the following:  
1. Training file from <code>data</code> is preprocessed for modelling:
    <ul>
     <li> Target column is label encoded, decoder is saved in <code>model</code> directory for future use;</li>
     <li> Data is split into training and validation parts;</li>
     <li> Train and validation datasets are converted to Dataloaders;</li>
    </ul>
2. Model is trained and validated on created dataloaders;
3. Model is saved in <code>model</code> directory;
4. Model checkpoint with best model performance is saved into <code>models</code> directories;
5. F1 score of best performing checkpoint is printed out;

## Inference
Running <code>inference.py</code> script performs the following:  
1. Inference file from <code>data</code> directory is preprocessed for predictions;
2. Model and checkpoint with best performance are loaded from <code>model</code> directory;
3. Inference data is passed into a model and outputs are saved in <code>results</code> directory as a .csv file;
