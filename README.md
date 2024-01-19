# Basics MLE Module Homework

## Project Structure

<code>epam_hometask
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
</code>

## Data prep
Running <code>data_prep.py</code> script performs the following:  
1. Downloads data from the webpage
2. Saves full dataset into "data" folder as a .csv file
3. Splits dataset into training and inference parts according to <code>test_size</code> parameter in settings.json
4. Saves training and inference dataset into "data" folder as a .csv file 

## Training
Running <code>data_prep.py</code> script performs the following:  

## Inference
