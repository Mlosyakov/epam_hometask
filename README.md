# Basics MLE Module Homework
This is a project for Basics MLE module of a course. All .py scripts were tested on MacOS, if there are any performance issues on different OS please let me know. <br />
Project does not require additional setup and can be run as is when cloned. <br />
Upon completion of intermediate steps script returns small and (sometimes) informative log.<br />
As a result you will recieve <code>.csv</code> file with predictions and <code>.pth</code> model file of the latest trained model.<br />

## Project Structure

~~~
epam_hometask
├── data                      # Data files used for training and inference, file containing complete dataset
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
├── results                    # Folder where final model and results are stored
│   ├── Outputs.csv
│   └── model files
├── utils.py                  # Utility functions and classes that are used in scripts
├── requirements.txt          # All requirements for the project
├── settings.json             # All configurable parameters and settings
└── README.md
~~~
## How to run
### Training
To run training you should first creare image for training.py
To create image run:
~~~
docker build -t train_img -f training/Dockerfile .
~~~
And then run this command to execute training
~~~
docker run -it train_img /bin/bash 
~~~
Doing the following will create docker image with copied data for training and output trained model.
### Inference
To run inference you should first creare image for inference.py
To create image run:
~~~
docker build -t inference_img -f inference/Dockerfile .
~~~
And then run this command to execute inference
~~~
docker run -it inference_img /bin/bash 
~~~
Doing the following will create docker image with copied data for training and output trained model.

Alternatively you can simply run python scripts to ensure that everything works as intended. 
These scripts should be run in order, demonstrated below to successfully build the model and not return any errors:

1. Run data_prep
2. Run training.py
3. Run inference.py

Succsessful run of data_prep is indicated by creating <code>data</code> directory and 3 files inside of it;  
Succsessful run of data_prep is indicated by creating <code>models</code> directory, model file, checkpoint file and decoder file inside of it  
Succsessful run of inference is indicated by creating <code>results</code> directory, outputs file inside of it  

## Information about each script

### Data prep
Running <code>data_prep.py</code> script performs the following:  
1. Downloads data from the webpage;
2. Saves full dataset into <code>data</code> directory as a <code>.csv</code> file. If <code>data</code> directory does not exist, directory is created;
3. Splits dataset into training and inference parts according to <code>test_size</code> parameter in settings.json;
4. Saves training and inference dataset into <code>data</code> directory as a <code>.csv</code> files with names specified in <code>settings.json</code>;

### Training
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

### Inference
Running <code>inference.py</code> script performs the following:  
1. Inference file from <code>data</code> directory is preprocessed for predictions;
2. Model and checkpoint with best performance are loaded from <code>model</code> directory;
3. Inference data is passed into a model and outputs are saved in <code>results</code> directory as a <code>.csv</code> file;
