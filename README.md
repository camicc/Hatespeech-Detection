# Sarcasm detection using sentiment analysis

Abstract— blblbblblbl

# EE-559 Deep Learning: Group Mini-Project, Group 49

**Team members**: Camille Pittet, David Friou and Théo Lacroix

# How to run our project

You can find all 

## Data

In order to be albe to test the enhancement through sentiment integration to the pre-existing algorithm used in the MUStARD model (see mentions), the following files are needed: 

 - [Download the pre-extracted BERT features](https://drive.google.com/file/d/1GYv74vN80iX_IkEmkJhkjDRGxLvraWuZ/view?usp=sharing) and place the two files directly under the folder `data/` (so they are `data/bert-output.jsonl` and `data/bert-output-context.jsonl`)

## External libraries
The following libraries are required to run our project:

- `numpy`
- `scikit-learn`
- `h5py`
- `jsonlines`
- `nltk`
- `pandas`
- `vaderSentiment`
- `transformers`
- `torch`

You can install them using one of the followings commands:
```
pip install -r requirements.txt
```
```
conda env create -f environment.yml
```

# Folders and files


# Mentions

This code uses modified code from the "MUStARD: Multimodal Sarcasm Detection Dataset" (`data_loader.py`,`train_svm.py` and `config.py`), available at https://github.com/soujanyaporia/MUStARD and is licensed under the MIT License.