# Sarcasm detection using sentiment analysis

Abstract— blblbblblbl

# EE-559 Deep Learning: Group Mini-Project, Group 49

**Team members**: Camille Pittet, David Friou and Théo Lacroix

# How to run our project

To reproduce our results,  navigate to the `src` directory and execute the following command:
```bash
 python train_svm.py --config-key [key]
```
## Configurations Keys

Use the  `--config-key` argument to select the model configuration:

- t: Only the MUStARD model.
- s: Only Sentiment features.
- st: Combination of MUStARD and Sentiment features.

Additionally, specify the training and testing conditions:

- i-: Speaker-independent settings (default is speaker-dependent)
- -c: Use context features for the MUStARD model along with the utterance (default uses only the utterance). 

**Note:** The results presented in the paper utilize the context for the MUStARD model and its combinations (M-V & M-H). The context for the Sentiment features is predetermined by the chosen data paths.

For example, using `i-st-c` would set the configuration to use both MUStARD and Sentiment features, in a speaker-independent setting, including context features.
 
 The configuration mappings are specified in a dictionary within the `config.py` file as follows:
```javascript
CONFIG_BY_KEY = {
    "": Config(),
    "t": SpeakerDependentTConfig(),
    "i-t": SpeakerIndependentTConfig(),
    "s": SpeakerDependentSConfig(),
    "i-s": SpeakerIndependentSConfig(),
    "st": SpeakerDependentSTConfig(),
    "i-st": SpeakerIndependentSTConfig(),
    "st-c": SpeakerDependentCSTConfig(),
    "i-st-c": SpeakerIndependentCSTConfig(),
    "t-c": SpeakerDependentCTConfig(),
    "i-t-c": SpeakerIndependentTConfig(),
}
```

## Specifying Data Paths

To choose between the variants in the Sentiment Analysis and Sentiment Processing Strategies, update the `DATA_PATH` in the `data_loader.py` file within the `preprocessed_data`folder. The path follows this nomenclature:

```python
DATA_PATH = "preprocessed_data/sarcasm_data_SentimentModel_###.json"
```

Where `SentimentModel` can be: 
- `hartmann` : Weighted scores across all categories.
- `hartmann_max` : Focus solely on the dominant sentiment.
- `vader1` : With the compound score.
- `vader2` : Without the compound score.

And `###` can be:

- `U` : Only Utterances.
- `UoC` : Overall Context Sentiment.
- `UpC` : Sentence-specific Sentiment.


You can manually modify the parameters in the config.py and execute the script without specifying a config key :

```
python train_svm.py
```

Results will be displayed in the terminal, and an output/SVM.json file will be generated, storing the output.

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
List of all the files we implemented or modified for the scope of this project.

## `src/`
Contains the source code of the project

### `train_svm.py`
Script used to run all experiments.

### `data_loader.py`
Script to help load the data into the classification model and select the *Data Paths* to choose between the variants in the Sentiment Analysis and Sentiment Processing Strategies

### `config.py`
A config file to select the parametres to choose the model configuration to use as well than the training and testing conditions, as explained in (how to run your code). 
It enable as well the *--config-key* argument for run the script `train_svm.py`.

### `Sentiments_features.ipynb`
Notebook used to perform the sentimen analysis with the two model VADER and Hartmann. As well as that prepare the sentiment Processing Strategies in the correct format to run our experiments.

## `preprocessed_data/`
Folder storing all the variants in the Sentiment Analysis and Sentiment Processing Strategies used in our experiments are stored here and were created by the the `Sentiments_features.py`

## `data/`
Folder storing the original dataset for the MUStARD project (see mention)
as well as the BERT features mentioned in the data section.

# Mentions

This code uses modified code from the "MUStARD: Multimodal Sarcasm Detection Dataset" (`data_loader.py`,`train_svm.py` and `config.py`), available at https://github.com/soujanyaporia/MUStARD and is licensed under the MIT License.