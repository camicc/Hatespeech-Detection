# Sarcasm detection using sentiment analysis

Abstract— blblbblblbl

# EE-559 Deep Learning: Group Mini-Project, Group 49

**Team members**: Camille Pittet, David Friou and Théo Lacroix

# How to run our project

To reproduce our results, you can execute the `train_svm.py` script. Use the following command when you are located in the main repository:

```bash
 python train_svm.py --config-key [key]
```
## Configurations Keys

The *--config-key* argument lets you select the model configuration to use:

- t: Only MUStARD model
- s: Only Sentiment features
- st: Combination of MUStARD and Sentiment features

You can also specify the training and testing conditions:

- i-: Speaker-independent settings (default is speaker-dependent)
- -c: Use context features along with the utterance (default uses only the utterance)

For example, using `i-st-c` would set the configuration to use both MUStARD and Sentiment features, in a speaker-independent setting, including context features.
 
 The configuration mappings are specified in a dictionary within the config.py file as follows:
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
- `hartmann` : Weighted scores across all categories
- `hartmann_max` : Focus solely on the dominant sentiment
- `vader1` : With the compound score
- `vader2` : Without the compound score

And `###` can be:

- `U` : Only Utterances
- `UoC` : Overall Context Sentiment
- `UpC` : Sentence-specific Sentiment


You can manually modify the parameters in the config.py and execute the script without specifying a config key::

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

## `train_svm.py`

## `data_loader.py`

## `config.py`

## `Sentiments_features.py`

## `preprocessed_data`

## `data`


# Mentions

This code uses modified code from the "MUStARD: Multimodal Sarcasm Detection Dataset" (`data_loader.py`,`train_svm.py` and `config.py`), available at https://github.com/soujanyaporia/MUStARD and is licensed under the MIT License.