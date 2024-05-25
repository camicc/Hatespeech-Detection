"""
This file uses modified code from the "MUStARD: Multimodal Sarcasm Detection Dataset", 
available at https://github.com/soujanyaporia/MUStARD
and is licensed under the MIT License.
"""

class Config:
    
    # DL PROJECT CONFIG
    use_sentiment_text = False  # adds sentiment features to text

    model = "SVM"
    runs = 1  # No. of runs of experiments

    # Training modes
    use_context = False # whether to use context information or not (default false)
    use_author = False # add author one-hot encoding in the input

    use_bert = False 

    use_target_text = False
    use_target_audio = False  # adds audio target utterance features.
    use_target_video = False  # adds video target utterance features.

    speaker_independent = False  # speaker independent experiments

    embedding_dim = 300  # GloVe embedding size
    word_embedding_path = "/home/sacastro/glove.840B.300d.txt"
    max_sent_length = 20
    max_context_length = 4  # Maximum sentences to take in context
    num_classes = 2  # Binary classification of sarcasm

    svm_c = 10.0
    svm_scale = True

    fold = None

# DL PROJECT CONFIG
class SpeakerDependentSConfig(Config): # s
    use_sentiment_text = True 
    svm_c = 1.0

class SpeakerIndependentSConfig(Config): # i-s
    use_sentiment_text = True
    svm_scale = False
    svm_c = 10.0
    speaker_independent = True

class SpeakerDependentSTConfig(Config): # st
    use_target_text = True
    use_bert = True
    use_sentiment_text = True
    svm_c = 1.0

class SpeakerDependentCSTConfig(Config): # st-c
    use_target_text = True
    use_bert = True
    use_context = True
    use_sentiment_text = True
    svm_c = 1.0

class SpeakerIndependentSTConfig(Config): # i-st
    svm_scale = False
    use_target_text = True
    use_bert = True
    use_sentiment_text = True
    svm_c = 10.0
    speaker_independent = True

class SpeakerIndependentCSTConfig(Config): # i-st-c
    svm_scale = False
    use_target_text = True
    use_bert = True
    use_context = True
    use_sentiment_text = True
    svm_c = 10.0
    speaker_independent = True

# DL PROJECT CONFIG

class SpeakerDependentTConfig(Config): # t
    use_target_text = True
    use_bert = True
    svm_c = 1.0

class SpeakerDependentCTConfig(Config): # t-c
    use_target_text = True
    use_bert = True
    use_context = True
    svm_c = 1.0

class SpeakerIndependentTConfig(Config): # i-t
    svm_scale = False
    use_target_text = True
    use_bert = True
    svm_c = 10.0
    speaker_independent = True

class SpeakerIndependentTConfig(Config): # i-t-c
    svm_scale = False
    use_target_text = True
    use_bert = True
    use_context = True
    svm_c = 10.0
    speaker_independent = True

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
