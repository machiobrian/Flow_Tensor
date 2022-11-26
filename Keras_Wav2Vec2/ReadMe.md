# Audio Classification with HuggingFace Transformers:

- Identification of speech commands

Aka.... Keyword Spotting (kws) is important for:
- indexing audio databases
- indexing keywords
- running speech models on uControllers: [our main task]

# Requirements
pip install

- transformers
- datasets
- huggingface-hub
- joblib
- librosa

# Python Random Seed() Method
the seed() method -> initializes the random number generator
random # generator needs a random number to start with (seed value = 42) inorder to generate a random #

- Wav2vec2  results in an output freq with a stride of about 20ms

# Google Speech Commands v1 Dataset
Popular benchmark for training and evaluting deep learning models built for the KWS task.
- 10 Classes of keywrds
- Class for silence
- Class for Unknown (false positives)
# stratify - split into distinct layers/strata

# labels on the datasets
- {'0': 'yes', '1': 'no', '2': 'up', '3': 'down', '4': 'left', '5': 'right', '6': 'on', '7': 'off', '8': 'stop', '9': 'go', '10': '_silence_', '11': '_unknown_'}


# Pre-processing of te Audio Utterances before they are fed into the model
- We make use of hugging face transformers - Feature Extractor

The feature extractor:

    - Resamples the inputs the the sample_rate the model expects.
    - Generates the inputs the models require.

The AutoFeatureExtractor ensures:
    - We get a feature extractor corresponding to the model we want to use
    - Look into the map function, loading dataset as numpy arrays

# Defining the Wav2vec2 w/Classification Head
    - Model Definition
- Define the Wav2Vec2.0 model
- Add a Classification Head on top -> Outputs a probability distribution of all classes for each audio sample
- Instantiate our main Wav2vec2 model using the TFWav2Vec2Model class
        
    - The instantiated model outputs 768/1024 dimensions acoording to the conf we choose for our model (base/large).

    - from_pretrained() -> loads the pretrained weights from the Huggingface hub.

           - Loads the pretrained weights with the config corresponding t the model (base)

Attention_mask -> Allows us to send a batch into the transformer even with exxmple in the batch having varying lengths

Dropout is a technique where randomly selected neurons are ignored during training

        - attention_mask = batch_size * max_seq_len