import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras.utils

def load_and_preprocess_data(filepath):
    # Load data
    data = pd.read_csv(filepath, sep=';')
    data.columns = ["Text", "Emotions"]
    
    # Extract texts and labels
    texts = data["Text"].tolist()
    labels = data["Emotions"].tolist()
    
    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    # Label encoding
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    one_hot_labels = keras.utils.to_categorical(labels)
    
    return padded_sequences, one_hot_labels, tokenizer, max_length, label_encoder