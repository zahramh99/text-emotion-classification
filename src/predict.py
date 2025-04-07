import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

class EmotionPredictor:
    def __init__(self, model_path, tokenizer_path=None):
        self.model = load_model(model_path)
        if tokenizer_path:
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
    
    def predict(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.model.input_shape[1])
        prediction = self.model.predict(padded)
        return prediction

if __name__ == "__main__":
    predictor = EmotionPredictor("../models/emotions_model.h5")
    test_text = "I'm feeling wonderful today!"
    print(f"Prediction: {predictor.predict(test_text)}")