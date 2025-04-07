from preprocess import load_and_preprocess_data
from model import build_and_train_model
import os

def main():
    # Load and preprocess data
    texts, labels, tokenizer, max_length, label_encoder = load_and_preprocess_data("../data/train.txt")
    
    # Build and train model
    model = build_and_train_model(texts, labels, tokenizer, max_length)
    
    # Save model and preprocessing artifacts
    os.makedirs("../models", exist_ok=True)
    model.save("../models/emotions_model.h5")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()