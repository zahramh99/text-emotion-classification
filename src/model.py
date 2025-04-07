from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

def build_and_train_model(texts, labels, tokenizer, max_length):
    # Split data
    xtrain, xtest, ytrain, ytest = train_test_split(texts, labels, test_size=0.2)
    
    # Build model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
                      output_dim=128, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=len(labels[0]), activation="softmax"))
    
    # Compile and train
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))
    
    return model