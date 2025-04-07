
# Text Emotions Classification

A deep learning model to classify text into emotional categories.


## Usage
1. Install dependencies:
pip install -r requirements.txt

Train the model:
python src/main.py
Make predictions:
python src/predict.py "Your text here"
Dataset
Place your dataset in data/train.txt in the format:
text;emotion
"I'm happy";joy
Training
Model will be saved to models/emotions_model.h5

To use:
1. Place your dataset in `data/train.txt`
2. Run `pip install -r requirements.txt`
3. Execute `python src/main.py` to train
4. Use `python src/predict.py` for predictions
