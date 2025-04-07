
# Text Emotions Classification

A deep learning model to classify text into emotional categories.

## Project Structure
text-emotions-classification/
├── src/ # Source code
├── models/ # Trained models
├── notebooks/ # Exploration notebooks
├── data/ # Dataset files
├── requirements.txt # Dependencies
└── README.md # This file


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
### 8. `.gitignore`
Python
pycache/
*.py[cod]
*.pyc
*.so
.Python
build/
*.egg-info/
.ipynb_checkpoints/

Data
*.csv
*.pkl
*.h5
models/

Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

IDE
.vscode/
.idea/
*.swp
*.swo

System
.DS_Store
Thumbs.db

To use:
1. Place your dataset in `data/train.txt`
2. Run `pip install -r requirements.txt`
3. Execute `python src/main.py` to train
4. Use `python src/predict.py` for predictions