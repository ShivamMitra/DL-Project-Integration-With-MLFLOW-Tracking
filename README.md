# DL-Project-Integration-With-MLFLOW-Tracking
A deep-learning project that predicts the quality of wine using neural networks, integrated with MLflow for experiment tracking, model logging and reproducibility.

---

## 📌 Project Overview  
This project aims to predict the quality score of wines from their chemical attributes using a deep learning model (e.g., neural network). The workflow includes:  
- Data preprocessing (feature scaling, train-test split)  
- Building a neural network model (e.g., via TensorFlow/Keras or PyTorch)  
- Training the model and logging parameters, metrics, and model artifacts via MLflow  
- Enabling experiment tracking and reproducibility  

---

## 🎯 MLflow Tracking Link  
- 🏃 **Run Details**: http://127.0.0.1:5000/#/experiments/405720205119111355/runs/d195d49ed5a14f45853da4689c2f5a90  
- 🧪 **Experiment Dashboard**: http://127.0.0.1:5000/#/experiments/405720205119111355  

---

## 🚀 Getting Started  
### Prerequisites  
Make sure you have:  
- Python 3.x  
- Virtual environment support (venv, conda, etc)  
- MLflow UI running (for tracking)  

### Steps  
1. Clone the repository:  
   ```bash
   git clone https://github.com/ShivamMitra/DL-Project-Integration-With-MLFLOW-Tracking.git
   cd DL-Project-Integration-With-MLFLOW-Tracking
2. Create & activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Start MLflow UI in a separate terminal (if not already running):
   ```bash
   mlflow ui
   # by default: http://127.0.0.1:5000
5. Run the model training script or notebook (e.g., wine_quality_model.py or wine_quality.ipynb) to log the run to MLflow.

## 📂 Project Structure
```
java
DL-Project-Integration-With-MLFLOW-Tracking/
├── data/                     ← dataset (e.g., winequality-red.csv / winequality-white.csv)
├── notebooks/                ← Jupyter notebook(s) for EDA and model building
├── src/                      ← source code (model, training script, utils)
├── models/                   ← saved model artifacts (optional)
├── requirements.txt          ← python dependencies
└── README.md                 ← this file
```
## 📋 Dependencies
Typical dependencies (versions may vary):
- numpy, pandas
- scikit-learn (for dataset handling and preprocessing)
- tensorflow or keras (for deep learning model)
- mlflow (for experiment tracking)
- matplotlib / seaborn (for plots)

(Be sure to pin exact versions in requirements.txt for reproducibility.)

## 🔍 Usage & Experimentation
- Open the notebook or run the script to train the model.

- Use MLflow UI to view logged parameters, metrics, and artifacts.
- Experiment with hyper-parameters (learning rate, number of layers, dropout, epochs) and compare runs.
- Optionally register the best model in MLflow Model Registry, and serve or deploy it.

## 🧩 Notes & Tips
- Ensure the MLflow tracking server is up and running (mlflow ui) before running training.
- Set MLFLOW_TRACKING_URI if you use a remote tracking server.
- Use meaningful run names and tags in MLflow to help organise experiments.
- Version your dataset or model artifacts for reproducibility.

## 🤝 Contributing
Feel free to contribute enhancements such as:
- Trying different network architectures (CNN, LSTM, etc)
- Dataset augmentation or alternate datasets
- Model serving or deployment (Flask/FastAPI + MLflow Model Serving)
- CI/CD pipeline for retraining and logging new runs

## 📄 License
This project is licensed under the MIT License (or whichever you choose). See the LICENSE file for details.
