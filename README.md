🩺 Lung Cancer Prediction – Machine Learning Project
This project predicts lung cancer severity levels using a machine learning model trained on patient health data. It utilizes a Random Forest Classifier to classify patient risk levels based on symptoms, habits, and medical history.

🎬 Demo Notebook
📓 Platform: Google Colab
📁 Model Export: lung_model.pkl (downloadable)
📊 Dataset: cancer patient data sets.csv

📌 Features
🔍 Predicts lung cancer severity (e.g., Low, Medium, High)
🧠 Trains a Random Forest Classifier
🧾 Supports 18+ features like Smoking, Chest Pain, Shortness of Breath
📈 Achieves 100% accuracy (on this specific dataset)
💾 Saves trained model as a .pkl file
🌐 Easy to deploy with Flask or Streamlit frontend

🛠️ Tech Stack
Python 3.x

pandas – data handling

scikit-learn – model training & evaluation

LabelEncoder – to encode categorical labels

pickle – for saving the model

Google Colab – development environment

🚀 Getting Started
1. Clone the Repository (Optional for GitHub Deployment)
bash
Copy
Edit
git clone https://github.com/your-username/lung-cancer-prediction.git
cd lung-cancer-prediction
2. Upload Data in Colab
Upload your cancer patient data sets.csv via:

python
Copy
Edit
from google.colab import files
files.upload()
3. Install Dependencies (if running locally)
bash
Copy
Edit
pip install pandas scikit-learn
⚙️ Model Training Steps
Load dataset and preprocess (LabelEncoder)

Select relevant features (X) and label (y)

Split data using train_test_split

Train RandomForestClassifier

Evaluate accuracy using accuracy_score

Save trained model using pickle

✅ Model Accuracy
🎯 Achieved Accuracy: 100%
⚠️ Note: This may be due to a clean, small, or imbalanced dataset. Always validate on real-world data.

🧪 Sample Code
python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
💡 Potential Extensions
📱 Build a web app (Flask / Streamlit)

📊 Add visualizations for patient risk

🧪 Improve model with more diverse data

🧮 Hyperparameter tuning with GridSearchCV

📁 Files Overview
File	Description
cancer patient data sets.csv	Dataset used for training
lung_model.pkl	Trained ML model (saved using pickle)
lung_cancer_prediction.ipynb	Colab notebook with full workflow

📞 Contact Information
Developer: Shanmathi G
📧 Email: shanmathigopikrishnan@gmail.com
🔗 GitHub: shanmathigopikrishnan
🔗 LinkedIn: Shanmathi G

✨ Feel free to connect for contributions, queries, or collaborations!

