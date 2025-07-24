# 🩺 Lung Cancer Prediction – Machine Learning Project

This project predicts **lung cancer severity levels** using a machine learning model trained on patient health data. It uses a **Random Forest Classifier** to classify risk based on symptoms, habits, and genetic/medical history.

---

## 🎬 Demo Video

📺 **Watch Demo on YouTube**
👉 [Click here to watch the demo](https://www.youtube.com/watch?v=your-demo-video-link) *(Replace with actual link)*

---

## 📌 Features

✅ Predicts **lung cancer severity** (Low / Medium / High)
🧠 Trains a **Random Forest** ML model
📁 Uses 18+ key features like **Smoking**, **Shortness of Breath**, **Chest Pain**, etc.
📈 Achieves high accuracy (100% in this dataset)
💾 Exports model as `.pkl` file
🌐 Ready for web integration using Flask/Streamlit

---

## 🛠️ Tech Stack

* Python 3.x
* `pandas` – Data manipulation
* `scikit-learn` – Machine learning
* `LabelEncoder` – Categorical encoding
* `pickle` – Model serialization
* `Google Colab` – Development platform

---

## 🚀 Getting Started

### 1. Clone the Repository (Optional)

```bash
git clone https://github.com/your-username/lung-cancer-prediction.git
cd lung-cancer-prediction
```

### 2. Upload Dataset in Colab

```python
from google.colab import files
files.upload()
```

> Upload `cancer patient data sets.csv`

### 3. Install Requirements (for local run)

```bash
pip install pandas scikit-learn
```

---

## ⚙️ Model Workflow

1. Load and preprocess dataset (`LabelEncoder`)
2. Select features (`X`) and labels (`y`)
3. Split dataset with `train_test_split`
4. Train model using `RandomForestClassifier`
5. Evaluate using `accuracy_score`
6. Save model using `pickle`

---

## ✅ Accuracy

🎯 Achieved **100% accuracy** on test data
⚠️ Real-world testing required for validation

---

## 💡 Sample Commands

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 📁 Files

| File                           | Description                          |
| ------------------------------ | ------------------------------------ |
| `cancer patient data sets.csv` | Dataset for model training           |
| `lung_model.pkl`               | Saved ML model (Pickle format)       |
| `lung_cancer_prediction.ipynb` | Full notebook with training pipeline |

---

## 🔮 Future Improvements

* 🧪 Use more diverse & real-world datasets
* 📊 Add EDA & data visualizations
* 🧮 Add hyperparameter tuning
* 🖥 Deploy as web app (Flask / Streamlit)

---

## 📞 Contact

**Developer:** Shanmathi G
📧 Email: [shanmathigopikrishnan@gmail.com](mailto:shanmathigopikrishnan@gmail.com)
🔗 GitHub: [shanmathigopikrishnan](https://github.com/shanmathigopikrishnan)
🔗 LinkedIn: [Shanmathi G](https://www.linkedin.com/in/shanmathigopikrishnan)

> 💬 *Feel free to reach out for collaborations, improvements, or issues.*
