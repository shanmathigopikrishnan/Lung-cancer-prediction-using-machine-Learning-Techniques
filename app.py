from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("lung_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [float(request.form[key]) for key in request.form]
    
    # Convert to 2D array
    input_data = np.array([features])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    return render_template("index.html", prediction_text=f"Predicted Lung Cancer Level: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
