import cmath
import click
from flask import Flask, render_template, request
import numpy as np
import joblib
import json

# Load your trained model here
model = joblib.load('/app/model.pkl')

feat_cols = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
             'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

context_dict = {
    'feats': feat_cols,
    'zip': zip,
    'range': range,
    'len': len,
    'list': list,
}

# Initiate the application
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict_churn():
    if request.method == 'POST':
        CreditScore = (request.form['CreditScore'])
        Geography = (request.form['Geography'])
        Gender = (request.form['Gender'])
        Age = (request.form['Age'])
        Tenure = (request.form['Tenure'])
        Balance = (request.form['Balance'])
        NumOfProducts = (request.form['NumOfProducts'])
        HasCrCard = (request.form['HasCrCard'])
        IsActiveMember = (request.form['IsActiveMember'])
        EstimatedSalary = (request.form['EstimatedSalary'])

        X = np.array([[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])

        # Make prediction
        pred = model.predict(X)
        print('Prediction:', pred)


        # Return prediction as JSON response
        return json.dumps({'prediction': pred.tolist()})
    else:
        # Render the template with input form
        return render_template('index.html', **context_dict)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)