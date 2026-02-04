from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for Home Page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=request.form.get('reading_score'),
                writing_score=request.form.get('writing_score'),
            )

            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:")
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            print("Prediction Result:")
            print(results)

            return render_template('home.html', results=round(results[0], 2))
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return render_template('home.html', results=f"Error: {str(e)}")


if __name__=="__main__":
    app.run('0.0.0.0',debug=True)