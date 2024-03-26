from flask import Flask,request,render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
logger.setLevel(logging.DEBUG)



application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/home')
def home():
    return render_template('home.html')


@application.route('/predict_datapoint',methods = ['GET','POST'])
def predict_datapoint():
    print(request.method)
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            logger.debug('Starting prediction...')
            data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
            )
        
            df = data.get_data_as_dataframe()
            #print(df)
            #print("Before Prediction")

            pred_pipeline = PredictPipeline()
            results = pred_pipeline.predict(df)
            #print("after Prediction")
            return render_template('home.html',results=round(results[0],2))
        except Exception as e:
            logger.debug(e)
            
    

    

if __name__=="__main__":
    application.run(host="0.0.0.0", debug=True)
