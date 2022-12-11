#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jraposoneto
"""
import yaml
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request
import sys
sys.path.append('../')
from src.functions import preprocess2, preprocess3

import h2o
import mlflow
from mlflow import MlflowClient
import mlflow.h2o

# load yaml main file with parameters
with open("../config/config.yaml", "r") as file:
    try:
        path = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
FEATS = path['FEATS']
MODEL = path['model']

# loading trained payloads
pipeline = joblib.load(path['pipeline']['path'])
kmeans = joblib.load(path['kmeans']['path'])
scaler = joblib.load(path['scaler']['path'])

# create Flask App
app = Flask(__name__)

# Initiate H2O instance and MLflow client
h2o.init()

# initiate MlFlow client
client = MlflowClient()

# loading best model
model = mlflow.h2o.load_model(MODEL)


@app.route('/', methods = ['GET','POST'])
def run_model():

    if request.method == 'GET':
        return ''

    elif request.method == 'POST':

        # loading json data
        data = request.get_json()
        a = data['gender']
        b = data['Partner']
        c = data['Dependents']
        d = data['TechSupport']
        e = data['Contract']
        f = data['PaperlessBilling']
        g = data['SeniorCitizen']
        h = data['tenure']
        i = data['TotalCharges']

        # input load payload data
        load = np.array([[
            a, b, c, d, e, f, g, h, i]])
        
        # convert to dataframe
        X = pd.DataFrame(load, columns = FEATS)
        
        # preprocess step 2
        X['Cluster'] = preprocess2(X, pipeline, kmeans, fit = False) # already fitted
        
        # preprocess step 3: scale numeric cols
        X = preprocess3(X, scaler)
        
        X = h2o.H2OFrame(X)

        # predicoes
        pred = model.predict(X)
        pred = pred.as_data_frame()['Yes'].values

        return {"Churn": float(pred[0])}

if __name__ == '__main__':
    app.run(debug = False)
