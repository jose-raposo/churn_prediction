# script to extract, transform and load raw data
# creator: jraposoneto
# sklearn version: 1.0.2

import pandas as pd
import numpy as np
import joblib
import yaml
from functions import preprocess, preprocess2, preprocess3
from sklearn.model_selection import train_test_split

# load yaml main file with parameters
with open("../config/config.yaml", "r") as file:
    try:
        path = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# config.yaml payloads
FEATS = path['FEATS']
TARGET = path['TARGET']
RAW = path['raw']['path']
DROP = path['DROP_ID']

# loading trained models
pipeline = joblib.load(path['pipeline']['path'])
kmeans = joblib.load(path['kmeans']['path'])
scaler = joblib.load(path['scaler']['path'])

def etl() -> pd.DataFrame:
    # loading and separating test set
    data = pd.read_csv(RAW).drop(DROP, 1)
    
    # 70% 30% train test split
    train, test = data.iloc[:int(data.shape[0]*.7), :], data.iloc[int(data.shape[0]*.7):, :]
    
    # preprocess step 1
    X, y = preprocess(train)
    Xtest, ytest = preprocess(test)
    
    # data splitting inside train dataset
    Xtrain, Xval, ytrain, yval = train_test_split(
        X, y, stratify = y,
        test_size = .3, random_state = 777
    )
    
    # preprocess step 2: create kmeans labels
    Xtrain['Cluster'] = preprocess2(Xtrain, pipeline, kmeans, fit = True)
    Xval['Cluster'] = preprocess2(Xval, pipeline, kmeans, fit = False)
    Xtest['Cluster'] = preprocess2(Xtest, pipeline, kmeans, fit = False)
    
    # preprocess step 3: scale numeric cols
    Xtrain = preprocess3(Xtrain, scaler)
    Xval = preprocess3(Xval, scaler)
    Xtest = preprocess3(Xtest, scaler)
    
    Xtrain.to_parquet(path['preprocessed']['Xtrain'], index = True)
    ytrain.to_parquet(path['preprocessed']['ytrain'], index = True)
    Xval.to_parquet(path['preprocessed']['Xval'], index = True)
    yval.to_parquet(path['preprocessed']['yval'], index = True)
    Xtest.to_parquet(path['preprocessed']['Xtest'], index = True)
    ytest.to_parquet(path['preprocessed']['ytest'], index = True)
    
    print('====FINISH!=====')

if __name__ == '__main__':
    etl()
