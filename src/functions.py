import pandas as pd
import numpy as np
import yaml

# load yaml main file with parameters
with open("../config/config.yaml", "r") as file:
    try:
        path = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
FEATS = path['FEATS']
TARGET = path['TARGET']

# handle function for TotalCharges column
def handle_strnumber(point):
  if type(point) == str:
    try:
      point = float(point)
      return point
    except:
      return 'dropme'

# initial preprocess step with mapping
def preprocess(data) -> pd.DataFrame:
  X = data.copy()

  X['TotalCharges'] = X['TotalCharges'].apply(lambda x: handle_strnumber(x))
  X = X[X['TotalCharges'] != 'dropme']
  X['TotalCharges'] = X['TotalCharges'].astype(float)
  X['Churn'] = X['Churn'].map({'Yes':1, 'No':0})
  X['gender'] = X['gender'].map({'Male':1, 'Female':0})
  X['Partner'] = X['Partner'].map({'Yes':1, 'No':0})
  X['Dependents'] = X['Dependents'].map({'Yes':1, 'No':0})
  X['TechSupport'] = X['TechSupport'].map({'Yes':1, 'No internet service':0, 'No': -1})
  X['Contract'] = X['Contract'].map({'Month-to-month':1, 'Two year':0, 'One year': -1})
  X['PaperlessBilling'] = X['PaperlessBilling'].map({'Yes':1, 'No':0})

  return X[FEATS], X[TARGET].to_frame()

# KMeans clustering feature engineering labels column
def preprocess2(data, pipeline, kmeans, fit = True) -> np.array:
  """
  <Docstring>
  Remember to first apply this function with fit param = True
  on the training set.
  """
  X = data.copy()
  if fit:
    scld = pipeline.fit_transform(X)
    scld = pd.DataFrame(scld, columns = ['PC1','PC2'])
    labels = kmeans.fit_predict(scld)
  else:
    scld = pipeline.transform(X)
    scld = pd.DataFrame(scld, columns = ['PC1','PC2'])
    labels = kmeans.predict(scld)
  return labels

# Standardizing only numeric features
def preprocess3(data, scaler, fit = True) -> pd.DataFrame:
  X = data.copy()
  cols = ['SeniorCitizen','tenure','TotalCharges']

  if fit:
    numerics = scaler.fit_transform(X[cols].copy())
  else:
    numerics = scaler.transform(X[cols].copy())
  X[cols] = numerics
  return X

if __name__ == '__main__':
  print('This is an import function!')