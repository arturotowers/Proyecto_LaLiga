# Importamos las librerias necesarias
import pandas as pd
import re
import pickle
import mlflow
import dagshub
import pathlib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from prefect import task, flow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_excel('LaLiga_Dataset_2023_2024.xlsx')

X = df[['Día','Sedes','Edad(opp)','Pos.(opp)', 'Ass(opp)', 'TPint(opp)',
  'PrgC(opp)', 'PrgP(opp)','% de TT(opp)', 'Dist(opp)', '% Cmp(opp)', 'Dist. tot.(opp)','TklG(opp)', 'Int(opp)',
  'Err(opp)', 'RL(opp)', 'PG(opp)', 'PE(opp)','PP(opp)', 'GF(opp)', 'GC(opp)', 'xG(opp)', 'xGA(opp)','Últimos 5(opp)',
  'Máximo Goleador del Equipo(opp)', 'Edad(tm)', 'Pos.(tm)', 'Ass(tm)', 'TPint(tm)', 'PrgC(tm)', 'PrgP(tm)',
  '% de TT(tm)', 'Dist(tm)', '% Cmp(tm)', 'Dist. tot.(tm)', 'TklG(tm)','Int(tm)', 'Err(tm)', 'RL(tm)', 'PG(tm)',
  'PE(tm)', 'PP(tm)', 'GF(tm)','GC(tm)', 'xG(tm)', 'xGA(tm)', 'Últimos 5(tm)','Máximo Goleador del Equipo(tm)']]
y = df['Resultado']

# Dividimos en conjuntos de entrenamiento y prueba
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=15)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)