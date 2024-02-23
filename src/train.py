# Tratamiento de datos
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

import pickle

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Correlaciones - Prueba de hipótesis
from scipy.stats import pearsonr

# Preprocesamiento de datos
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.utils import resample

# Modelos
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, make_scorer

from sklearn.model_selection import GridSearchCV

df = pd.read_csv('../src/data/processed/df_numerico_para_F_E.csv')

X = df.drop(columns='costo')
y = df['costo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = np.log(y_train)
y_test = np.log(y_test)

X_train['edad'] = np.log(X_train['edad'])
X_test['edad'] = np.log(X_test['edad'])

scaler_X = StandardScaler()
columns_to_scale = ['edad', 'hijos', 'imc']
X_train[columns_to_scale] = scaler_X.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler_X.transform(X_test[columns_to_scale])

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

modelo_gradientboosting = GradientBoostingRegressor()
parametros = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}
grid_search = GridSearchCV(modelo_gradientboosting, parametros, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

mejor_modelo_gb = grid_search.best_estimator_
with open('../models/my_model.pkl', 'wb') as f:
    pickle.dump(mejor_modelo_gb, f)

predicciones_train = mejor_modelo_gb.predict(X_train)
predicciones_test = mejor_modelo_gb.predict(X_test)
r2_train = r2_score(y_train, predicciones_train)
r2_test = r2_score(y_test, predicciones_test)
print('R^2 en datos de entrenamiento:', r2_train)
print('R^2 en datos de prueba:', r2_test)

my_model = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=50)
my_model.fit(X_train, y_train)
predicciones_train = my_model.predict(X_train)
predicciones_test = my_model.predict(X_test)
r2_train = r2_score(y_train, predicciones_train)
r2_test = r2_score(y_test, predicciones_test)
print('R^2 en datos de entrenamiento:', r2_train)
print('R^2 en datos de prueba:', r2_test)

y_test_inverso = np.exp(scaler_y.inverse_transform(predicciones_test.reshape(-1, 1)))

predicciones_df = pd.DataFrame(y_test_inverso, columns=['Predicciones'], index=X_test.index)
datos_prueba_con_predicciones = pd.concat([X_test, predicciones_df], axis=1)