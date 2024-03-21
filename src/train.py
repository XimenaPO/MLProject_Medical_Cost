# IMPORTACIÓN DE LIBRERÍAS

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
from scipy.stats import pearsonr, ttest_ind, f_oneway

# División de datos 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate

# Preprocesamiento de datos
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# Métricas
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, make_scorer

# Búsqueda hiperparámetros
from sklearn.model_selection import GridSearchCV

# Modelos
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
#-----------------------------------------------------------------------------

# INSTANCIAR EN UNA VARIABLE EL DATASET NUMÉRICO
df = pd.read_csv('../src/data/processed/df_numerico_para_F_E.csv')
#-----------------------------------------------------------------------------

# DIVISIÓN DE DATOS
X = df.drop(columns='costo')
y = df['costo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#-----------------------------------------------------------------------------

# TRANSFORMACIONES

# Logaritmo
y_train = np.log(y_train)
y_test = np.log(y_test)

X_train['edad'] = np.log(X_train['edad'])
X_test['edad'] = np.log(X_test['edad'])

# Escalado de datos 
scaler_X = StandardScaler()
columns_to_scale = ['edad', 'hijos', 'imc']
X_train[columns_to_scale] = scaler_X.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler_X.transform(X_test[columns_to_scale])

scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
#-----------------------------------------------------------------------------

# ENTRENAMIENTO DE MODELO CON MEJORES HIPERPARÁMETROS Y PREDICCIONES
# Instanciar modelo con hiperparámetros
my_model = GradientBoostingRegressor(learning_rate=0.1,
                                     max_depth=3, 
                                     n_estimators=50)

# Entrenamiento
my_model.fit(X_train, y_train)

# Predicciones en datos de entrenamiento y prueba
predicciones_train = my_model.predict(X_train)
predicciones_test = my_model.predict(X_test)

# R^2 en datos de entrenamiento y prueba
r2_train = r2_score(y_train, predicciones_train)
r2_test = r2_score(y_test, predicciones_test)
mae_train = mean_absolute_error(y_train, predicciones_train)
mae_test = mean_absolute_error(y_test, predicciones_test)
rmse_train = mean_squared_error(y_train, predicciones_train, squared=False)
rmse_test = mean_squared_error(y_test, predicciones_test, squared=False)


# Guardar el modelo
with open('../models/my_model.pkl', 'wb') as f:
    pickle.dump(my_model, f)
#-----------------------------------------------------------------------------

# REVERSIÓN DE TRANSFORMACIONES
y_test_inverso = np.exp(scaler_y.inverse_transform(predicciones_test.reshape(-1, 1)))

# RESULTADOS PREDICCIONES CON LOS DATOS EN ESCALA ORIGINAL 
predicciones_df = pd.DataFrame(y_test_inverso, columns=['Predicciones'], index=X_test.index)
datos_prueba_con_predicciones = pd.concat([X_test, predicciones_df], axis=1)