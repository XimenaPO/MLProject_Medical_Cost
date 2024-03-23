import os
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Función para cargar datos
def cargar_datos(ruta):
    df = pd.read_csv(ruta, index_col=0)
    X = df.drop('costo', axis=1)
    X.columns = ['edad', 'imc', 'hijos', 'fumador', 'es_femenino']  # Agregar nombres de características
    y = df['costo']
    return X, y

# Función para preprocesar datos
def preprocesar_datos(X_train, y_train):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = X_train.copy()
    X_train_scaled['edad'] = np.log(X_train_scaled['edad'])
    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train_scaled), columns=X_train.columns)  # Corregir esta línea

    y_train_scaled = np.log(y_train)
    y_train_scaled = scaler_y.fit_transform(y_train_scaled.values.reshape(-1, 1)).flatten()

    return X_train_scaled, y_train_scaled, scaler_X, scaler_y

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train):
    modelo = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=50)
    modelo.fit(X_train, y_train)
    return modelo

# Función para guardar el modelo
def guardar_modelo(modelo, ruta):
    with open(ruta, 'wb') as f:
        pickle.dump(modelo, f)

# Función para guardar los transformadores
def guardar_transformadores(scaler_X, scaler_y, ruta):
    with open(ruta, 'wb') as f:
        pickle.dump((scaler_X, scaler_y), f)

# Función para hacer predicciones y calcular métricas
def evaluar_modelo(modelo, X_test, y_test, scaler_X, scaler_y):
    X_test_scaled = X_test.copy()
    X_test_scaled['edad'] = np.log(X_test_scaled['edad'])
    X_test_scaled = scaler_X.transform(X_test_scaled)
    y_test_scaled = np.log(y_test)
    y_test_scaled = scaler_y.transform(y_test_scaled.values.reshape(-1, 1)).flatten()
    
    y_test_pred = modelo.predict(X_test_scaled)
    
    test_r2 = r2_score(y_test, np.exp(scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))))
    
    return test_r2

# Obtener el directorio del script actual
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta al archivo CSV utilizando os.path.join()
ruta_datos = os.path.join(script_dir, '..', 'data', 'processed', 'df_numerico_para_F_E.csv')

# Cargar datos
X, y = cargar_datos(ruta_datos)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesar datos
X_train_scaled, y_train_scaled, scaler_X, scaler_y = preprocesar_datos(X_train, y_train)

# Entrenar modelo
modelo = entrenar_modelo(X_train_scaled, y_train_scaled)

# Guardar modelo en la misma ubicación que produccion.py
ruta_modelo = os.path.join(script_dir, 'mi_modelo_regresion.pkl')
guardar_modelo(modelo, ruta_modelo)

# Guardar transformadores en la misma ubicación que produccion.py
ruta_transformadores = os.path.join(script_dir, 'transformadores.pkl')
guardar_transformadores(scaler_X, scaler_y, ruta_transformadores)

# Evaluar modelo
test_r2 = evaluar_modelo(modelo, X_test, y_test, scaler_X, scaler_y)

# Imprimir resultados
print("Scores:")
print("Test R2:", test_r2)
