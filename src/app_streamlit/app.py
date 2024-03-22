import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar datos
@st.cache
def cargar_datos(ruta):
    df = pd.read_csv(ruta, index_col=0)
    X = df.drop('costo', axis=1)
    y = df['costo']
    return X, y

# Preprocesar datos
def preprocesar_datos(X_train, X_test, y_train, y_test):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

# Entrenar modelo
def entrenar_modelo(X_train, y_train):
    modelo = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=50)
    modelo.fit(X_train, y_train)
    return modelo

# Guardar modelo
def guardar_modelo(modelo, ruta):
    with open(ruta, 'wb') as f:
        pickle.dump(modelo, f)

# Cargar modelo
def cargar_modelo(ruta):
    with open(ruta, 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# Realizar predicción
def hacer_prediccion(modelo, datos):
    return modelo.predict(datos)

# Cargar datos
ruta_datos = r'C:\Users\Ximena\Documents\INFORMÁTICA\Bootcamp_Data_Science\MLProject_Medical_Cost\src\data\processed\df_numerico_para_F_E.csv'
X, y = cargar_datos(ruta_datos)

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesar datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = X_train.copy()  # Copiar datos originales
X_train_scaled['edad'] = np.log(X_train_scaled['edad'])  # Aplicar logaritmo a 'edad'
X_train_scaled = scaler_X.fit_transform(X_train_scaled)  # Escalar datos

X_test_scaled = X_test.copy()  # Copiar datos originales
X_test_scaled['edad'] = np.log(X_test_scaled['edad'])  # Aplicar logaritmo a 'edad'
X_test_scaled = scaler_X.transform(X_test_scaled)  # Escalar datos

y_train_scaled = np.log(y_train)  # Aplicar logaritmo a 'costo'
y_train_scaled = scaler_y.fit_transform(y_train_scaled.values.reshape(-1, 1)).flatten()  # Escalar datos

# Entrenar modelo
modelo = entrenar_modelo(X_train_scaled, y_train_scaled)

# Guardar modelo
ruta_modelo = 'mi_modelo_regresion.pkl'
guardar_modelo(modelo, ruta_modelo)

# Interfaz de usuario con Streamlit
st.title('Predicción de Costo Médico')

# Inputs de usuario
edad = st.number_input('Edad:', min_value=0, max_value=120, value=30)
imc = st.number_input('IMC (Índice de Masa Corporal):', min_value=10.0, max_value=50.0, value=25.0)
hijos = st.number_input('Cantidad de Hijos:', min_value=0, max_value=10, value=0)

# Convertir selecciones a valores numéricos
fumador = st.selectbox('¿Fuma?', ('Sí', 'No'))
es_femenino = st.selectbox('¿Es mujer?', ('Sí', 'No'))
fumador = 1 if fumador == 'Sí' else 0
es_femenino = 1 if es_femenino == 'Sí' else 0

# Realizar predicción cuando se presiona el botón
if st.button('Predecir Costo Médico'):
    datos = np.array([[edad, imc, hijos, fumador, es_femenino]])
    datos[:, 0] = np.log(datos[:, 0])  # Aplicar logaritmo a 'edad'
    datos = scaler_X.transform(datos)  # Escalar datos
    costo_predicho = hacer_prediccion(modelo, datos)
    costo_predicho = np.exp(scaler_y.inverse_transform(costo_predicho.reshape(-1, 1))[0])  # Revertir transformaciones
    st.success(f'El costo médico predicho es: {costo_predicho}')
