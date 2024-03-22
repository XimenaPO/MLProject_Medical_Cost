import warnings
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

# Cargar modelo
def cargar_modelo(ruta):
    with open(ruta, 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# Cargar transformadores de datos
def cargar_transformadores(ruta):
    with open(ruta, 'rb') as f:
        scaler_X, scaler_y = pickle.load(f)
    return scaler_X, scaler_y

# Preprocesar entrada de usuario
def preprocesar_entrada(datos, scaler_X):
    datos[:, 0] = np.log(datos[:, 0])  
    datos = scaler_X.transform(datos)  
    return datos

# Revertir transformación de salida
def revertir_transformacion_salida(prediccion, scaler_y):
    prediccion = np.exp(scaler_y.inverse_transform(prediccion.reshape(-1, 1))[0])  
    return prediccion

# Definir ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Definir ruta para manejar las solicitudes de predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario
    edad = float(request.form['edad'])
    imc = float(request.form['imc'])
    hijos = int(request.form['hijos'])
    fumador = int(request.form['fumador'])
    es_femenino = int(request.form['es_femenino'])

    # Preprocesar la entrada
    datos = np.array([[edad, imc, hijos, fumador, es_femenino]])
    datos = preprocesar_entrada(datos, scaler_X)

    # Asignar nombres de características a los datos de entrada
    datos_con_nombres = pd.DataFrame(datos, columns=['edad', 'imc', 'hijos', 'fumador', 'es_femenino'])

    # Realizar la predicción
    costo_predicho = modelo.predict(datos_con_nombres)

    # Revertir la transformación en la salida
    costo_predicho = revertir_transformacion_salida(costo_predicho, scaler_y)

    # Mostrar el resultado de la predicción
    return render_template('predict.html', costo_predicho=costo_predicho)

if __name__ == '__main__':
    ruta_modelo = os.path.join(os.path.dirname(__file__), 'mi_modelo_regresion.pkl')
    ruta_transformadores = os.path.join(os.path.dirname(__file__), 'transformadores.pkl')
    
    modelo = cargar_modelo(ruta_modelo)
    scaler_X, scaler_y = cargar_transformadores(ruta_transformadores)
    
    app.run(debug=True)
