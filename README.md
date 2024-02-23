# MLProyect_Medical_Cost
Proyecto de Machine Learning para predecir mediante modelos el costo médico de un seguro, según características personales.

INTRODUCCIÓN
El presente proyecto tiene como objetivo predecir los costos médicos según determinadas características de cada persona. 

Dichas predicciones serán realizadas aplicando Machine Learning, y específicamente modelos de Regresión. Anteriormente haré un EDA para analizar previamente el dataset.

El DataFrame contiene 1338 filas y 7 columnas, sin valores nulos.

OBJETIVO
Lo que se busca predecir es el costo médico, es decir, los cargos económicos que tendría una persona según algunos factores como la edad, el sexo, el índice de masa corporal (imc), cantidad de hijos, si es fumadora o no, etc.

EDA
En cuanto al EDA, vamos a analizar mediante funciones y gráficos la cardinalidad de las variables, observar si hay correlaciones entre las mismas, o la relación es nula, cómo se distribuyen los datos en cada característica, si hay asimetría o distribución normal, entre otras cosas. 

Posteriormente, en base al análisis realizado, haremos la limpieza correspondiente según haya valores nulos, traducciones, duplicados, etc.

FEATURE ENGINEERING
Y por último, en la etapa de Machine Learning, en primer lugar, realizaremos las transformaciones necesarias a las variables (Feature Engineering) como ser: convertir las variables categóricas en numéricas, utilizando distintos métodos como mapeos, Label Encoder; aplicaremos el logaritmo a las características que correspondan para obtener una distribución normal o escalado mediante Standard Scaler con el fin de tener todas las variables en una misma escala, para un mejor rendimiento del modelo.

MODELOS DE REGRESIÓN - CROSS VALIDATION - GRIDSEARCH PARA BÚSQUEDA DE HIPERPARÁMETROS.
Posteriormente, analizaremos cuál modelo de Regresión generaliza mejor, utilizando 'Cross Validation' y también 'GridSearch', para encontrar los mejores hiperparámetros respecto a dichos modelos. 

Por último, haremos las predicciones correspondientes en base a los mejores modelos y utilizaremos las métricas que se utilizan en estos problemas de regresión como por ejemplo, el Coeficiente de Determinación (R2), la Raíz del Error Cuadrático Medio (RMSE), para saber el rendimiento del modelo y cómo éste generaliza los nuevos datos.

MODELOS DE APRENDIZAJE SUPERVISADO UTILIZADOS: 
LinearRegression()
Ridge()
SVR(),
DecisionTreeRegressor(),
RandomForestRegressor(),
KNeighborsRegressor(),
GradientBoostingRegressor(),
XGBRegressor(),
LGBMRegressor(),
CatBoostRegressor()

MÉTRICAS UTILIZADAS: 
R2, RMSE.

