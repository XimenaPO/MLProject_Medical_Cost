PROYECTO DE MACHINE LEARNING - REGRESIÓN LINEAL 

Análsis Exploratorio de Datos (EDA) + Proyecto de Machine Learning para predecir mediante modelos el costo médico de un seguro, según características personales.

INTRODUCCIÓN:

El presente proyecto tiene como objetivo predecir los costos médicos según determinadas características de cada persona. 

Dichas predicciones serán realizadas aplicando Machine Learning, y específicamente modelos de Regresión. Anteriormente haré un EDA para analizar previamente el dataset.

Además, en el repositorio se encuentra la presentación del proyecto realizada en PowerPoint, para ser expuesto en el Bootcamp. Por ello, es que se aclara que, la misma solo contiene ideas específicas e importantes, ya que las explicaciones pertinentes las he relatado y explicado de manera oral. 

El DataFrame contiene 1338 filas y 7 columnas, sin valores nulos.

OBJETIVO:

Lo que se busca predecir es el costo médico, es decir, los cargos económicos que tendría una persona según algunos factores como la edad, el sexo, el índice de masa corporal (imc), cantidad de hijos, si es fumadora o no, etc.

EDA:

En cuanto al EDA, vamos a analizar mediante funciones y gráficos la cardinalidad de las variables, observar si hay correlaciones entre las mismas, o la relación es nula, cómo se distribuyen los datos en cada característica, si hay asimetría o distribución normal, entre otras cosas. 

Posteriormente, en base al análisis realizado, haremos la limpieza correspondiente según haya valores nulos, traducciones, duplicados, etc.

Asimismo, se ha realizado un exhaustivo análisis univariante y bivariante con sus  conclusiones y comentarios, agregando pruebas de hipótesis según correspondiera, como el Coeficiente de Correlación de Pearson, Prueba t de Student y Prueba de Anova. 

FEATURE ENGINEERING:

En la etapa de Machine Learning, en primer lugar, realizaremos las transformaciones necesarias a las variables (Feature Engineering) como ser: convertir las variables categóricas en numéricas, utilizando distintos métodos como mapeos, Label Encoder; aplicaremos el logaritmo a las características que correspondan para obtener una distribución normal o escalado mediante Standard Scaler con el fin de tener todas las variables en una misma escala, para un mejor rendimiento del modelo.

MODELOS DE REGRESIÓN - CROSS VALIDATION - GRIDSEARCH PARA BÚSQUEDA DE HIPERPARÁMETROS

Posteriormente, analizaremos cuál modelo de Regresión generaliza mejor, utilizando 'Cross Validation' y también 'GridSearchCV', para encontrar los mejores hiperparámetros respecto a dichos modelos. 

Por último, haremos las predicciones correspondientes en base a los mejores modelos y utilizaremos las métricas que se utilizan en estos problemas de regresión como por ejemplo, el Coeficiente de Determinación (R2), la Raíz del Error Cuadrático Medio (RMSE) y el Error Absoluto Medio (MAE) para saber el rendimiento del modelo y cómo éste generaliza los nuevos datos.

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

R2, RMSE y MAE.


APLICACIÓN FLASK: 

Para finalizar el proyecto, utilicé Flask con el objeto de productivizar mi modelo, lo que significa que lo hice accesible a través de una aplicación web. 
Para ello, también se utilizaron archivos HTML con el fin de renderizar el diseño.

Por último, mediante la app creada con Flask, puede obtenerse la predicción del costo médico utilizando el modelo entrenado. Para ello, debe ingresarse en el formulario que aparece en la web, las características que se solicitan para realizar la predicción. 
