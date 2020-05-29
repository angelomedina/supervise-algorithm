#Regresion lineal simple 

#Importando librerias
import numpy as np  #Matematica
import matplotlib.pyplot as plt #Graficos
import pandas as pd  #Carga de datos

dataset = pd.read_csv("salary_dataset.csv")
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

##Predice valores
## Usar si existe relaciones entre las variables


#Dividir el dataset en entrenamiento y testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=12)

# La regresion lineal simple no requiere de un escalado

#Realizar el entrenamiento con las variables de entrenamiento x y
from sklearn.linear_model import LinearRegression
regresion = LinearRegression()  #Se crea objeto regresion lineal
regresion.fit(x_train, y_train) #Se entrena el modelo

#Predecir el test
y_pred= regresion.predict(x_test)
t= regresion.predict([[20.5]]) #Prediccion para 20 años y medios

#Visualizar los resultados
#Usando las variables de entrenamiento
plt.scatter(x_train,y_train, color="green")
plt.plot(x_train, regresion.predict(x_train), color="red")
plt.title("Regresion lineal")
plt.xlabel("Años de experiencias")
plt.ylabel("Salario")
plt.show()
















