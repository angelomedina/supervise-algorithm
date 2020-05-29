#Regresion lineal simple 

#Importando librerias
import numpy as np  #Matematica
import matplotlib.pyplot as plt #Graficos
import pandas as pd  #Carga de datos
import sys

dataset = pd.read_csv("covid19CR.csv")
x= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

##Predice valores
## Usar si existe relaciones entre las variables


#Dividir el dataset en entrenamiento y testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=12)

# La regresion lineal simple no requiere de un escalado

#Realizar el entrenamiento con las variables d e entrenamiento x y
from sklearn.linear_model import LinearRegression
regresion = LinearRegression() #Se crea objeto regresion lineal
regresion.fit(x_train, y_train) #Se entrena el modelo

#Predecir el test
y_pred= regresion.predict(x_test)
t= regresion.predict([[int(sys.argv[1])]]) #Prediccion para el día 100 del covid-19 en Costa Rica 

#Visualizar los resultados
##Usando las variables de entrenamiento
print(t);
plt.scatter(x_train,y_train, color="green")
plt.plot(x_train, regresion.predict(x_train), color="red")
plt.title("Regresion lineal")
plt.xlabel("Dia")
plt.ylabel("Casos Nuevos")
#plt.show()
plt.savefig("Data.png")















