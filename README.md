# Algoritmo Supervisado: Regresión Lineal

Ejecución del modelo

Para ejecutar el algoritmo se debe seguir las siguientes instrucciones
    1. Instalar Docker.
    2. Ejecutar el siguiente comando
        ◦ “docker run juanra027/ia-regresion-lineal:SecondCommit python /regresionLineal.py <Día que desea saber el número de casos>”
    3. Ejecutar el siguiente comando para obtener el id del contenedor
        ◦ docker ps -a
    4. Ejecutar el siguiente comando para obtener un gráfico
        ◦ “docker cp <id del contenedor>:/Data.png .”
        ◦ Esto guardará la imagen localmente para que la pueda observar.