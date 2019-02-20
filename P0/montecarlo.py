# Código de Alda Martín Muñoz y Carlos Moro García

import matplotlib.pyplot as plt
import random
import time
import numpy as np

def fun(x):
    return x * x

def integra_mc(fun, a, b, num_puntos=10000):
    start_time = time.process_time()  # Momento en el que empieza a ejecutarse la función
    valuesX = []  # Valores de X para la curva
    valuesY = []  # Valores de Y para la curva
    puntosX = []  # Valores de X de los puntos
    puntosY = []  # Valores de Y de los puntos
    num_puntos_abajo = 0  # Número de puntos que están por debajo de la curva
    minValue = fun(a)  # Valor mínimo de la función entre 'a' y 'b' (Como valor inicial se le asigan el primer valor de la función)
    maxValue = fun(a)  # Valor máximo de la función entre 'a' y 'b' (Como valor inicial se le asigan el primer valor de la función)
    for x in range(a, b + 1):  # Para todos los valores entre 'a' y 'b'
        y = fun(x)  # Calculamos el valor de la 'y'
        valuesX.append(x)  # Y la añadimos tanto la 'x' como la 'y' al array con los valores
        valuesY.append(y)
        # Si el valor está por debajo del mínimo o por encima del máximo este valor se convierte en el nuevo min/max
        if y < minValue:
            minValue = y
        if y > maxValue:
            maxValue = y
    #plt.plot(valuesX, valuesY)  # Dibujamos la función
    #plt.axis([a, b, minValue, maxValue])  # Fijamos los ejes
    # Calcular aleatoriamente puntos
    for n in range(num_puntos):
        # Calculamos de manera uniforme un valor aleatorio para 'x' entre 'a' y 'b'
        # y un valor para 'y' entre 'minValue' y 'maxValue'
        x = random.uniform(a, b)
        y = random.uniform(minValue, maxValue)
        # Y se guardan en sus respectivos arrays
        puntosX.append(x)
        puntosY.append(y)
        # Si el valor de la curva en ese punto es mayor al de 'y' generado, se suma 1 al número de puntos que hay por debajo
        if fun(x) > y:
            num_puntos_abajo += 1

    #plt.plot(puntosX, puntosY, 'rx')  # Se dibujan los puntos (r: color rojo; x: con forma de 'x')
    #print("El área con bucle es: {}".format((num_puntos_abajo/num_puntos)*(b - a)*maxValue))  # Se calcula el área con la fórmula de montecarlo
    #print("Bucle tarda {} s".format(time.process_time() - start_time))  # Se calcula el tiempo que ha tardado
    return time.process_time() - start_time
    #plt.show()  # Se muestra todo

def integra_mc_vector(fun, a, b, num_puntos=10000):
    start_time = time.process_time()  # Momento en el que empieza a ejecutarse la función
    valuesX = []  # Valores de X para la curva
    valuesY = []  # Valores de Y para la curva
    puntosX = []  # Valores de X de los puntos
    puntosY = []  # Valores de Y de los puntos
    num_puntos_abajo = 0  # Número de puntos que están por debajo de la curva
    minValue = 0  # Valor mínimo de la función entre 'a' y 'b' (Como valor inicial se le asigan el primer valor de la función)
    maxValue = 0  # Valor máximo de la función entre 'a' y 'b' (Como valor inicial se le asigan el primer valor de la función)
    valuesX = np.arange(a, b + 1)
    valuesY = fun(valuesX)
    maxValue = valuesY.max()
    minValue = valuesY.min()
    # Calcular aleatoriamente puntos
    puntosX = np.random.uniform(a, b, (1, num_puntos))
    puntosY = np.random.uniform(minValue, maxValue, (1, num_puntos))
    num_puntos_abajo = np.sum(fun(puntosX) > puntosY)

    return time.process_time() - start_time

def compara_time(fun, a, b):
    sizes = np.linspace(100, 1000000, 20)
    time_bucle = []
    time_vector = []

    for size in sizes:
        time_bucle += [integra_mc(fun, a, b, int(size))]
        time_vector += [integra_mc_vector(fun, a, b, int(size))]

    plt.figure()
    plt.scatter(sizes, time_bucle, c='red', label='bucle')
    plt.scatter(sizes, time_vector, c='blue', label='vector')
    plt.legend()
    plt.savefig('time.png')

compara_time(fun, 1, 10)