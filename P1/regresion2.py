from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pandas.io.parsers import read_csv

def h(theta, x):
	return np.dot(x, theta)  # Hace el producto escalar entre x y theta. y=x0*theta0 + x1*theta1 + x2*theta2...

def main():
	valores = read_csv('ex1data2.csv', header=None).values  # Lee los valores
	valores = valores.astype(float)  # Los convierte a float

	x = valores[:, 0:-1]  # Separa la matriz de valores en x
	y = valores[:, -1:]  # e y

	a = []

	# Añade valores al array de los valores alpha de entrenamiento 
	for i in range(1, 5):
		a.append(1 / (10**i))
		a.append(3 / (10**i))

	normalX, mu, sigma = normal(x)  # Normaliza los valores x y devuelve el valor mu y sigma

	Theta, coste, a = calcA(normalX, y, a)  # Calcula el mejor valor a, el coste más bajo y las thetas que lo provocan (Se necesita normalizar los valores)
	Theta2 = ecuacion_normal(x, y)  # Calcula los valores theta con la ecuación normal

	x1 = np.array([1, 1650, 3])  # Genera un nuevo dato
	x1n = np.array([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]])  # Genera un neuvo dato normalizado

	y1 = h(Theta, x1n.T)  # Calcula el valor de la y con las primeras thetas añadas y el valor normalizado
	y2 = h(Theta2, x1.T)  # Calcula el valor de la y con las segundas thetas añadas y el valor según llega
	print('La casa cuesta {:.2f}€ según el método del descenso de gradiente'.format(y1[0]))
	print('La casa cuesta {:.2f}€ según el método de la ecuación normal'.format(y2[0]))

def ecuacion_normal(x, y):
	m = len(x)
	x = np.hstack([np.ones((m, 1)), x])  # Añade una columna de unos
	part1 = np.linalg.inv(np.dot(x.T, x))  # Hace la inversa del producto escalar entre x transpuesta y x
	part2 = np.dot(part1, x.T)  # Hace el producto escalar entre el valor anterior y x transpuesta
	theta = np.dot(part2, y)  # Hace el producto escalar entre el valor anterior e y
	return theta

def normal(x):
	mu = np.mean(x, axis=0)
	sigma = np.std(x, axis=0)
	normalX = (x - mu) / sigma  # Para normalizr los valores resta la media y divide por la sigma
	return (normalX, mu, sigma)  # Devuelve la matriz normalizada y los valores mu y sigma que se han usado

def descenso_gradiente(normalX, y, a):
	m = len(normalX)
	
	normalX = np.hstack([np.ones((m, 1)), normalX])
	theta = np.array(np.ones((len(normalX[0])))).reshape(len(normalX[0]), 1)  # Genera tantas thetas de valor 1 como variables distintas

	fig = plt.figure()  # Se crea una gráfica que tendrá los costes por cada repetición
	ax = fig.gca()

	costes = []
	thetas_array = []
	for i in range(1500):
		columns = len(normalX.T)
		sumat = [0.] * columns  # Genera un array del tamaño del número de variables donde se le añadirá los sumatorios
		for n in range(columns):  # Por cada columna (variable)
			for j in range(m):  # Y por cada fila (valores distintos de cada variable)
				sumat[n] += (h(theta, normalX[j]) - y[j]) * normalX[j, n]  # Calcula el sumatorio con esta ecuación
			theta[n] = theta[n] - (a / m) * sumat[n]  # Se actualiza el valor de las thetas con esta ecuación
		costeJ = coste(theta, normalX, y)  # Se calcula el coste
		ax.plot(i, costeJ[0][0], 'bx')  # Se añade el punto con el coste
		costes.append(costeJ)  # Se añade el coste al array de costes
		thetas_array.append(theta)  # Se añaden las thetas al array
	plt.savefig('costes{}.png'.format(a))
	# Devuelve el coste mínimo y las thetas que lo provocan
	return (thetas_array[np.argmin(costes)], np.min(costes))

def coste(theta, x, y):
	m = len(x)
	coste = (1 / (2 * m)) * np.dot((h(theta, x) - y).T, h(theta, x) - y)  # Ecuación del coste (Una matriz por su transpuesta es equivante al cuadrado de cada valor)
	return coste

def calcA(normalX, y, array_a):
	thetas_array = []
	costes_array = []
	for a in array_a:  # Por cada valor de alpha que hemos añadido
		Theta, coste = descenso_gradiente(normalX, y, a)  # Calculamos el menor coste y con que thetas se obtiene
		thetas_array.append(Theta)
		costes_array.append(coste)
	# Devuelve el menor coste entre las a, la propia a y los valores theta que lo provocan
	return (thetas_array[np.argmin(costes_array)], np.min(costes_array), array_a[np.argmin(costes_array)])

main()