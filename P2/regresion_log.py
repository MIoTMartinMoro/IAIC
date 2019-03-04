from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.optimize as opt
from pandas.io.parsers import read_csv

def main():
	valores = read_csv('ex2data1.csv', header=None).values  # Lee los valores
	valores = valores.astype(float)  # Los convierte a float

	x = valores[:, 0:2]  # Separa la matriz de valores en x
	y = valores[:, -1:]  # e y

	#Obtiene un vector con los índices de los ejemplos positivos
	pos = np.where(y == 1)

	#Dibuja los ejemplos positivos
	plt.scatter(x[pos, 0] , x[pos, 1], marker='+', c='k')

	#Obtiene un vector con los índices de los ejemplos negativos
	pos = np.where(y == 0)

	#Dibuja los ejemplos negativos
	plt.scatter(x[pos, 0] , x[pos, 1], marker='o', c='y')
	plt.legend(('admitido', 'no admitido'))
	plt.savefig('Valores_iniciales.png')

	Theta = min_coste(x, y)

	pinta_frontera_recta(x, y, Theta)
	eval(Theta, x, y)

def g(z):
	return 1 / (1 + np.exp(-z))

def coste(theta, x, y):
	m = len(x)
	array = y * np.log(h(theta, x)) + (1 - y) * np.log(1 - h(theta, x))
	coste = -(1 / m) * np.sum(y.T * np.log(h(theta, x)) + (1 - y.T) * np.log(1 - h(theta, x)))
	return coste

def h(theta, x):
	return g(np.dot(theta, x.T))

def grad(theta, x, y):
	m = len(x)
	columns = len(x.T)
	sumat = [0.] * columns  # Genera un array del tamaño del número de variables donde se le añadirá los sumatorios
	for n in range(columns):  # Por cada columna (variable)
		for j in range(m):  # Y por cada fila (valores distintos de cada variable)
			sumat[n] += (h(theta, x[j]) - y[j]) * x[j, n]  # Calcula el sumatorio con esta ecuación
	grad = np.divide(sumat, m)
	return grad

def min_coste(x, y):
	m = len(x)
	initialTheta = np.zeros(3)
	x = np.hstack([np.ones((m, 1)), x])

	result = opt.fmin_tnc(func=coste, x0=initialTheta, fprime=grad, args=(x, y))
	return result[0]

def pinta_frontera_recta(X, Y, theta):
	x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
	x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

	h = g(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(theta))
	h = h.reshape(xx1.shape)

	# el cuarto parámetro es el valor de z cuya frontera se
	# quiere pintar
	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
	plt.savefig('frontera.png')
	plt.close()

def eval(Theta, x, y):
	m = len(x)
	x = np.hstack([np.ones((m, 1)), x])
	pred = h(Theta, x)
	count = 0

	for i in range(m):
		if (pred.T[i] >= 0.5 and y[i] == 1) or (pred.T[i] < 0.5 and y[i] == 0):
			count += 1

	print('Hay un {}% de aciertos'.format((count/m)*100))

main()