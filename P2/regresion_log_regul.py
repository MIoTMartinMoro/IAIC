from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures
from pandas.io.parsers import read_csv

def main():
	valores = read_csv('ex2data2.csv', header=None).values  # Lee los valores
	valores = valores.astype(float)  # Los convierte a float

	x = valores[:, 0:2]  # Separa la matriz de valores en x
	y = valores[:, -1:]  # e y

	pinta_valores_iniciales(x, y)
	plt.title('Valores iniciales')
	plt.savefig('Valores_iniciales_regular.png')

	pol = PolynomialFeatures(6)
	x_pol = pol.fit_transform(x)

	ls = [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 15, 20, 30, 90, 200]

	for l in ls:
		Theta = min_coste(x_pol, y, l)
		pinta_frontera_curva(x, y, Theta, pol, l)

def g(z):
	return 1 / (1 + np.exp(-z))

def coste(theta, x, y, l):
	m = len(x)
	coste1 = -(1 / m) * np.sum(y.T * np.log(h(theta, x)) + (1 - y.T) * np.log(1 - h(theta, x)))
	coste2 = (l / (2 * m)) * np.sum(theta[1:]**2)
	coste = coste1 + coste2
	return coste

def h(theta, x):
	return g(np.dot(theta, x.T))

def grad(theta, x, y, l):
	m = len(x)
	columns = len(x.T)
	sumat = [0.] * columns  # Genera un array del tamaño del número de variables donde se le añadirá los sumatorios
	for n in range(columns):  # Por cada columna (variable)
		for j in range(m):  # Y por cada fila (valores distintos de cada variable)
			sumat[n] += (h(theta, x[j]) - y[j]) * x[j, n]  # Calcula el sumatorio con esta ecuación
			if not j == 0:
				sumat[n] += (l / m) * theta[n]  # Si no es theta0, se le suma la regularización
	grad = np.divide(sumat, m)
	return grad

def min_coste(x, y, l):
	m = len(x)
	initialTheta = np.zeros(len(x[0]))  # Tantas thetas como columnas de X

	result = opt.fmin_tnc(func=coste, x0=initialTheta, fprime=grad, args=(x, y, l))
	return result[0]

def pinta_frontera_curva(X, Y, theta, poly, l):
	pinta_valores_iniciales(X, Y)
	x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
	x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

	xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

	h = g(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta))
	h = h.reshape(xx1.shape)

	plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='g')
	plt.title('Lambda={}'.format(l))
	plt.savefig('frontera_regular{}.png'.format(l))
	plt.close()

def pinta_valores_iniciales(x, y):
	#Obtiene un vector con los índices de los ejemplos positivos
	pos = np.where(y == 1)

	#Dibuja los ejemplos positivos
	plt.scatter(x[pos, 0] , x[pos, 1], marker='+', c='k')

	#Obtiene un vector con los índices de los ejemplos negativos
	pos = np.where(y == 0)

	#Dibuja los ejemplos negativos
	plt.scatter(x[pos, 0] , x[pos, 1], marker='o', c='y')
	plt.legend(('1', '0'))

main()
