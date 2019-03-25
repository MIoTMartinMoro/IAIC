from scipy.io  import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures
from pandas.io.parsers import read_csv

def g(z):
	return 1 / (1 + np.exp(-z))

def buildY(y, K):
	m = len(y)
	y_aux = np.zeros((m, K))
	ks = np.eye(K)
	for i in range(m):
		for k in range(K):
			if (y[i][0] == k + 1):
				y_aux[i] = ks[k]
	return y_aux

def coste(h, y, l, theta):
	K = len(h)
	m = len(h[0])
	y = buildY(y, K)
	L = len(theta)
	coste1 = 0
	coste2 = 0
	for k in range(K):
		coste1 += -(1 / m) * np.sum(y.T[k] * np.log(h[k]) + (1 - y.T[k]) * np.log(1 - h[k]))
	for i in range(L):
		coste2 += (l / (2 * m)) * np.sum(theta[i][1:]**2)
	coste = coste1 + coste2
	return coste


def main():
	weights = loadmat('ex3weights.mat')
	theta1, theta2 = weights['Theta1'], weights['Theta2']
	# Theta1 es de dimensión 25x401
	# Theta2 es de dimensión 10x26

	THETA = np.array([theta1, theta2])

	data = loadmat('ex3data1.mat')
	# Se pueden consultar las claves con data.keys()
	y = data['y']
	X = data['X']
	# Almacena los datos leídos en X e y
	l = 0.1
	
	h = getH(X, theta1, theta2)
	cost = coste(h, y, l, THETA)

def getH(X, theta1, theta2):
	m = len(X)
	x = np.hstack([np.ones((m, 1)), X])  # Le añade una columna de unos a las x. Así se puede hacer la función y=theta0 + x*theta1 como un producto escalar

	z2 = np.dot(theta1, x.T)

	a2 = g(z2)

	m = len(a2.T)
	a2 = np.hstack([np.ones((m, 1)), a2.T])

	z3 = np.dot(theta2, a2.T)

	a3 = g(z3)

	return a3

main()