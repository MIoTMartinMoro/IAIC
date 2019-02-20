import numpy as np
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

def h(theta, x):
	return np.dot(theta, x.T)

def main():
	valores = read_csv('ex1data1.csv', header=None).values
	valores = valores.astype(float)
	m = len(valores)

	v = np.hstack([np.ones((m, 1)), valores])

	x = v[:, 0:2]
	y = v[:, -1:]

	a = 0.01

	Thetas, coste = descenso_gradiente(x, y, a)
	dibuja(x[:, -1], y, Thetas)

def descenso_gradiente(x, y, a):
	m = len(x)
	theta = np.array([1., 1.])
	costes = []
	thetas_array = []
	for i in range(1500):
		sum0 = 0
		sum1 = 0
		for j in range(m):
			sum0 += h(theta, x[j, :]) - y[j]
			sum1 += (h(theta, x[j, :]) - y[j]) * x[j, -1:]
		theta[0] = theta[0] - (a / m) * sum0
		theta[1] = theta[1] - (a / m) * sum1
		costes.append(coste(theta, x, y))
		thetas_array.append(theta)

	return (thetas_array[np.argmin(costes)], np.min(costes))

def coste(theta, x, y):
	m = len(x)
	coste = (1 / (2 * m)) * np.sum((h(theta, x) - y)**2)
	return coste

def dibuja(x, y, theta):
	plt.plot(x, y, 'rx')

	x_h = np.hstack([np.ones((23, 1)), [range(23)]])
	print(x_h)
	plt.plot(x_h[-1:].T, h(theta, x_h))
	plt.show()


main()