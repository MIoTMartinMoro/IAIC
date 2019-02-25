from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
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
	dibuja_3d(x, y)

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
	x_range = range(23)
	x_matriz = np.array(x_range).reshape(23, 1)
	x_h = np.hstack([np.ones((23, 1)), x_matriz])
	plt.plot(x_range, h(theta, x_h))
	plt.savefig('regresion.png')

def dibuja_3d(x, y):
	theta0 = np.arange(-5, 5, 0.1)
	theta1 = np.arange(-5, 5, 0.1)
	theta0, theta1 = np.meshgrid(theta0, theta1)
	costes = np.empty_like(theta0)
	
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	for xi, yi in np.ndindex(theta0.shape):
		theta = np.array([theta0[xi, yi], theta1[xi, yi]])
		costes[xi, yi] = coste(theta, x, y)
	# Plot the surface.
	surf = ax.plot_surface(theta0, theta1, costes, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.savefig('3d.png')

main()