from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pandas.io.parsers import read_csv

def h(theta, x):
	return np.dot(x, theta)

def main():
	valores = read_csv('ex1data2.csv', header=None).values
	valores = valores.astype(float)

	x = valores[:, 0:-1]
	y = valores[:, -1:]

	a = []

	for i in range(2, 5):
		a.append(1 / (10**i))
		a.append(3 / (10**i))

	normalX, mu, sigma = normal(x)

	Theta, coste, a = calcA(normalX, y, a)
	Theta2 = ecuacion_normal(x, y)

	x1 = np.array([1, 1650, 3])

	y1 = h(Theta, x1.T)
	y2 = h(Theta2, x1.T)
	print(y1)
	print(y2)

def ecuacion_normal(x, y):
	m = len(x)
	x = np.hstack([np.ones((m, 1)), x])
	part1 = np.linalg.inv(np.dot(x.T, x))
	part2 = np.dot(part1, x.T)
	theta = np.dot(part2, y)
	return theta

def normal(x):
	mu = np.mean(x, axis=0)
	sigma = np.std(x, axis=0)
	normalX = (x - mu) / sigma
	return (normalX, mu, sigma)

def descenso_gradiente(normalX, y, a):
	m = len(normalX)
	
	normalX = np.hstack([np.ones((m, 1)), normalX])
	theta = np.array(np.ones((len(normalX[0])))).reshape(len(normalX[0]), 1)

	fig = plt.figure()
	ax = fig.gca()

	costes = []
	thetas_array = []
	for i in range(1500):
		columns = len(normalX.T)
		sumat = [0.] * columns
		for n in range(columns):
			for j in range(m):
				sumat[n] += (h(theta, normalX[j, :]) - y[j, :]) * normalX[j, n]
			theta[n] = theta[n] - (a / m) * sumat[n]
		costeJ = coste(theta, normalX, y)
		ax.plot(i, costeJ[0][0], 'bx')
		costes.append(costeJ)
		thetas_array.append(theta)
	plt.savefig('costes{}.png'.format(a))
	return (thetas_array[np.argmin(costes)], np.min(costes))

def coste(theta, x, y):
	m = len(x)
	coste = (1 / (2 * m)) * np.dot((h(theta, x) - y).T, h(theta, x) - y)
	return coste

def calcA(normalX, y, array_a):
	thetas_array = []
	costes_array = []
	for a in array_a:
		Theta, coste = descenso_gradiente(normalX, y, a)
		thetas_array.append(Theta)
		costes_array.append(coste)
	return (thetas_array[np.argmin(costes_array)], np.min(costes_array), array_a[np.argmin(costes_array)])

main()