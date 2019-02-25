from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pandas.io.parsers import read_csv

def h(theta, x):
	return np.dot(theta, x.T)

def main():
	valores = read_csv('ex1data2.csv', header=None).values
	valores = valores.astype(float)

	x = valores[:, 0:-1]
	y = valores[:, -1:]

	a = 0.01

	costes(x, y, a)

def costes(x, y, a):
	m = len(x)
	mediaX = np.mean(x, axis=0)
	stdX = np.std(x, axis=0)
	normalX = (x - mediaX) / stdX
	normalX = np.hstack([np.ones((m, 1)), normalX])

	theta = np.array(np.ones((len(x[0]))))
	costes = []
	thetas_array = []
	for i in range(1500):
		sumat = 0
		for xi, yi in np.ndindex(normalX.shape):
			sumat[yi] += (h(theta, x[xi, :]) - y[xi]) * x[xi, yi]
		theta = theta - (a / m) * sumat
		print(theta)
		costes.append(coste(theta, x, y))
		thetas_array.append(theta)


main()