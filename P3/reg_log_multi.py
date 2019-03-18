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

def coste(theta, x, y, l):
	m = len(x)
	coste1 = -(1 / m) * np.sum(y.T * np.log(h(theta, x)) + (1 - y.T) * np.log(1 - h(theta, x)))
	coste2 = (l / (2 * m)) * np.sum(theta[1:]**2)
	coste = coste1 + coste2
	return coste

def h(theta, x):
	return g(np.dot(theta, x.T))

def classifier(X, Thetas):
	vector = [0] * len(Thetas)
	i = 0
	for theta in Thetas:
		vector[i] = h(theta, X)
		i += 1
	vector[0] = vector[10]
	return np.argmax(vector[:10])

def oneVsAllTrain(X, y, num_etiquetas, reg):
	m = len(X[0])
	Thetas = np.zeros((num_etiquetas + 1, m))
	for etiqueta in range(num_etiquetas + 1):
		if etiqueta == 0:
			continue
		y_aux = (y == etiqueta).astype('int')
		Thetas[etiqueta] = min_coste(X, y_aux, reg)
	return Thetas

def grad(theta, x, y, l):
	m = len(x)
	columns = len(x.T)
	theta_aux = np.array([np.hstack([0, theta[1:]])]).T
	grad = [0.] * columns  # Genera un array del tamaño del número de variables donde se le añadirá los sumatorios
	H = np.array([h(theta, x)]).T
	grad = np.dot(x.T, (H - y)) + (l / m) * theta_aux
	grad = np.divide(grad, m)
	return grad

def min_coste(x, y, l):
	m = len(x)
	initialTheta = np.zeros(len(x[0]))  # Tantas thetas como columnas de X

	result = opt.fmin_tnc(func=coste, x0=initialTheta, fprime=grad, args=(x, y, l))
	return result[0]

def tryExample(X, thetas):
	for i in range(10):
		sample = np.random.choice(X.shape[0], 1)
		plt.imshow(X[sample, :].reshape(-1, 20).T)
		plt.axis('off')
		val = classifier(X[sample, :], thetas)
		plt.title('Se clasifica como: {}'.format(val))
		plt.savefig('ejemplo{}.png'.format(i))
		print('Es un {}'.format(val))

def main():
	data = loadmat('ex3data1.mat')
	# Se pueden consultar las claves con data.keys()
	y = data['y']
	X = data['X']
	# Almacena los datos leídos en X e y

	#pol = PolynomialFeatures(6)
	#x_pol = pol.fit_transform(X)

	l = 0
	sample = np.random.choice(X.shape[0], int(len(X) * 0.7))
	Thetas = oneVsAllTrain(X[sample, :], y[sample, :], 10, l)
	tryExample(X, Thetas)


main()