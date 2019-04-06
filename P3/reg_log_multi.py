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
	# Calcula el valor de la función para cada etiqueta
	for theta in Thetas:
		vector[i] = h(theta, X)
		i += 1
	vector[0] = vector[10]  # La posición 10 guarda el valor de 0, por lo que lo cambiamos
	return np.argmax(vector[:10])  # Devuelve la posición (la etiqueta) del valor máximo de salida de la función

def oneVsAllTrain(X, y, num_etiquetas, reg):
	m = len(X[0])
	Thetas = np.zeros((num_etiquetas + 1, m))
	for etiqueta in range(num_etiquetas + 1):  # Pone el +1 porque la posición 0 no tiene valor
		if etiqueta == 0:  # Por lo que se la salta
			continue
		y_aux = (y == etiqueta).astype('int')  # Convierta la posición que nos interesa en 1 y el resto en 0s
		Thetas[etiqueta] = min_coste(X, y_aux, reg)  # Calcula el coste de que sea esa etiqueta
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

def evalue(x, thetas, y):
	count = 0  # Número de aciertos
	m = len(x)
	for i in range(m):
		val = classifier(x[i], thetas)
		if (val == y[i]):  # Lo clasifica como "val" y debería ser "y[i]". Si coincide es un acierto
			count += 1
	print('Acierta el {}% de los números'.format((count / m) * 100))  # Se imprime el porcentaje de aciertos

def tryExample(X, thetas):
	for i in range(10):  # Prueba 10 ejemplos aleatorios
		sample = np.random.choice(X.shape[0], 1)
		plt.imshow(X[sample, :].reshape(-1, 20).T)
		plt.axis('off')
		val = classifier(X[sample, :], thetas)  # Los clasifica
		plt.title('Se clasifica como: {}'.format(val))
		plt.savefig('ejemplo{}.png'.format(i))  # y los pinta

def main():
	data = loadmat('ex3data1.mat')
	# Se pueden consultar las claves con data.keys()
	y = data['y']
	X = data['X']
	# Almacena los datos leídos en X e y

	l = 0.1
	Thetas = oneVsAllTrain(X, y, 10, l)
	tryExample(X, Thetas)
	evalue(X, Thetas, y)


main()