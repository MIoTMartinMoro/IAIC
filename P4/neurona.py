from keras.utils.np_utils import to_categorical
from scipy.io  import loadmat
import math
import numpy as np
import scipy.optimize as opt

from checkNNGradients import checkNNGradients
from displayData import displayData

def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
	"""
	Devuelve el coste y el gradiente de una red neuronal de dos capas.
	"""
	# Calcular coste
	cost = coste(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg)

	# Calcular gradiente
	gradi = grad(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg)
	return (cost, gradi)

def coste(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, l):
	theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
	theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

	m = len(X)
	cost1 = 0
	cost2 = 0
	cost3 = 0
	for i in range(m):
		for k in range(num_etiquetas):
			cost1 += y[i][k] * np.log(h(X[i], theta1, theta2)[k]) + (1 - y[i][k]) * np.log(1 - h(X[i], theta1, theta2)[k])
	cost1 = - cost1 / m

	if l != 0:
		for j in range(1, num_ocultas):
			for k in range(1, num_entradas):
				cost2 += (theta1[j][k])**2
		cost2 = (l * cost2) / (2 * m)
		for j in range(1, num_etiquetas):
			for k in range(1, num_ocultas):
				cost3 += (theta2[j][k])**2
		cost3 = (l * cost3) / (2 * m)

	return cost1 + cost2 + cost3

def evaluar(h, y):
	m = len(h.T)
	cont = 0
	for i in range(m):
		if (np.argmax(h.T[i]) + 1) == y[i, 0]:
			cont += 1
	print('Acierta el {}%\n'.format((cont/m)*100))

def g(z):
	return 1 / (1 + np.exp(-z))

def g_(z):
	return g(z) * (1 - g(z))

def getH(x, theta1, theta2):
	z2 = np.dot(theta1, x.T)

	a2 = g(z2)

	m = len(a2.T)
	a2 = np.hstack([np.ones((m, 1)), a2.T])

	z3 = np.dot(theta2, a2.T)

	a3 = g(z3)

	return a3

def grad(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, l):
	theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
	theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
	delta1 = np.zeros((num_ocultas, num_entradas + 1))
	delta2 = np.zeros((num_etiquetas, num_ocultas + 1))
	m = len(X)

	for i in range(1, m):
		a3 = h(X[i], theta1, theta2)

		l3 = a3 - y[i]

		z2 = np.hstack([1, np.dot(theta1, X[i].T)])
		a2 = g(z2)

		arg1 = np.dot(theta2.T, l3)
		arg2 = g_(z2)

		l2 = (arg1 * arg2)
		l2 = l2[1:]

		l3 = l3.reshape(len(l3), 1)
		a2 = a2.reshape(len(a2), 1)
		theta_aux = theta2[:, :]
		theta_aux[:, 0] = 0
		delta2 += np.dot(l3, a2.T) + (l / m) * theta_aux		

		a1 = X[i]

		l2 = l2.reshape(len(l2), 1)
		a1 = a1.reshape(len(a1), 1)
		theta_aux = theta1[:, :]
		theta_aux[:, 0] = 0
		delta1 += np.dot(l2, a1.T) + (l / m) * theta_aux

	delta = np.concatenate((delta1.ravel(), delta2.ravel())) / m

	return delta

def h(x, theta1, theta2):
	z2 = np.dot(theta1, x.T)

	a2 = g(z2)
	a2 = np.hstack([1, a2.T])

	z3 = np.dot(theta2, a2.T)

	a3 = g(z3)

	return a3

def main():
	# Get data
	data = loadmat('ex4data1.mat')
	X = data['X']

	y = data['y']
	y_cat = to_categorical(y)  # Categoriza los datos
	y_cat = y_cat[:, 1:]  # Se busca que el 1 esté en la primera posición y el 0 en la última

	# Show data
	sample = np.random.choice(X.shape[0], 100)
	fig, ax = displayData(X[sample, :])
	fig.savefig('numeros.png')

	X = np.hstack([np.ones((len(X), 1)), X])  # Le añade una columna de unos a las x

	# Get thetas
	weights = loadmat('ex4weights.mat')
	theta1, theta2 = weights['Theta1'], weights['Theta2']
	# Theta1 es de dimensión 25x401
	# Theta2 es de dimensión 10x26

	params_rn = np.concatenate((theta1.ravel(), theta2.ravel()))

	l = 1

	#cost, grad = backprop(params_rn, len(X[0]) - 1, len(theta1), len(theta2), X, y_cat, l)
	#print(cost)
	#checkNNGradients(backprop, 0)

	theta1, theta2 = min_coste(len(X[0]) - 1, len(theta1), len(theta2), X, y_cat, l)
	evaluar(getH(X, theta1, theta2), y)

def min_coste(num_entradas, num_ocultas, num_etiquetas, X, y, reg):
	initialTheta1 = pesosAleatorios(num_entradas, num_ocultas)
	initialTheta2 = pesosAleatorios(num_ocultas, num_etiquetas)
	params_rn = np.concatenate((initialTheta1.ravel(), initialTheta2.ravel()))

	result = opt.fmin_tnc(func=coste, x0=params_rn, fprime=grad, args=(num_entradas, num_ocultas, num_etiquetas, X, y, reg))
	theta1 = np.reshape(result[0][:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
	theta2 = np.reshape(result[0][num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))
	return (theta1, theta2)

def pesosAleatorios(L_in, L_out):
	e = math.sqrt(6) / math.sqrt(L_in + L_out)
	pesos = 2 * e * np.random.rand(L_out, L_in + 1) - e
	return pesos

if (__name__ == '__main__'):
	main()