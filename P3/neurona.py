from scipy.io  import loadmat
import numpy as np

def g(z):
	return 1 / (1 + np.exp(-z))

def evaluar(h, y):
	m = len(h.T)
	cont = 0
	for i in range(m):
		if (np.argmax(h.T[i]) + 1) == y[i, 0]:
			cont += 1
	print('Acierta el {}%\n'.format((cont/m)*100))

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
	
	h = getH(X, theta1, theta2)  # Hace la propagación hacia delante
	evaluar(h, y)


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