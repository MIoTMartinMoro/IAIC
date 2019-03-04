from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from pandas.io.parsers import read_csv

def h(theta, x):
	return np.dot(theta, x.T)  # Hace el producto escalar entre theta y x transpuesta. y=x0*theta0 + x1*theta1

def main():
	valores = read_csv('ex1data1.csv', header=None).values  # Lee los valores
	valores = valores.astype(float)  # Los convierte a float
	m = len(valores)

	v = np.hstack([np.ones((m, 1)), valores])  # Le añade una columna de unos a las x. Así se puede hacer la función y=theta0 + x*theta1 como un producto escalar

	x = v[:, 0:2]  # Separa la matriz de valores en x
	y = v[:, -1:]  # e y

	a = 0.01  # Valor alpha de entrenamiento

	Thetas, coste = descenso_gradiente(x, y, a)  # Calculamos el menor coste y con que thetas se obtiene
	dibuja(x[:, -1], y, Thetas)  # Dibuja la recta de regresión sobre los puntos de los valores
	theta0, theta1, costes = coste_3d(x, y)  # Calcula todos los costes en forma de malla para dibujar la gráfica en 3D
	dibuja_3d(theta0, theta1, costes)  # Dibuja la gráfica en 3D
	dibuja_contorno(theta0, theta1, costes, Thetas)  # Dibuja la proyección 2D del plano de la 3D y el punto de mínimo coste.

def descenso_gradiente(x, y, a):
	m = len(x)
	theta = np.array([1., 1.])  # Valor inicial de thetas que luego se irán actualizando
	costes = []
	thetas_array = []
	for i in range(1500):
		sum0 = 0
		sum1 = 0
		for j in range(m):  # Para cada uno de los valores (filas de x o y):
			sum0 += h(theta, x[j, :]) - y[j]  # Se le resta el valor real (y) al que da como respuesta la ecuación con las thetas actuales y se suman al sumatorio
			sum1 += (h(theta, x[j, :]) - y[j]) * x[j, -1:]  # Para theta1, además se multiplica por la variable leída de x. En el caso anterior también, pero como siempre se 1 no lo ponemos
		theta[0] = theta[0] - (a / m) * sum0  # Se actualiza el valor de theta0 y theta1 con estas ecuaciones.
		theta[1] = theta[1] - (a / m) * sum1  # MUY IMPORTANTE hacer primero los dos sumatorios antes de actualizar las thetas
		costes.append(coste(theta, x, y))  # Se calcula el coste y se añade al array de costes
		thetas_array.append(theta)  # Se añaden las thetas al array con todas las thetas calculadas

	# Se devuelve el coste mínimo y las thetas que lo provocan
	return (thetas_array[np.argmin(costes)], np.min(costes))

def coste(theta, x, y):
	m = len(x)
	coste = (1 / (2 * m)) * np.sum((h(theta, x) - y.T)**2)  # Ecuación del coste
	return coste

def coste_3d(x, y):
	theta0 = np.arange(-10, 10, 0.1)
	theta1 = np.arange(-1, 4, 0.1)
	theta0, theta1 = np.meshgrid(theta0, theta1)  # Convierte los valores de theta0 y theta1 en una malla.
	costes = np.empty_like(theta0)  # Se crea una matriz costes vacía igual que theta0

	for xi, yi in np.ndindex(theta0.shape):
		theta = np.array([theta0[xi, yi], theta1[xi, yi]])
		costes[xi, yi] = coste(theta, x, y)

	return (theta0, theta1, costes)  # Devuelve las matrices de thetas0, thetas1 y costes

def dibuja(x, y, theta):
	plt.plot(x, y, 'rx')  # Dibuja los valores como una 'x' roja
	x_range = range(23)
	x_matriz = np.array(x_range).reshape(23, 1)  # Convierte la fila en columna
	x_h = np.hstack([np.ones((23, 1)), x_matriz])  # Le añade una columna de unos
	plt.plot(x_range, h(theta, x_h))  # Dibuja la recta de regresión
	plt.savefig('regresion.png')

def dibuja_3d(theta0, theta1, costes):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Plot the surface.
	surf = ax.plot_surface(theta0, theta1, costes, cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.savefig('3d.png')

def dibuja_contorno(theta0, theta1, costes, Thetas):
	fig = plt.figure()
	# Plot the surface.
	plt.contour(theta0, theta1, costes, np.logspace(-2, 3, 20))
	plt.plot(Thetas[0], Thetas[1], 'rx')

	plt.savefig('contour.png')

main()