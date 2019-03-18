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

weights = loadmat('ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
# Theta1 es de dimensión 25x401
# Theta2 es de dimensión 10x26

data = loadmat('ex3data1.mat')
# Se pueden consultar las claves con data.keys()
y = data['y']
X = data['X']
# Almacena los datos leídos en X e y

m = len(X)
x = np.hstack([np.ones((m, 1)), X])  # Le añade una columna de unos a las x. Así se puede hacer la función y=theta0 + x*theta1 como un producto escalar

z2 = np.dot(theta1, x.T)

a2 = g(z2)

m = len(a2.T)
a2 = np.hstack([np.ones((m, 1)), a2.T])

z3 = np.dot(theta2, a2.T)

a3 = g(z3)

print(a3.shape)