from scipy.io  import loadmat
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from helper import *
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

def main():
	data = loadmat('numbers.mat')
	# Se pueden consultar las claves con data.keys()
	y = data['y']
	X = data['X']
	# Almacena los datos leídos en X e y

	y[y == 10] = 0

	neurona_profunda(X, y, 'digitos')

def tryExample(X, y_predict):
	for i in range(10):  # Prueba 10 ejemplos aleatorios
		plt.figure()
		sample = np.random.choice(X.shape[0], 1)
		plt.imshow(X[sample, :].reshape(-1, 20).T)
		plt.axis('off')
		val = np.argmax(y_predict[sample])
		plt.title('Se clasifica como: {}'.format(val))
		plt.savefig('ejemplo{}.png'.format(i))  # y los pinta

def neurona_profunda(X, y, label):
	y_cat = to_categorical(y)

	model = Sequential()
	model.add(Dense(64, input_shape=(400,), activation='tanh'))
	model.add(Dense(32, activation='tanh'))
	model.add(Dense(16, activation='tanh'))
	model.add(Dense(10, activation='softmax'))

	model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(X, y_cat, verbose=0, epochs=50)

	plot_confusion_matrix(model, X, y, label)
	plot_loss_accuracy(history, label)
	y_predict = model.predict(X)
	for row in y_predict:
		i = np.argmax(row)
		row[:] = 0
		row[i] = 1
	print(classification_report(y_cat, y_predict, target_names=['Class_0', 'Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9']))
	tryExample(X, y_predict)

main()