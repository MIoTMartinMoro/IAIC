from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.optimize as opt
from pandas.io.parsers import read_csv
from helper import *
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

def main():
	X, y = make_multiclass(K=3)
	plot_data_3(X, y, 'espiral')
	una_neurona(X, y, 'una_neurona')
	neurona_profunda(X, y, 'neurona_profunda')

def una_neurona(X, y, label):
	y_cat = to_categorical(y)

	model = Sequential()
	model.add(Dense(3, input_shape=(2,), activation='softmax'))

	model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(X, y_cat, verbose=0, epochs=20)

	plot_multiclass_decision_boundary(model, X, y_cat, label)
	plot_confusion_matrix(model, X, y, label)
	plot_loss_accuracy(history, label)
	y_predict = model.predict(X)
	for row in y_predict:
		i = np.argmax(row)
		row[:] = 0
		row[i] = 1
	print(classification_report(y_cat, y_predict, target_names=['Class_0', 'Class_1', 'Class_2']))

def neurona_profunda(X, y, label):
	y_cat = to_categorical(y)

	model = Sequential()
	model.add(Dense(64, input_shape=(2,), activation='tanh'))
	model.add(Dense(32, activation='tanh'))
	model.add(Dense(16, activation='tanh'))
	model.add(Dense(3, activation='softmax'))

	model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(X, y_cat, verbose=0, epochs=50)

	plot_multiclass_decision_boundary(model, X, y_cat, label)
	plot_confusion_matrix(model, X, y, label)
	plot_loss_accuracy(history, label)
	y_predict = model.predict(X)
	for row in y_predict:
		i = np.argmax(row)
		row[:] = 0
		row[i] = 1
	print(classification_report(y_cat, y_predict, target_names=['Class_0', 'Class_1', 'Class_2']))

main()