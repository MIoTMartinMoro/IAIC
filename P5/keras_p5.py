from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import classification_report
import numpy as np
from helper import *
from keras.models import Sequential
from keras.layers import Dense

def regresion(x, y, label):
	model = Sequential()
	model.add(Dense(1, input_shape=(2,), activation='sigmoid'))

	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

	history = model.fit(x, y, verbose=0, epochs=100)
	plot_loss_accuracy(history, label)
	y_predict = model.predict(x)
	plot_decision_boundary(lambda x: model.predict(x), x, y, label)
	plot_confusion_matrix(model, x, y, label)
	return y_predict
def main():
	X1, y1 = make_circles (n_samples=1000, noise =0.05,  factor =0.3, random_state=0)
	X2, y2 = make_moons(n_samples=1000, noise =0.05, random_state=0)

	plot_data(X1, y1, 'circles')
	y_predict = regresion(X1, y1, 'circles_regr')
	y_predict = (y_predict > 0.5).astype('int')
	y_predict = y_predict.ravel()
	print(classification_report(y1, y_predict, target_names=['Class_0', 'Class_1']))


	plot_data(X2, y2, 'moon')
	y_predict = regresion(X2, y2, 'moon_regr')
	y_predict = (y_predict > 0.5).astype('int')
	y_predict = y_predict.ravel()
	print(classification_report(y2, y_predict, target_names=['Class_0', 'Class_1']))

main()
