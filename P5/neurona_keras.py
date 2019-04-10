from sklearn.metrics import classification_report
import numpy as np
from helper import *
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

def main():
	X, y = make_multiclass(K=3)  # Genera los datos en forma de espiral
	plot_data_3(X, y, 'espiral')  # Muestra los datos
	una_neurona(X, y, 'una_neurona')  # Predice los datos con una sola neurona
	neurona_profunda(X, y, 'neurona_profunda')  # Predice los datos con varias neuronas

def una_neurona(X, y, label):
	y_cat = to_categorical(y)  # Categoriza los datos de y

	# Crea el modelo
	model = Sequential()
	model.add(Dense(3, input_shape=(2,), activation='softmax'))
	model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(X, y_cat, verbose=0, epochs=20)  # Entrena el modelo

	plot_multiclass_decision_boundary(model, X, y_cat, label)  # Muestra los datos y las distintas secciones en las que se incluye
	plot_confusion_matrix(model, X, y, label)  # Genera una matriz de confusión
	plot_loss_accuracy(history, label)  # Muestra la gráfica de loss y accuracy

	y_predict = model.predict(X)  # Predice los datos a partir del modelo

	# Formatea los valores predichos
	for row in y_predict:
		i = np.argmax(row)  # Saca en que posición está el máximo valor
		row[:] = 0  # Todos los valores son 0
		row[i] = 1  # Menos el de máximo valor que es 1
	print(classification_report(y_cat, y_predict, target_names=['Class_0', 'Class_1', 'Class_2']))  # Genera un reporte de clasificación y lo muestra por pantalla

def neurona_profunda(X, y, label):
	y_cat = to_categorical(y)  # Categoriza los datos de y

	# Crea el modelo
	model = Sequential()
	model.add(Dense(64, input_shape=(2,), activation='tanh'))
	model.add(Dense(32, activation='tanh'))
	model.add(Dense(16, activation='tanh'))
	model.add(Dense(3, activation='softmax'))
	model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(X, y_cat, verbose=0, epochs=50)  # Entrena el modelo

	plot_multiclass_decision_boundary(model, X, y_cat, label)  # Muestra los datos y las distintas secciones en las que se incluye
	plot_confusion_matrix(model, X, y, label)  # Genera una matriz de confusión
	plot_loss_accuracy(history, label)  # Muestra la gráfica de loss y accuracy

	y_predict = model.predict(X)  # Predice los datos a partir del modelo

	# Formatea los valores predichos
	for row in y_predict:
		i = np.argmax(row)  # Saca en que posición está el máximo valor
		row[:] = 0  # Todos los valores son 0
		row[i] = 1  # Menos el de máximo valor que es 1
	print(classification_report(y_cat, y_predict, target_names=['Class_0', 'Class_1', 'Class_2']))  # Genera un reporte de clasificación y lo muestra por pantalla

main()