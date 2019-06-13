from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import classification_report
import numpy as np
from helper import *
from keras.models import Sequential
from keras.layers import Dense

def regresion(x, y, label):
	# Crea el modelo
	model = Sequential()
	model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
	model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

	history = model.fit(x, y, verbose=0, epochs=100) # Entrena el modelo
	plot_loss_accuracy(history, label) # Muestra la gráfica de loss accuracy
	y_predict = model.predict(x) # Calcula la predicción de la y
	plot_decision_boundary(lambda x: model.predict(x), x, y, label) # Dibuja los datos y las distintas secciones en las que se incluyen
	plot_confusion_matrix(model, x, y, label) # Muestra la matriz de confusión
	return y_predict

def main():
	X1, y1 = make_circles (n_samples=1000, noise =0.05,  factor =0.3, random_state=0) # Genera los datos en forma de círculo
	X2, y2 = make_moons(n_samples=1000, noise =0.05, random_state=0) # Genera los datos en forma de lunas
	print(y1)
	print(X1)
	# Los dos siguientes bloques de datos hacen exactamente lo mismo, pero a los diferentes conjuntos de datos, el primero a los datos en círculo y el segundo a los datos en forma de lunas
	# Pinta los puntos de los datos
	# Aplica regresión a los datos
	# Transforma los resultados de la regresión a una matriz de 0 y 1, si es (>0,5) -> 1
	# Convierte la matriz en única fila de números
	# Imprime la tabla de clasificación con las predicciones obtenidas

	# Círculos
	plot_data(X1, y1, 'circles') 
	y_predict = regresion(X1, y1, 'circles_regr') 
	y_predict = (y_predict > 0.5).astype('int') 
	y_predict = y_predict.ravel() 
	print(classification_report(y1, y_predict, target_names=['Class_0', 'Class_1'])) 

	# Forma de lunas
	plot_data(X2, y2, 'moon')
	y_predict = regresion(X2, y2, 'moon_regr')
	y_predict = (y_predict > 0.5).astype('int')
	y_predict = y_predict.ravel()
	print(classification_report(y2, y_predict, target_names=['Class_0', 'Class_1']))

main()
