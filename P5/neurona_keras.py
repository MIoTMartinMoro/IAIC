from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy.optimize as opt
from pandas.io.parsers import read_csv
from helper import *
from keras.models import Sequential
from keras.layers import Dense

X, y = make_multiclass(K=3)
plot_data_3(X, y, 'espiral')