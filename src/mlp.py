__author__ = 'Youssef Hmamouche'

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np



np.random.seed(3)
tf.random.set_seed(3)

#--------------------------------------------------------
class MLP_WRAP:
	#---------------------------------------#
	def __init__(self, lag):
		self. lag = lag
		self. model = None

	def save (self, file_path):
		self. model. save (file_path)

	def load (self, file_path):
		self. model = load_model(file_path)

	def summary (self):
		self. model. summary ()

	#---------------------------------------#
	def fit (self, X, Y, epochs = 100, batch_size = 1, verbose = 0, shuffle = False):

		n_samples = X.shape[0]
		n_features = X.shape[1]
		n_neurons_1 = int (0.67 * (n_features + 1))
		n_neurons_2 = int (0.33 * (n_features + 1))

		self. model = Sequential()
		self. model. add (Dense (n_neurons_1, activation='relu',  input_dim = n_features))
		self. model. add (Dense (n_neurons_2, activation='relu',  input_dim = n_features))

		self. model. add (Dense (1, activation='sigmoid'))

		opt = SGD(lr=0.01)
		self.model.compile (loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
		self.model.fit (X, Y,  epochs = epochs, batch_size = batch_size, verbose = verbose, shuffle = shuffle)

	#---------------------------------------#
	def predict (self, X):
		preds = self. model. predict (X, batch_size = 1). flatten ()
		return preds

	def get_normalized_weights (self):
		for layer in self.model.layers:
			weights = layer.get_weights()[0]
			break
		#print (len (weights))
		weights = np. abs (np.mean (weights, axis=1)). tolist ()
		sum = np.sum (weights)

		return [a / sum for a in weights]
