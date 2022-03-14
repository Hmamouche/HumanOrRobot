__author__ = 'Youssef Hmamouche'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf

#--------------------------------------------------------
class LSTM_WRAP:
	def __init__(self, lag):
		self. look_back = lag
		self. model = None

	def save (self, file_path):
		self. model. save (file_path)

	def load (self, file_path):
		self. model = load_model(file_path)

	def summary (self):
		self. model. summary ()

	def fit (self, X, Y, epochs = 20, batch_size = 32, verbose = 0, validation_split = 0, shuffle = True):
		n_features =  int (X.shape[1] / self.look_back)
		n_samples = X.shape[0]

		#n_neurons = int (0.67 * (n_features + 1))
		n_neurons = 2*n_features
		new_shape = [n_samples, self.look_back, n_features]
		X_reshaped = np. reshape (X, new_shape, order = 'F')

		self.model = Sequential()
		self. model. add (LSTM (24, input_shape=(self. look_back , n_features), return_sequences = True))
		self. model. add (LSTM (12))
		#self. model.add(Dropout(0.9))
		self. model. add (Dense(10, activation='relu'))
		self. model. add (Dense(1, activation='sigmoid'))
		opt = SGD (learning_rate=0.01)
		self.model.compile (loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


		history = self.model.fit (X_reshaped, Y,  epochs = epochs, batch_size = batch_size,
								  validation_split = validation_split, verbose = verbose, shuffle = shuffle)
		return history

	def predict (self, X):
		X_reshaped = np. reshape (X, (X.shape[0], self. look_back, int (X.shape[1] / self. look_back)), order = 'F')
		preds = self. model. predict (X_reshaped, batch_size = 1). flatten ()
		return preds
	def get_normalized_weights (self):
		for layer in self.model.layers:
			weights = layer.get_weights()[0]
			break
		weights = np. abs (np.mean (weights, axis=1)). tolist ()
		sum = np.sum (weights)

		return [a / sum for a in weights]



#--------------------------------------------------------
