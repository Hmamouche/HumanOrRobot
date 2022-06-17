__author__ = 'Youssef Hmamouche'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout, MultiHeadAttention, Input, Flatten, Add, BatchNormalization, Reshape, TimeDistributed, Multiply
from tensorflow.keras.optimizers import SGD, Adam
#from attention import SeqSelfAttention
from keras_self_attention import SeqSelfAttention

import tensorflow as tf

#--------------------------------------------------------
class Attention(tf.keras.Model):
	def __init__(self, units):
		super(Attention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units) # input x weights
		self.W2 = tf.keras.layers.Dense(units) # hidden states h weights
		self.V = tf.keras.layers.Dense(1) # V

	def call(self, features, hidden):
		# hidden shape == (batch_size, hidden size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden size)
		# we are doing this to perform addition to calculate the score
		hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# score shape == (batch_size, max_length, 1)
		# we get 1 at the last axis because we are applying score to self.V
		# the shape of the tensor before applying self.V is (batch_size, max_length, units)
		score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)) ## w[x, h]
		# attention_weights shape == (batch_size, max_length, 1)
		attention_weights = tf.nn.softmax(self.V(score), axis=1) ## v tanh(w[x,h])

		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * features ## attention_weights * x, right now the context_vector shape [batzh_size, max_length, hidden_size]
		context_vector = tf.reduce_sum(context_vector, axis=1)
		return context_vector, attention_weights

	def get_config(self):
		config = super(Attention, self).get_config().copy()
		config.update({
		})
		return config

class LSTM_ATT:
	def __init__(self, lag, n_features):
		self. look_back = lag
		self. model = None
		self.n_features = n_features

	def save (self, file_path):
		self. model. save (file_path)

	def save_weights (self, file_path):
		self. model. save_weights (file_path)


	def create_model (self):
		inputs =Input(shape=(self. look_back , self.n_features))
		lstm1 = LSTM (24, return_sequences = True)(inputs)
		lstm2 = LSTM (12, return_sequences = True)(lstm1)

		attention = (Dense(1, activation='tanh'))(lstm2)
		attention = Activation('softmax')(attention)
		activations = Multiply()([lstm2, attention])

		activations = Flatten()(activations)
		activations = Dropout(0.2)(activations)

		dense1 = Dense(50, activation='relu')(activations)
		output = Dense(1, activation='sigmoid')(dense1)
		self.model = Model(inputs, output)

	def load_weights (self, file_path):
		self.create_model()
		self. model. load_weights (file_path)

	def load (self, file_path):
		self. model = load_model(file_path, custom_objects=SeqSelfAttention.get_custom_objects())

	def summary (self):
		self. model. summary ()


	def fit (self, X, Y, epochs = 1, batch_size = 32, verbose = 0, validation_split = 0, shuffle = True):
		n_samples = X.shape[0]
		new_shape = [n_samples, self.look_back, self.n_features]
		X_reshaped = np. reshape (X, new_shape, order = 'F')

		self.create_model()

		self.model.summary ()

		opt = SGD (learning_rate=0.01)
		self.model.compile (loss = 'mse', optimizer = opt, metrics = ['accuracy'])

		history = self.model.fit (X_reshaped, Y,  epochs = epochs, batch_size = batch_size,
								  validation_split = validation_split, verbose = verbose, shuffle = shuffle)
		return history

	def predict (self, X):
		X_reshaped = np. reshape (X, (X.shape[0], self. look_back, int (X.shape[1] / self. look_back)), order = 'F')
		preds = self. model. predict (X_reshaped, batch_size = 1). flatten ()
		return preds
	def get_normalized_weights (self):
		for layer in self.model.layers[1:]:
			weights = layer.get_weights()[0]
			break
		weights = np. abs (np.mean (weights, axis=1)). tolist ()
		sum = np.sum (weights)

		return [a / sum for a in weights]



#--------------------------------------------------------
