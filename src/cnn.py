__author__ = 'Youssef Hmamouche'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam


np.random.seed(1234)
tf.random.set_seed(1234)

#--------------------------------------------------------
class CNN_WRAP:
    def __init__(self, lag):
        self. look_back = lag
        self. model = Sequential()

    def save (self, file_path):
        self. model. save (file_path)

    def load (self, file_path):
        self. model = load_model(file_path)

    def summary (self):
        self. model. summary ()

    def fit (self, X, Y, epochs = 100, batch_size = 1, verbose = 0, shuffle = True):

        n_features =  int (X.shape[1] / self.look_back)
        n_samples = X.shape[0]

        new_shape = [n_samples, self.look_back, n_features, 1]
        X_reshaped = np. reshape (X, new_shape, order = 'F')

        self. model = Sequential()
        self. model. add (Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same', input_shape=(self. look_back , n_features, 1)))
        self. model. add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self. model. add (Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same'))
        self. model. add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        self. model. add(Dropout(0.2))
        self. model. add (Flatten())

        self. model. add (Dense(100, activation='relu'))
        self. model. add (Dense(1, activation='sigmoid'))
        opt = Adam (lr=0.01)
        self. model.compile (loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
        self. model. summary ()
        self. model.fit (X_reshaped, Y,  epochs = epochs, batch_size = batch_size, verbose = verbose, shuffle = shuffle)

    def predict (self, X):
        X_reshaped = np. reshape (X, (X.shape[0], self. look_back, int (X.shape[1] / self. look_back), 1), order = 'F')
        preds = self. model. predict (X_reshaped, batch_size = 1). flatten ()
        return preds

#--------------------------------------------------------
