__author__ = 'Youssef Hmamouche'

import pandas as pd
import pickle

class normalizer:
	"""
		A class for normalizing of a pandas dataframe
		The order of the features doesn't matter (distinct colnames are required)
	"""

	#----------------------------------#
	def __init__ (self, df = pd.DataFrame ()):
		self. features = list (df. columns)
		self. minMax = {}

		if not df. empty:
			for feature in df. columns:
				self. minMax [feature] = [df. loc [:, feature]. min (), df. loc [:, feature]. max ()]

	#----------------------------------#
	@staticmethod
	def normalize (vect, minMax):
		if (minMax[1] - minMax[0]) > 0:
			for i in range (len (vect)):
				vect [i] = (vect[i] - minMax[0]) / float(minMax[1] - minMax[0])
		return vect

	#----------------------------------#
	def transform (self, df_):
		df = df_. copy (). astype (float)
		for feature in df. columns:

			if feature in self. features:
				df [feature] = self. normalize (df.loc [:,feature]. values, self. minMax [feature])

		return df

	#----------------------------------#
	def save (self, filename):
		pickle_filename = open ("%s.pkl"%filename.split ('.')[0],"wb")
		pickle. dump (self. minMax, pickle_filename)
		pickle_filename.close()

	#------------------------------------#
	def load (self, filename):
		pickle_in = open(filename,"rb")
		self. minMax = pickle. load (pickle_in)
		self. features = self. minMax. keys ()
