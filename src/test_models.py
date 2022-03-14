__author__ = 'Youssef Hmamouche'

import pandas as pd
import numpy as np
import os, argparse, joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import f1_score,  balanced_accuracy_score, recall_score

from cnn import *
from mlp import *
from lstm import *
from bi_lstm import *

from normalizer import normalizer

#-------------------------------------
def pred_real_scores (real, cont_pred, threshold):

	disc_pred = [a for a in cont_pred]
	if threshold is not None:
		for i in range (len (disc_pred)):
			if disc_pred[i] < threshold:
				disc_pred[i] = 0
			else:
				disc_pred[i] = 1

	fscore 	= np. round (f1_score (real, disc_pred, average = 'weighted'), 2)
	accuracy = balanced_accuracy_score (real, disc_pred)
	recall 	= np. round (recall_score (real, disc_pred, average = 'weighted'), 2)

	return fscore, accuracy, recall

#-------------------------------------
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", "-m", choices = ["mlp", "lstm","bi-lstm", "cnn", "RF"])
	parser.add_argument("--type", "-t", choices = ["ling", "ling_80", "fbank"])
	parser.add_argument("--remove", "-rm", help="remove last results.", action="store_true")
	args = parser.parse_args()

	test_data = pd. read_pickle ("data/%s_test_data.pkl"%args.type)
	lag = int (test_data. columns[1]. split ('_')[-1][1:])
	test_data = test_data. values
	X = test_data[:,1:]
	y = test_data[:,0]

	if args. model == "cnn":
		model = CNN_WRAP (lag)
		model. load ("trained_models/cnn_%s.h5"%args.type)
		model. summary ()
	elif args. model == "lstm":
		model = LSTM_WRAP (lag)
		model. load ("trained_models/lstm_%s"%args.type)
		model. summary ()
	elif args. model == "bi-lstm":
		model = BiLSTM_WRAP (lag)
		model. load ("trained_models/bi_lstm_%s"%args.type)
		model. summary ()
	elif args. model == "mlp":
		model = MLP_WRAP (lag)
		model. load ("trained_models/mlp_%s.h5"%args.type)
		model. summary ()
	elif args. model == "RF":
		model = joblib. load ("trained_models/random_forest_%s.pkl"%args.type)

	# make predictions and compute the scores
	predictions = model. predict (X)
	if args. model == "RF":
		seuils = [None]
	else:
		seuils = [0.3, 0.4, 0.5, 0.6, 0.7]

	results = []
	for seuil in seuils:
		fscore, accuracy, recall = pred_real_scores (y, predictions, seuil)
		print (seuil, ': ', accuracy, fscore, recall)
		results. append ([args. model.upper (), accuracy, fscore, recall])

	df = pd.DataFrame (results, columns = ["Model", "Fscore", "Accuracy", "Recall"])
	if len (df) > 1:
		df = df[df["Fscore"] == df["Fscore"].max ()]


	# remove last results is specified in the input arguments
	if args. remove:
		os. system ("rm results/scores_test_%s.csv"%args.type)
	# store the results
	if not os.path.exists ("results/scores_test_%s.csv"%args.type):
		df. to_csv ("results/scores_test_%s.csv"%args.type, sep = ";", index = False, mode = 'w', header = True)
	elif (pd.read_csv ("results/scores_test_%s.csv"%args.type, sep = ";"). shape[0] > 0):
		df. to_csv ("results/scores_test_%s.csv"%args.type, sep = ";", index = False, mode = 'a', header = False)
	else:
		df. to_csv ("results/scores_test_%s.csv"%args.type, sep = ";", index = False, mode = 'w', header = True)
