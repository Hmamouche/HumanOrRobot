__author__ = 'Youssef Hmamouche'

import pandas as pd
import numpy as np
import os, argparse, joblib, pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cnn import *
from mlp import *
from lstm import *
from bi_lstm import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV
from normalizer import normalizer

import matplotlib.pyplot as plt

np.random.seed(5)
tf.random.set_seed(5)


#----------------------------------------
def k_fold_cv (X, y, n_folds, classifier, params_dict):

	"""
		make a k-fold-cross-validation with a random search strategy
	"""

	kf_obj = KFold (n_splits = n_folds, shuffle = False)
	splits = kf_obj. split (X)

	rf_random = RandomizedSearchCV (estimator = classifier, param_distributions = params_dict, n_iter = 20, cv = splits, verbose=0, random_state=5,\
									scoring = ['balanced_accuracy', 'average_precision', 'f1_weighted'], refit = 'f1_weighted', n_jobs = 4)

	search = rf_random. fit (X, y. ravel ())

	std_accuracy = search. cv_results_['std_test_balanced_accuracy'][search.best_index_]
	std_precision = search. cv_results_['std_test_average_precision'][search.best_index_]
	std_fscore = search. cv_results_['std_test_f1_weighted'][search.best_index_]

	mean_accuracy = search. cv_results_['mean_test_balanced_accuracy'][search.best_index_]
	mean_precision = search. cv_results_['mean_test_average_precision'][search.best_index_]
	mean_fscore = search. cv_results_['mean_test_f1_weighted'][search.best_index_]

	best_mean_scores = [mean_accuracy, mean_precision, mean_fscore]
	best_std_scores = [std_accuracy, std_precision, std_fscore]

	k_cv_scores = [search. cv_results_['split%d_test_f1_weighted'%i][search.best_index_] for i in range (n_folds)]

	best_model_params = search. best_params_
	best_model = search. best_estimator_

	return best_model, best_model_params, best_mean_scores, best_std_scores, k_cv_scores

#----------------------------------------
def fit_sklearn_RandForest (X_train, y_train, find_params, pickle_filename):

	# Hyperparameters choices
	random_grid = {'bootstrap': [True, False],\
				   'max_depth': [10, 50, 100, 500, None],\
				  'max_features': ['auto', 'sqrt'],\
				  'n_estimators': [2, 5, 10, 50, 100, 200, 300],\
				  'random_state': [5],\
				  'class_weight': ["balanced_subsample"]}

	# Empty classifer
	model = RandomForestClassifier()

	if find_params:
		# K-fold-cross-validation: fidning best params and the scores on each fold.
		model, params, mean_scores, std_scores, k_cv_scores = k_fold_cv (X_train, y_train, 5, model, random_grid)
		with open('%s.pickle'%pickle_filename, 'wb') as handle:
			joblib.dump (params, handle, protocol=pickle.HIGHEST_PROTOCOL)

	else:
		with open('%s.pickle'%pickle_filename, 'rb') as handle:
			params = pickle.load(handle)
			model.set_params (**params)
			model. fit (X_train, y_train)

	return model

#----------------------------------------
def summarize_features_importance (features, importance, lag = True):

	if lag:
		reduce_features = [('_'). join (a. split ('_')[:-1]) for a in features]
	else:
		reduce_features = features

	df = pd.DataFrame ()

	if len (importance) == 0:
		return list (set (reduce_features)), importance

	df ["feats"] = reduce_features
	df ["scores"] = importance

	df = df. groupby ('feats')["scores"].sum (). reset_index ()

	df = df. sort_values (by = "scores", ascending = False). values

	feats = df[:,0]
	scores = df[:,1]
	return df[:,0]. tolist (), df[:,1]. tolist ()

#----------------------------------------
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", "-m", choices = ["mlp", "lstm", "bi-lstm", "cnn", "RF"])
	parser.add_argument("--type", "-t", choices = ["ling", "ling_80", "fbank"])
	args = parser.parse_args()

	try:
		data = pd.read_pickle ("data/%s_train_data.pkl"%args.type)
		colnames = data. columns[1:]
		lag = int (data. columns[1]. split ('_')[-1][1:])
	except Exception as e:
		print ('There is problem when loading training data.')
		print (e)


	# training based on the input model and the input features
	X = data. values[:,1:]
	y = data. values[:,0]

	features_results = []
	if args. model == "cnn":
		model = CNN_WRAP (lag)
		model.fit (X, y, epochs = 20, batch_size = 32, verbose = 0, shuffle = True)
		model. save ("trained_models/cnn_%s.h5"%args.type)
		model. summary ()
	elif args. model == "lstm":
		model = LSTM_WRAP (lag)

		history = model.fit (X, y, epochs = 20, batch_size = 32, verbose = 0, validation_split = 0.0, shuffle = True)
		model. save ("trained_models/lstm_%s"%args.type)
		weights = model. get_normalized_weights ()

		reduced_colnames = list (set ([('_'). join (a. split ('_')[:-1]) for a in colnames]))
		features, importances = summarize_features_importance (reduced_colnames, weights, lag=False)
		features_results. append ([args. model, features, importances])

	elif args. model == "bi-lstm":
		model = BiLSTM_WRAP (lag)

		model.fit (X, y, epochs = 20, batch_size = 32, verbose = 0, shuffle = True)
		model. save ("trained_models/bi_lstm_%s"%args.type)

		weights = model. get_normalized_weights ()

		reduced_colnames = list (set ([('_'). join (a. split ('_')[:-1]) for a in colnames]))
		features, importances = summarize_features_importance (reduced_colnames, weights, lag=False)
		features_results. append ([args. model, features, importances])


		model. summary ()
	elif args. model == "mlp":
		model = MLP_WRAP (lag)
		model.fit (X, y, epochs = 15, batch_size = 32, verbose = 0, shuffle = True)
		weights = model.get_normalized_weights ()
		features, importances = summarize_features_importance (colnames, weights)
		features_results. append ([args. model, features, importances])
		model. save ("trained_models/mlp_%s.h5"%args.type)
		model. summary ()
	elif args. model == "RF":
		model = fit_sklearn_RandForest (X, y, find_params = 1, pickle_filename = "trained_models/random_forest_params_%s"%args.type)
		joblib. dump (model, "trained_models/random_forest_%s.pkl"%args.type, compress=3)
		features_importance = model.feature_importances_

		reduced_features, importances = summarize_features_importance (colnames, features_importance, lag =True)
		features_results. append ([args.model, reduced_features, importances])

	# save feature importances in a csv file (append to existing results)
	df = pd.DataFrame (features_results, columns = ["Model", "Features", "Importance scores"])

	if not os.path.exists ("results/importance_scores_%s.csv"%args.type):
		df. to_csv ("results/importance_scores_%s.csv"%args.type, sep = ";", index = False, mode = 'w', header = True)
	elif (pd.read_csv ("results/importance_scores_%s.csv"%args.type, sep = ";"). shape[0] > 0):
		df. to_csv ("results/importance_scores_%s.csv"%args.type, sep = ";", index = False, mode = 'a', header = False)
	else:
		df. to_csv ("results/importance_scores_%s.csv"%args.type, sep = ";", index = False, mode = 'w', header = True)
