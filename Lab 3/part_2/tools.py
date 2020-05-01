import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Load labels from CSV file
def load_labels(files):

	train_labels = files[0]
	devel_labels = files[1]
	
	y_train = pd.read_csv(train_labels, sep=',')['Label'].values
	y_devel = pd.read_csv(devel_labels, sep=',')['Label'].values
	
	return y_train, y_devel



# Load data from CSV file
def load_data(files):
	train_file = files[0]
	devel_file = files[1]
	test_file  = files[2]
	
	X_train = pd.read_csv(train_file, header=0, index_col=False, sep=';', usecols = lambda column : column not in ["name", "frameTime"]).values
	X_devel = pd.read_csv(devel_file, header=0, index_col=False, sep=';', usecols = lambda column : column not in ["name", "frameTime"]).values
	X_test  = pd.read_csv(test_file, header=0, index_col=False, sep=';', usecols = lambda column : column not in ["name", "frameTime"]).values
	
	return X_train, X_devel, X_test


# Saves predictions
def save_predictions(file_list_path, predictions, output_path):
	files = pd.read_csv(file_list_path)
	file_ids = files.file_id.values
	pred_df = pd.DataFrame({'file_id': file_ids, 'predictions': predictions})
	pred_df.to_csv(output_path, index=False)


# plot training history
def plot_training_history(epochs, plottable, ylabel='', name=''):
	plt.clf()
	plt.xlabel('Epoch')
	plt.ylabel(ylabel)
	if len(plottable) == 1:
		plt.plot(np.arange(epochs), plottable[0], label='Loss')
	elif len(plottable) == 2:
		plt.plot(np.arange(epochs), plottable[0], label='Acc')
		plt.plot(np.arange(epochs), plottable[1], label='UAR')
	else:
		raise ValueError('plottable passed to plot function has incorrect dim.')
	plt.legend()
	plt.savefig('%s.png' % (name), bbox_inches='tight')
