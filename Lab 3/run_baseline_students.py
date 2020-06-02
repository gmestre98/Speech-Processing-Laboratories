import os
import pickle as pkl

from tools import *

from nn_torch_functions import *
from svm_functions import *

import sklearn
import numpy as np
import random as rn
import torch

# Fix random seeds for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(12456)
rn.seed(12345)
torch.manual_seed(1234)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_svm(data_files, label_files, feature_set):
	'''
	 For the functions load_data() and load_labels() to work correctly:
		- data_files should be a size 3 array with the path of the data file for the Training, Development and Test sets.
		- label_files should be a size 2 array with the path of the label file for the Training and Development sets.
		- Each data file should contain a matrix with shape (N_samples, N_features), with semicolon separated values.
		- Each label file should contain 2 columns, separated with a comma. One column should contain the id of the speaker
		  and the second sould contain the label (0 or 1). The header corresponding to these two columns should be file_id,
		  label. The file may contain other columns, but those will be ignored.
	'''
	# Load data and labels
	X_t, X_d, X_te = load_data(data_files)
	y_train, y_devel = load_labels(label_files)
	nfeat = X_t.shape[1]

	scaler = sklearn.preprocessing.StandardScaler()
	scaler.fit(X_te)
	X_train = scaler.transform(X_t)
	X_devel = scaler.transform(X_d)
	X_test = scaler.transform(X_te)
	# Define Model Parameters
	if feature_set == "avec":
		parms = {'kernel': 'rbf',
				'C'	 : 0.5,
				'g'	 : 0.5/nfeat,
			 	'd'	 : 3}
	else:
		parms = {'kernel': 'rbf',
				'C'	 : 0.8,
				'g'	 : 1/nfeat,
			 	'd'	 : 3}

	# Train Model
	# Inspect the function train_svm at svm_functions.py and change class_weight
	print ('Train the model...')
	model = train_svm(X_train,y_train,parms)

	# Test the model: compute predictions and metrics for train and devel
	train_prf, train_accuracy = test_svm(X_train, y_train, model)
	print('train - accuracy: ', train_accuracy, 'prf:', train_prf)

	dev_prf, dev_accuracy = test_svm(X_devel ,y_devel, model)
	print('dev - accuracy: ', dev_accuracy, 'prf:', dev_prf)

	# Compute predictions for dev and test data
	predictions_dev  = model.predict(X_devel)
	predictions_test = model.predict(X_test)

	# Save test predictions
	cwd = os.getcwd()
	dev_path = cwd + "/csvfiles/dev_labels.csv"
	test_path = cwd + "/csvfiles/test_labels.csv"
	output_dev_path = cwd + "/predictions/"+ feature_set + ".dev.result.svm.csv"
	output_test_path = cwd + "/predictions/"+ feature_set + ".test.result.svm.csv"
	# Save test predictions
	save_predictions(dev_path, predictions_dev, output_dev_path)
	save_predictions(test_path, predictions_test, output_test_path)
	# you may use the function save_predictions in tools.py

	# Save Model - After we train a model we can save it for later use
	pkl.dump(model, open('svm_model.pkl','wb'))



def run_nn(data_files, label_files, feature_set):
	# define training parameters:
	# epochs = 20
	# learning_rate = 0.001
	# l2_decay = 0
	# batch_size = 64
	# dropout = 0.1

	if feature_set == "is11":
		epochs 		  = 15
		learning_rate = 0.15
		l2_decay 	  = 0
		batch_size    = 128
		dropout 	  = 0.3
	elif feature_set == "avec":
		epochs 		  = 11
		learning_rate = 0.1
		l2_decay 	  = 0
		batch_size    = 128
		dropout 	  = 0.45
	elif feature_set == "egemaps":
		epochs 		  = 40
		learning_rate = 0.15
		l2_decay 	  = 0
		batch_size    = 128
		dropout 	  = 0.1


	# define loss function. The weights tensor corresponds to the weight we give
	# to each class. It corresponds to the inverse of the frequency of that class
	# in the training set. This is a strategy to deal with imbalanced datasets.
	criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1], dtype=torch.float).to(device)) # TODO: change class weights

	# initialize dataset with the data files and label files
	dataset = SleepinessDataset(data_files, label_files)

	# Get number of classes and number of features from dataset
	n_classes  = torch.unique(dataset.y).shape[0]
	n_features = dataset.X.shape[1]

	# initialize the model
	model = FeedforwardNetwork(n_classes, n_features, dropout)
	model = model.to(device)

	# get an optimizer
	# define the optimizer:
	optimizer = 'adam'
	optims = optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adagrad": torch.optim.Adagrad}
	#Try different optimizers: Adam, Adagrad, ... Full list can be found in pytorch's documentation
	optim_cls = optims[optimizer]
	optimizer = optim_cls(
		model.parameters(),
		lr=learning_rate,
		weight_decay=l2_decay)

	# train the model
	model, train_mean_losses, valid_accs , valid_uar = train(dataset, model, optimizer, criterion, batch_size, epochs)


	# evaluate on train set
	train_X, train_y 	 = dataset.X, dataset.y
	train_acc, train_prf = evaluate(model, train_X, train_y)

	print('Final Train acc: %.4f' % (train_acc))
	print('Final Train prf: ', train_prf)


	# evaluate on dev set
	dev_X, dev_y 	 = dataset.dev_X, dataset.dev_y
	dev_acc, dev_prf = evaluate(model, dev_X, dev_y)

	print('Final dev acc: %.4f' % (dev_acc))
	print('Final dev prf: ', dev_prf)


	# get predictions for test and dev set
	test_X = dataset.test_X
	predictions_dev = predict(model, dev_X)
	predictions_dev  = predictions_dev.detach().cpu().numpy()

	predictions_test = predict(model, test_X)
	predictions_test = predictions_test.detach().cpu().numpy()

	cwd = os.getcwd()
	dev_path = cwd + "/csvfiles/dev_labels.csv"
	test_path = cwd + "/csvfiles/test_labels.csv"
	output_dev_path = cwd + "/predictions/"+ feature_set + ".dev.result.nn.csv"
	output_test_path = cwd + "/predictions/"+ feature_set + ".test.result.nn.csv"
	# Save test predictions
	save_predictions(dev_path, predictions_dev, output_dev_path)
	save_predictions(test_path, predictions_test, output_test_path)
	# you may use the function save_predictions in tools.py

	# save the model
	torch.save(model, 'nn_model.pth')

	# plot training history
	plot_training_history(epochs, [train_mean_losses], ylabel='Loss', name='training-loss')
	plot_training_history(epochs, [valid_accs, valid_uar], ylabel='Accuracy', name='validation-metrics')



def main():

	directory = os.getcwd() # Full path to your current folder
	feature_set = "is11" # name of the folder with the feature set

	if not os.path.exists('csvfiles'):
		os.makedirs('csvfiles')
	# Label files
	labels_train = 'csvfiles/train_labels.csv'
	labels_devel = 'csvfiles/dev_labels.csv'

	label_files = [labels_train, labels_devel]

	if not os.path.exists('predictions'):
		os.makedirs('predictions')
	# Data files is11_train_data.csv
	data_train = directory + '/csvfiles/' + feature_set + '_train.csv'
	data_devel = directory + '/csvfiles/' + feature_set + '_dev.csv'
	data_test  = directory + '/csvfiles/' + feature_set + '_test.csv'
	data_files = [data_train, data_devel, data_test]
	# Run SVM - PART 2
	run_svm(data_files, label_files, feature_set)

	# Run NN - PART 3
	run_nn(data_files, label_files, feature_set)

if __name__ == "__main__":
	main()
