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


def run_svm(data_files, label_files):
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

   	
	X_train = sklearn.preprocessing.scale(X_t)
	X_devel = sklearn.preprocessing.scale(X_d)
	X_test = sklearn.preprocessing.scale(X_te)
	# Define Model Parameters
	parms = {'kernel': 'rbf',
			 'C'	 : 1,
			 'g'	 : 1/X_train.shape[1],
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
	#save_predictions('caminho ate file_txt',predictions_test,'caminho para output file')
	# you may use the function save_predictions in tools.py

	# Save Model - After we train a model we can save it for later use
	pkl.dump(model, open('svm_model.pkl','wb'))



def run_nn(data_files, label_files):
	# define training parameters:
	# epochs = 20
	# learning_rate = 0.001
	# l2_decay = 0
	# batch_size = 64
	# dropout = 0.1

	epochs 		  = 20
	learning_rate = 0.001
	l2_decay 	  = 0
	batch_size    = 64
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
	optimizer = 'sgd'
	optims = optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD} 
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

	# Save test predictions
	# TODO
	# you may use the function save_predictions in tools.py

	# save the model
	torch.save(model, 'nn_model.pth')

	# plot training history
	plot_training_history(epochs, [train_mean_losses], ylabel='Loss', name='training-loss')
	plot_training_history(epochs, [valid_accs, valid_uar], ylabel='Accuracy', name='validation-metrics')



def main():

	directory = "/mnt/c/Users/Goncalo/Documents/Tecnico/MestreMEEC/Mestrado/2 semestre/Processamento da Fala/Speech-Processing-Laboratories/Lab 3/part_2" # Full path to your current folder
	feature_set = "is11" # name of the folder with the feature set

	# Label files
	labels_train = 'train_labels.csv'
	labels_devel = 'dev_labels.csv'

	label_files = [labels_train, labels_devel]

	# Data files is11_train_data.csv
	data_train = directory + '/features/' + feature_set + '_train.csv'
	data_devel = directory + '/features/' + feature_set + '_dev.csv'
	data_test  = directory + '/features/' + feature_set + '_test.csv'
	data_files = [data_train, data_devel, data_test]
	# Run SVM - PART 2
	run_svm(data_files, label_files)

	# Run NN - PART 3
	#run_nn(data_files, label_files)

if __name__ == "__main__":
	main()
