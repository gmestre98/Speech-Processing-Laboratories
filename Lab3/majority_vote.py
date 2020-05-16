import os
import numpy as np
import pandas as pd
from tools import *

from sklearn.metrics import precision_recall_fscore_support

'''
To use this script your prediction and label files should have the following columns, in this order:
Prediction files: file_id predictions
Label files: file_id label
Files should include a header with the name of each column.

In this script you should change the path to your own prediction and label files.

You will have to compute the majory vote between the predictions you choose. In this script, we
are only loading two predictions, but you may use more.
'''

def main():

	# Load Label
	directory = os.getcwd()
	devel_label_file = directory + '/csvfiles/dev_labels.csv'
	labels_devel = pd.read_csv(devel_label_file, sep=',')
	test_label_file = directory + '/csvfiles/test_labels.csv'
	labels_test = pd.read_csv(test_label_file, sep=',')

	# prediction files -> EDIT THESE PATHS TO YOUR PREDICTIONS
	dev_1_pred = directory + '/predictions/egemaps_dev_svm_predictions.csv'
	test_1_pred = directory + '/predictions/egemaps_test_svm_predictions.csv'

	dev_2_pred = directory + '/predictions/egemaps_dev_nn_predictions.csv'
	test_2_pred = directory + '/predictions/egemaps_test_nn_predictions.csv'

	dev_3_pred = directory + '/predictions/is11_dev_svm_predictions.csv'
	test_3_pred = directory + '/predictions/is11_test_svm_predictions.csv'

	dev_4_pred = directory + '/predictions/is11_dev_nn_predictions.csv'
	test_4_pred = directory + '/predictions/is11_test_nn_predictions.csv'


	# Load the predictions by speaker obtained with get_predictions_by_speaker.py
	preds_devel_1 = pd.read_csv(dev_1_pred)
	preds_devel_1.columns=['file_id', 'predictions_1']
	preds_test_1 = pd.read_csv(test_1_pred)
	preds_test_1.columns=['file_id', 'predictions_1']

	preds_devel_2 = pd.read_csv(dev_2_pred)
	preds_devel_2.columns=['file_id', 'predictions_2']
	preds_test_2 = pd.read_csv(test_2_pred)
	preds_test_2.columns=['file_id', 'predictions_2']

	preds_devel_3 = pd.read_csv(dev_3_pred)
	preds_devel_3.columns=['file_id', 'predictions_3']
	preds_test_3 = pd.read_csv(test_3_pred)
	preds_test_3.columns=['file_id', 'predictions_3']

	preds_devel_4 = pd.read_csv(dev_4_pred)
	preds_devel_4.columns=['file_id', 'predictions_4']
	preds_test_4 = pd.read_csv(test_4_pred)
	preds_test_4.columns=['file_id', 'predictions_4']

	# Merge all predictions and labels (if available) as columns of the same dataframe:
	devel = pd.merge(pd.merge(pd.merge(pd.merge(labels_devel, preds_devel_1, on='file_id'), preds_devel_2, on='file_id'), preds_devel_3, on='file_id'), preds_devel_4, on='file_id')
	test = pd.merge(pd.merge(pd.merge(pd.merge(labels_test, preds_test_1, on='file_id'), preds_test_2, on='file_id'), preds_test_3, on='file_id'), preds_test_4, on='file_id')


	print(devel['predictions_1'])
	print(devel['predictions_2'])
	print(devel['predictions_3'])
	print(devel['predictions_4'])
	print(devel.shape[0])
	print(test['predictions_1'])
	print(test['predictions_2'])
	print(test['predictions_3'])
	print(test['predictions_4'])
	print(test.shape[0])

	res = np.zeros((devel.shape[0],1))

	for i in range (0, devel.shape[0]-1):
		count_1 = 0
		if devel['predictions_1'][i] == 1:
			count_1 = count_1 + 1
		if devel['predictions_2'][i] == 1:
			count_1 = count_1 + 1
		if devel['predictions_3'][i] == 1:
			count_1 = count_1 + 1
		if devel['predictions_4'][i] == 1:
			count_1 = count_1 + 1
		if count_1 > 2:
			res[i] = 1
		else:
			res[i] = 0


	devel['mv'] = res

	# Print out the results for each model and for the final combination
	print("Results for the Development Dataset")

	print("Results for prediction 1")
	f1 = precision_recall_fscore_support(devel.Label.values, devel.predictions_1.values, labels=[0,1], average='macro')
	print(f1)

	print("Results for prediction 2")
	f1 = precision_recall_fscore_support(devel.Label.values, devel.predictions_2.values, labels=[0,1], average='macro')
	print(f1)

	print("Results for prediction 3")
	f1 = precision_recall_fscore_support(devel.Label.values, devel.predictions_3.values, labels=[0,1], average='macro')
	print(f1)

	print("Results for prediction 4")
	f1 = precision_recall_fscore_support(devel.Label.values, devel.predictions_4.values, labels=[0,1], average='macro')
	print(f1)

	print("Results for majority vote")
	f1 = precision_recall_fscore_support(devel.Label.values, devel.mv.values, labels=[0,1], average='macro')
	print(f1)

	# Save predictions...
	dev_path = directory + "/csvfiles/dev_labels.csv"
	output_path = directory + "/predictions/dev.result.fusion.csv"

	save_predictions(dev_path, devel.mv.values, output_path)


	res = np.zeros((test.shape[0],1))

	for i in range (0, test.shape[0]-1):
		count_1 = 0
		if test['predictions_1'][i] == 1:
			count_1 = count_1 + 1
		if test['predictions_2'][i] == 1:
			count_1 = count_1 + 1
		if test['predictions_3'][i] == 1:
			count_1 = count_1 + 1
		if test['predictions_4'][i] == 1:
			count_1 = count_1 + 1
		if count_1 > 2:
			res[i] = 1
		else:
			res[i] = 0
			
	test['mv'] = res

	test_path = directory + "/csvfiles/test_labels.csv"
	output_path = directory + "/predictions/test.result.fusion.csv"

	save_predictions(test_path, test.mv.values, output_path)







if __name__ == "__main__":
	main()
