import numpy as np
import pandas as pd

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
	devel_label_file = 'labels/dev_labels.csv'
	labels_devel = pd.read_csv(devel_label_file, sep=',')

	# prediction files -> EDIT THESE PATHS TO YOUR PREDICTIONS
	dev_1_pred = 'predictions/is11_dev_svm_predictions.csv'
	test_1_pred = 'predictions/is11_test_svm_predictions.csv'
	
	dev_2_pred = 'predictions/egemaps_dev_nn_predictions.csv'
	test_2_pred = 'predictions/egemaps_test_nn_predictions.csv'
	
	# Load the predictions by speaker obtained with get_predictions_by_speaker.py
	preds_devel_1 = pd.read_csv(dev_1_pred)
	preds_devel_1.columns=['file_id', 'predictions_1']
	preds_test_1 = pd.read_csv(test_1_pred)
	preds_test_1.columns=['file_id', 'predictions_1']

	preds_devel_2 = pd.read_csv(dev_2_pred)
	preds_devel_2.columns=['file_id', 'predictions_2']
	preds_test_2 = pd.read_csv(test_2_pred)
	preds_test_2.columns=['file_id', 'predictions_2']


	# Merge all predictions and labels (if available) as columns of the same dataframe:
	devel = pd.merge(pd.merge(labels_devel, preds_devel_1, on='file_id'), preds_devel_2, on='file_id')
	test = pd.merge(preds_test_1, preds_test_2, on='file_id')
	
	
	# TODO: compute the majority vote
	#devel['mv'] = ...
	#test['mv'] = ...


	# Print out the results for each model and for the final combination
	print("Results for the Development Dataset")
	
	print("Results for prediction 1")
	f1 = precision_recall_fscore_support(devel.label.values, devel.predictions_1.values, labels=[0,1], average='macro')
	print(f1)
	
	print("Results for tprediction 2")
	f1 = precision_recall_fscore_support(devel.label.values, devel.predictions_2.values, labels=[0,1], average='macro')
	print(f1)


	print("Results for majority vote")
	f1 = precision_recall_fscore_support(devel.label.values, devel.mv.values, labels=[0,1], average='macro')
	print(f1)
	
	# Save predictions...

	
if __name__ == "__main__":
	main()
