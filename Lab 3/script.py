import pandas as pd

directory = "part_2" # Full path to your current folder
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

X_train = pd.read_csv(data_train, header=0, index_col=False, sep=';', usecols = lambda column : column not in ["name", "frameTime"]).values
print(X_train.shape)
print(X_train)