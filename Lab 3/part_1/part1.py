import pandas as pd
import numpy as np

train_data = pd.read_csv("../corpus/labels/train_labels.csv")
dev_data = pd.read_csv("../corpus/labels/dev_labels.csv")


train_length = np.size(train_data, 0)
print("The train dataset length is:", train_length)
dev_length = np.size(dev_data, 0)
print("The dev dataset length is:", dev_length)

train_label = np.zeros(train_length, dtype=np.int8)
dev_label = np.zeros(dev_length, dtype=np.int8)

for i in range(1,train_length):
	if train_data.kss[i] > 7.5:
		train_label[i] = 1
for i in range(1,dev_length):
	if dev_data.kss[i] > 7.5:
		dev_label[i] = 1

train_data['Label'] = train_label
dev_data['Label'] = dev_label

fspeakers = 0
mspeakers = 0
fsl = 0
msl = 0
fnsl = 0
mnsl = 0

for i in range(0, train_length):
	if train_data.Gender[i] == 'F':
		fspeakers = fspeakers + 1
		if train_data.Label[i] == 1:
			fsl = fsl + 1
		if train_data.Label[i] == 0:
			fnsl = fnsl + 1
	if train_data.Gender[i] == 'M':
		mspeakers = mspeakers + 1
		if train_data.Label[i] == 1:
			msl = msl + 1
		if train_data.Label[i] == 0:
			mnsl = mnsl + 1

print("train female speakers:", fspeakers)
print("train male speakers", mspeakers)
print("train sl recordings female", fsl)
print("train sl recordings male", msl)
print("train nsl recordings female", fnsl)
print("train nsl recordings male", mnsl)

fspeakers = 0
mspeakers = 0
fsl = 0
msl = 0
fnsl = 0
mnsl = 0

for i in range(0, dev_length):
	if dev_data.Gender[i] == 'F':
		fspeakers = fspeakers + 1
		if dev_data.Label[i] == 1:
			fsl = fsl + 1
		if dev_data.Label[i] == 0:
			fnsl = fnsl + 1
	if dev_data.Gender[i] == 'M':
		mspeakers = mspeakers + 1
		if dev_data.Label[i] == 1:
			msl = msl + 1
		if dev_data.Label[i] == 0:
			mnsl = mnsl + 1

print("dev female speakers:", fspeakers)
print("dev male speakers", mspeakers)
print("dev sl recordings female", fsl)
print("dev sl recordings male", msl)
print("dev nsl recordings female", fnsl)
print("dev nsl recordings male", mnsl)


train_data.to_csv('../part_2/train_labels.csv', index=False)
dev_data.to_csv('../part_2/dev_labels.csv', index=False)