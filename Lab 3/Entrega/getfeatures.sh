mainDir=`pwd`
wavDir="$mainDir/corpus/wav" 
filesDir= "$mainDir/csvfiles"

TRAINSET="train"
DEVSET="dev"
TESTSET="test"
DATASETS=($TRAINSET $DEVSET)

mkdir -p csvfiles

# loops ober the partitions
for dataset in ${DATASETS[*]}; do
	echo "Partition: $dataset set"

	# creates a list of all .wav files. Notice that for the train and dev sets,
	# the list is the same as the labels file, assuring that files are in the
	# same order as in the labels file. For the test set, we create a list of
	# all the .wav files from the wav directory, and save that list in the labels
	# dir for later use.
	awk -F, 'NR>1{print $1}' csvfiles/${dataset}_labels.csv > file_list.txt
	# loops over the list of files
	while read line; do
		#Uncomment the line for the type of features you want to extract
		#./opensmile-2.3.0/inst/bin/SMILExtract -C opensmile-2.3.0/config/IS11_speaker_state.conf -I $wavDir/$dataset/$line -csvoutput csvfiles/is11_${dataset}.csv
		#./opensmile-2.3.0/inst/bin/SMILExtract -C opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -I $wavDir/$dataset/$line -csvoutput csvfiles/egemaps_${dataset}.csv
		#./opensmile-2.3.0/inst/bin/SMILExtract -C opensmile-2.3.0/config/avec2013.conf -I $wavDir/$dataset/$line -csvoutput csvfiles/avec_${dataset}.csv
	done < file_list.txt
	
	# deletes file_list.txt
	rm file_list.txt

done