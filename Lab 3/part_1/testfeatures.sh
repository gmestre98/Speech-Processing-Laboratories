mainDir=`pwd`
wavDir="$mainDir/../corpus/wav" 
labelsDir="$mainDir/../corpus/labels"

DATASET="test"

# loops ober the partitions
for dataset in ${DATASET[*]}; do
	echo "Partition: $dataset set"

	# creates a list of all .wav files. Notice that for the train and dev sets,
	# the list is the same as the labels file, assuring that files are in the
	# same order as in the labels file. For the test set, we create a list of
	# all the .wav files from the wav directory, and save that list in the labels
	# dir for later use.
	find $wavDir/$dataset | grep "\.wav" | awk 'BEGING{print "file_id"} {print $0}' > $labelsDir/${dataset}_file_list.txt
	awk -F, 'NR>1{print $1}' $labelsDir/${dataset}_file_list.txt > file_list.txt
	# loops over the list of files
	while read line; do
		./opensmile-2.3.0/inst/bin/SMILExtract -C opensmile-2.3.0/config/IS11_speaker_state.conf -I $line -csvoutput ../part_2/is11_test.csv
		./opensmile-2.3.0/inst/bin/SMILExtract -C opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -I $line -csvoutput ../part_2/egemaps_test.csv
	done < file_list.txt
	
	# deletes file_list.txt
	rm file_list.txt

done