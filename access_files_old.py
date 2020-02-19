import os
import label_wav
#https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

words = ['one', 'two', 'three', 'four', 'on', 'off', 'stop', 'go']

LowLatencySVDF = "models/LowLatencySVDF_10Classes_110720.pb"
LowLatencySVDFLabels = "models/low_latency_svdf_labels.txt"
AccurateConv = "models/ConvNet_10Classes_070220.pb"
AccurateConvLabels = "models/conv_labels.txt"
LowLatencyConv = "models/LowLatencyConv_7Classes_110220.pb"
LowLatencyConvLabels = "models/low_latency_conv_labels.txt"

TRIALSPERWORD = 50
record = True

# Choice of Model 
label_wav.load_graph(AccurateConv)
labels_list = label_wav.load_labels(AccurateConvLabels)

def files_finder(word):
    # If file not found, you will have to download the speech commands dataset. Run train.py.
	for dirpath, dirnames, files in os.walk(('/tmp/speech_dataset/{}').format(word)):
		continue
	return(files)

def single_step_label_wav(word, index):
	filename = "/tmp/speech_dataset/" + word + "/" + files_finder(word)[index]
	
	with open(filename, 'rb') as wav_file:
		wav_data = wav_file.read()

	prediction = label_wav.run_graph(wav_data, 
	labels=labels_list,
	input_layer_name='wav_data:0', 
	output_layer_name='labels_softmax:0',
	num_top_predictions=1)

	expected_label = word

	return(expected_label, prediction[0], prediction[1])

def confusion_matrix_generator(expected_label, prediction):
    	return()
for i in range(0, TRIALSPERWORD):
	print(single_step_label_wav(words[0], i))