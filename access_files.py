import os
import label_wav
import tensorflow as tf

positive_words = ['one', 'two', 'three', 'four', 'on', 'off', 'stop', 'go']
negative_words = ['cat', 'follow', 'learn', 'up', 'forward', 'left', 'dog', 'marvin', 'right',
 'visual', 'backward', 'down', 'nine', 'seven', 'wow', 'bed', 'eight', 'happy', 'no', 'sheila', 'tree',
  'yes', 'bird', 'five', 'house', 'six', 'zero']
words = positive_words + negative_words

robot_arm_labels = {'_silence_': 0, '_unknown_': 0, 'one': 1, 'two': 2, 
					'three': 3, 'four': 4, 'on': 5, 'off': 6, 'stop': 7, 'go': 8}

LowLatencySVDF = "models/LowLatencySVDF_10Classes_110720.pb"
LowLatencySVDFLabels = "models/low_latency_svdf_labels.txt"
AccurateConv = "models/ConvNet_10Classes_070220.pb"
AccurateConvLabels = "models/conv_labels.txt"
LowLatencyConv = "models/LowLatencyConv_7Classes_110220.pb"
LowLatencyConvLabels = "models/low_latency_conv_labels.txt"

TRIALSPERWORD = 100
record = True

# Choice of Model 
label_wav.load_graph(AccurateConv)
labels_list = label_wav.load_labels(AccurateConvLabels)

def read_test_files_into_list():
	with open("/tmp/speech_dataset/testing_list.txt", 'rb') as test_text_file:
		test_set = test_text_file.read()
		test_list_temp = test_set.splitlines()
	test_list = []
	for line in test_list_temp:
		test_list.append(str(line, "utf-8"))
	return(test_list)
# returns: [four/speech.wav, one/speech2.wav]

def split_test_list_into_word_dictionary(test_list):
	test_dictionary = {}

	# Convert to unknown word if not in positive words
	for line in test_list:
		word, filename = (line.split('/'))
		if word not in test_dictionary:
			test_dictionary[word] = []
		test_dictionary[word].append(line)
	return(test_dictionary)
	# Returns dictionary with word as a key

def files_finder(word):
    # If file not found, you will have to download the speech commands dataset. Run train.py.
	for dirpath, dirnames, files in os.walk(('/tmp/speech_dataset/{}').format(word)):
		continue
	return(files)

def single_step_label_wav_v1(word, filename):
	file_path = "/tmp/speech_dataset/" + filename

	with open(file_path, 'rb') as wav_file:
		wav_data = wav_file.read()
	
	prediction = label_wav.run_graph(wav_data, 
									labels=labels_list,
									input_layer_name='wav_data:0', 
									output_layer_name='labels_softmax:0',
									num_top_predictions=1)
	
	# Relabelling
	if word not in positive_words:
		word = '_unknown_'

	ground_truth = robot_arm_labels[word]
	predicted_label = prediction[2]
	score = prediction[1]

	return(ground_truth, predicted_label, score)

def confusion_matrix_helper_v1(test_dictionary):
	y_true = []
	y_pred = []

	for word in positive_words:
		print('Now testing against %d instances of word: %s' % (len((test_dictionary[word])), word))
		for i in range(0, len((test_dictionary[word]))): #len((test_dictionary[word]))
			(ground_truth, predicted_label, score) = single_step_label_wav_v1(word, test_dictionary[word][i])
			y_true.append(ground_truth)
			y_pred.append(predicted_label)
	return(y_true, y_pred)

'''
Execution
'''

test_dictionary = split_test_list_into_word_dictionary(read_test_files_into_list())
y_true, y_pred = confusion_matrix_helper_v1(test_dictionary)

confusion_matrix = tf.math.confusion_matrix(labels=y_true,
											predictions=y_pred,
											num_classes=9)

print('Confusion matrix for %d words per word: 9 classes' % TRIALSPERWORD)
print(confusion_matrix)



print("Accuracy Metrics")

def perf_measure(y_true, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_true[i]!=y_pred[i]:
           FP += 1
        if y_true[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_true[i]!=y_pred[i]:
           FN += 1

    return(TP, FP, TN, FN)

TP, FP, TN, FN = perf_measure(y_true, y_pred)

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("Precision Score: " + str(precision))
print("Recall: " + str(recall))