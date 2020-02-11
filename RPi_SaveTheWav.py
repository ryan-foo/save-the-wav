import tensorflow as tf
from queue import Queue
import label_wav
import numpy as np
from datetime import datetime
import time
import os
import wave
import argparse
import pyaudio

RaspberryPi = True
run = True
verbose = False

if (RaspberryPi):
    channels=1
    input_device_index=0
else:
    channels=1
    input_device_index=0

LowLatencySVDF = "models/low_latency_SVDF_100000_onetwothreefouronoffstopgo.pb"
LowLatencySVDFLabels = "models/low_latency_svdf_labels.txt"
AccurateConv = "models/ConvNet.pb"
AccurateConvLabels = "models/conv_labels.txt"
LowLatencyConv = "models/low_latency_conv_41000_onetwothreestopgo.pb"
LowLatencyConvLabels = "models/low_latency_conv_labels.txt"

'''
Audio Listener. Write to a pyaudio.Stream() <Open Microphone>
'''
threshold = 0.5 #threshold for detection
filename = 'output.wav' 
p = pyaudio.PyAudio() # Interface to Port Audio

# Run Demo for timeout number of seconds
timeout = time.time() + 0.5 * 300 # 150 seconds from now
feed_samples = 16000

# Initialize Queue
q = Queue()

print('Recording... stop and close the stream by pressing Ctrl + C.')

def callback(in_data, frame_count, time_info, status):
    global run, timeout, data 
    if time.time() > timeout:
        run = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    data = np.append(data,data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:] 
        # Take the last second of data, the most recent second
        # Process data async by sending a queue.
        q.put(data)

    return (in_data, pyaudio.paContinue)

def get_audio_input_stream(callback):
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=16000,
        input_device_index=input_device_index, #Raspberry Pi Only
        input=True,
        output=False,
        frames_per_buffer=16000,
        stream_callback=callback)
    return stream

# Data Buffer for input waveform
data = np.zeros(feed_samples, dtype='int16')

# Storing predictions to do post-processing upon
predictions = []
prediction_times = []

print('Get audio, begin callback.')

stream = get_audio_input_stream(callback)
# Could it have a specific method
# You have to follow the encoding method that you used in your training
# What other encoding
stream.start_stream()

# If Buffer is full, empty the older buffer
# By tuning, you know the length of the frame is a good representation of your expected voice
# You are confident that is a sufficient length
# Averaging the probability over the whole length of the frame
# if along the buffer, all the probability is high, then you increase your predicted probability
#because we pick up the voice real time, we have to have a buffer
# any time within the length of buffer

# Initialize loading labels, graphs

# def SaveTheWav(graph=LowLatencySVDF, label=LowLatencySVDFLabels):
    # label_wav.load_graph(graph)
    # labels_list = label_wav.load_labels(label)

label_wav.load_graph(AccurateConv)
labels_list = label_wav.load_labels(AccurateConvLabels)

# Continual Live Inference
try:
    while run:
        # Shift data
        start_time_main_loop = time.time()
        data = q.get()

        print("Queue size:", q.qsize())

        # Convert np int16 to .wav format
        # start_time_write_to_wav = time.time()

        # Saves the .wav
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(data))
        wf.close()

        # stop_time_write_to_wav = time.time()

        # print("Writing to a .wav took ", round(stop_time_write_to_wav - start_time_write_to_wav, 2), " seconds to run.")
        # wf.close()

        # Label.wav takes .wav, labels, graph, input_name, output_name and how many labels you want, running inference to print the predictions.

        start_time_prediction = time.time()

        with open(filename, 'rb') as wav_file:
            wav_data = wav_file.read()

        prediction = label_wav.run_graph(wav_data=wav_data, 
            labels=labels_list,
            input_layer_name='wav_data:0', 
            output_layer_name='labels_softmax:0',
            num_top_predictions=1)

        stop_time_prediction = time.time()
        prediction_time = round(stop_time_prediction-start_time_prediction,2)

        print("Prediction took {} seconds to run.".format(prediction_time))

        # if (prediction[1] > threshold):
        #     predictions.append(prediction[2]) # Append encoding, push old encoding out if its above 5!
        #     predictions.push()
        # else:
        #     predictions.append(0)

        if (verbose == True):
            print(prediction) # (predicted word, score <predicted probability>, encoding in a tuple)
            print(predictions) # Current list of predictions thus far

        # Detects if threshold is passed, and if the prediction is not 'silence' or 'unknown'
        if (prediction[1] > threshold and prediction[2] != 0):
            print('Detected the word ' + str(prediction[0]) + ' with {} confidence.'.format(prediction[1]))

        os.remove("output.wav")

        stop_time_main_loop = time.time()
        print ("One main loop takes ", round(stop_time_main_loop - start_time_main_loop, 2), " seconds to run")

        # Esmond put code here


except (KeyboardInterrupt, SystemExit):
    wf.close()
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False

# def main(_):
#   """Entry point for script, converts flags to arguments."""
#   SaveTheWav(FLAGS.graph, FLAGS.label)

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument(
#       '--graph',
#       type=str,
#       default="models/low_latency_SVDF_100000_onetwothreefouronoffstopgo.pb",
#       help='The graph you intend to use.')
#   parser.add_argument(
#       '--label',
#       type=str,
#       default="models/low_latency_svdf_labels.txt",
#       help='The labels you intend to use.')

#   FLAGS, unparsed = parser.parse_known_args()
#   tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
