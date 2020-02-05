'''
Ryan Foo

Ensure you have already trained the model, and its labels, and placed it in the appropriate directory.
'''

# Imports

# import tensorflow as tf
# import collections
from queue import Queue
# from threading import Thread
import label_wav
import numpy as np
import time
# import os
import wave

import pyaudio
# from scipy.io import wavfile

'''
Audio Listener. Write to a pyaudio.Stream() <Open Microphone>
'''

run = True
verbose = True

threshold = 0.5 #threshold for detection
chunk = 16000 # Chunks of 16000 samples # How muc audio sample per frame are we going to display?
sample_format = pyaudio.paInt16
fs = 16000 # mic sampling rate
feed_duration = 1
channels = 1
filename = 'output.wav' 
p = pyaudio.PyAudio() # Interface to Port Audio

# Run Demo for timeout number of seconds
timeout = time.time() + 0.5 * 300 # 150 seconds from now
feed_samples = int(fs * feed_duration)

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
        # Process data async by sending a queue.
        q.put(data)

    return (in_data, pyaudio.paContinue)

def get_audio_input_stream(callback):
    stream = p.open(
        format=sample_format,
        channels=channels,
        rate=fs,
        input=True,
        output=True,
        frames_per_buffer=chunk,
        stream_callback=callback)
    return stream

# Data Buffer for input waveform
data = np.zeros(feed_samples, dtype='int16')

# Storing predictions to do post-processing upon
predictions = []

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

# Continual Live Inference

try:
    while run:
        # Dequeue data
        start_time_main_loop = time.time()
        data = q.get()

        print("Queue size:", q.qsize())

        # Convert np int16 to .wav format
        start_time_write_to_wav = time.time()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(data))

        stop_time_write_to_wav = time.time()

        print("Writing to a .wav took ", round(stop_time_write_to_wav - start_time_write_to_wav, 2), " seconds to run.")
        # wf.close()

        # Label.wav takes .wav, labels, graph, input_name, output_name and how many labels you want, running inference to print the predictions.

        start_time_prediction = time.time()

        prediction = label_wav.label_wav(wav='output.wav', 
            labels='conv_labels.txt',
            graph='ConvNet.pb', 
            input_name='wav_data:0', 
            output_name='labels_softmax:0', 
            how_many_labels=1)

        stop_time_prediction = time.time()

        print("Prediction took ", round(stop_time_prediction - start_time_prediction, 2), " seconds to run.")

        # if (prediction[1] > threshold):
        #     predictions.append(prediction[2]) # Append encoding, push old encoding out if its above 5!
        #     predictions.push()
        # else:
        #     predictions.append(0)

        # if (verbose == True):
        #     print(prediction) # (predicted word, score <predicted probability>, encoding in a tuple)
        #     print(predictions) # Current list of predictions thus far

        # Detects if threshold is passed, and if the prediction is not 'silence' or 'unknown'
        if (prediction[1] > threshold and prediction[2] != 0):

            print('Detected the word ' + str(prediction[0]) + ' with {} confidence.'.format(prediction[1])
                )

        stop_time_main_loop = time.time()
        print ("One main loop takes ", round(stop_time_main_loop - start_time_main_loop, 2), " seconds to run")


except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False