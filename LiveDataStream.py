'''
Ryan Foo

Ensure you have already trained the model and placed it in the appropriate directory.
'''

# Imports

import tensorflow as tf
import collections
from queue import Queue
from threading import Thread
import label_wav
import numpy as np
import sys
import time
import os
import wave

import pyaudio
from scipy.io import wavfile

_, arr2 = wavfile.read('output.wav')

'''
Load the trained model
'''

# model = 

'''
Audio Listener. Write to a pyaudio.Stream() <Open Microphone>
'''

# Change between Live and saving the .wav.
SAVE_THE_WAV = False
run = True

# 10 buffer == 1 frame
# chunk == buffer (same thing / example)

chunk = 16000 # Chunks of 16000 samples # How muc audio sample per frame are we going to display?
sample_format = pyaudio.paInt16
fs = 16000 # mic sampling rate
feed_duration = 1
channels = 1
filename = 'output.wav' 
p = pyaudio.PyAudio() # Interface to Port Audio

# Run Demo for timeout number of seconds
timeout = time.time() + 0.5 * 60 # 30 seconds from now
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

print('Get audio, begin callback.')

stream = get_audio_input_stream(callback)
stream.start_stream()

# Continual Live Inference

try:
    while run:
        data = q.get()

        # print('Data shape: {}'.format(data.shape))
        # print('Arr 2 shape: {}'.format(arr2.shape))

        # Convert np int16 to .wav format, without saving
        print(data)
        print(arr2)

        time.sleep(1.0)

        '''
        recognize_graph = load_graph(FLAGS.model)
        with recognize_graph.as_default():
        	with tf.Session() as sess:
        '''


except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False


# End stream (if SAVE_THE_WAV is True)

if (SAVE_THE_WAV):

	stream.stop_stream()
	stream.close()
	p.terminate()

	print('Finished Recording.')

	# Save recorded data as a WAV file
	wf = wave.open(filename, 'wb')
	wf.setnchannels(channels)
	wf.setsampwidth(p.get_sample_size(sample_format))
	wf.setframerate(fs)
	wf.writeframes(b''.join(data))
	wf.close()

	print('Successfully saved the last 10 seconds to {}.'.format(filename))




'''
Run inference on the .wav file
'''

'''
Post-process the probability
'''

'''
Return the probability
'''






'''
Resources

https://realpython.com/playing-and-recording-sound-python/
Andrew Ng on Coursera
https://github.com/douglas125/SpeechCmdRecognition
'''