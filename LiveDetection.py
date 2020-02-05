'''
Ryan Foo

Ensure you have already trained the model and placed it in the appropriate directory.
'''

# Imports

import tensorflow as tf
# import matplotlib.pyplot as plt

import collections
import numpy as np
import os
import wave

import pyaudio
from scipy.io import wavfile

'''
Load the trained model
'''

# model = 

'''
Audio Listener. Write to a pyaudio.Stream() <Open Microphone>
'''

# Change between Live and saving the .wav.
SAVE_THE_WAV = True

# 10 buffer == 1 frame
# chunk == buffer (same thing / example)

chunk = 16000 # Chunks of 16000 samples # Also buffer
sample_format = pyaudio.paInt16
fs = 16000 # mic sampling rate
frames = 10
channels = 1
filename = 'output.wav' 
p = pyaudio.PyAudio() # Interface to Port Audio

# Run Demo for timeout number of seconds

print('Recording... stop and close the stream by pressing Ctrl + C.')

stream = p.open(
        format=sample_format,
        channels=channels,
        rate=fs,
        input=True,
        frames_per_buffer=chunk
        )

buff = [] # Initialize array to store frames

print('Storing Frames in buffer.')


# Store data in chunks for desired amount of seconds
for i in range(0, int(fs / chunk * frames)):
    data = stream.read(chunk)
    buff.append(data)

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
	wf.writeframes(b''.join(buff))
	wf.close()

	print('Successfully saved to {}.'.format(filename))







'''
Cut data into 10 pieces of 1 second each
'''



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