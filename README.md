# Save the Wav Speech Commands

I wrote a tool that takes a .wav input of 16000 samples, in 16000 chunks, uses a queue to store the audio data in `numpy.int16` format.

In the main loop, it writes to a `.wav` called `output.wav`, which is then passed to the model to run inference upon via calling the `label_wav` function in `label_wav.py`. This takes a 1 second, 16000 samples `.wav` input, converts it to a spectrogram, runs inference on the spectrogram, and returns a range of probabilities.

The aim of the tool was to provide pre-processing for raw audio input from a microphone, store the data in a data buffer, before passing the data to the model.

Post-Processing of information happens in `label_wav.py`. We take the prediction with the highest predicted probability, and return the prediction, score (predicted probability), and dictionary encoding of the highest predicted word in a tuple.

Key actors here are `label_wav.py` and `SaveTheWav_debug.py`. Running `SaveTheWav_debug.py` starts inference, taking `ConvNet.pb` as a graph input and `conv_labels.txt` as desired labels for said input. 

The problem begins when the program has been running for more than 30 time steps, as the prediction time increaes linearly (from 0.12 seconds to 2.2 seconds.) By the time we have hit 150 seconds of runtime, the queue size (buffer) has filled up to 37 in queue. This is not great.

Requirements are TensorFlow, Numpy, PyAudio. (just install the latest versions, as at 06/02. Particularly important is TF 2.0.)


### Reference

This is forked from the basic speech recognition example by TensorFlow. For more information on the Tensorflow Implementation, see the
tutorial at https://www.tensorflow.org/tutorials/sequences/audio_recognition.
