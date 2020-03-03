# Save the Wav Speech Commands

I wrote a tool that takes a .wav input of 16000 samples, in 16000 chunks, uses a queue to store the audio data in `numpy.int16` format.

In the main loop, it writes to a `.wav` called `output.wav`, which is then passed to the model to run inference upon via calling the `label_wav` function in `label_wav.py`. This takes a 1 second, 16000 samples `.wav` input, converts it to a spectrogram, runs inference on the spectrogram, and returns a range of probabilities.

The aim of the tool was to provide pre-processing for raw audio input from a microphone, store the data in a data buffer, before passing the data to the model.

Post-Processing of information happens in `label_wav.py`. We take the prediction with the highest predicted probability, and return the prediction, score (predicted probability), and dictionary encoding of the highest predicted word in a tuple.

Key actors here are `label_wav.py` and `debugSaveTheWav.py`. Running `debugSaveTheWav.py` starts inference, taking a `.pb` as a graph input and `conv_labels.txt` as desired labels for the output.

Requirements are TensorFlow, Numpy, PyAudio, sklearn. (just install the latest versions, as at 06/02/20. Particularly important is using TF=1.15.)

### SaveTheWav Model Zoo

```
python3 SaveTheWav.py
```
<Models and Architectures from the speech recognition example. You can find more credits there. I do not claim credit for these architectures.>

Mac OSX: Run on Mac OSX with Python3.
RPi: Raspberry Pi 3.

Two separate types of model being trained at the moment.

```
10 classes
'_silence_ _unknown_ one two three four on off stop go'

7 classes
'_silence_ _unknown_ one two three stop go'
```

Desired words -- words that have been trained for and are meant to output commands.

Detection ratio: Desired words detected divided by desired words spoken.
False alarm: Desired words detected when there is not a desired word being spoken.

Hypothesis:
Less classes means less false alarms, higher detection ratio.
More parameters means better inferences (less false alarm rate, higher detection ratio.)

#### ConvNet (10 Classes) (ConvNet_10Classes_070220.pb)

Your standard Convolutional Neural Network model.

Runtime (Mac OSX): 0.1-0.2 seconds avg.

#### Low Latency SVDF (10 Classes) (LowLatencySVDF_10Classes_110220.pb)

Should run faster than Low Latency Conv.

Runtime (Mac OSX): 0.1-0.2 seconds.

Observation:
Runtime difference is negligible, even on Raspberry Pi.

#### ConvNet (36 Classes) (models/ConvNet_220220_37Classes.pb)

### Testing

Confusion matrix can be generated via running `access_files.py`. This will test the model against the test set, and generate validation / recall scores.
store the final confusion matrix from testing.

To get the accuracy of the model over time,
`tensorboard --logdir logs/retrain_logs`

### Running Debug

When running `debugSaveTheWav.py`, it will write to `debug_log.txt` re: the time taken to run the inference.

### Converting to TFLite

Finding `input_array` and `output_array` is non trivial. Use `convert.py` while training to find the desired arrays. The current `input_name` is `wav_data`, `output_name` is `labels_softmax`, and you will need to specify the shape of `wav_data`. There are unsupported TensorFlow ops we attempt to convert.

`converter = tf.lite.TFLiteConverter.from_frozen_graph(
    localpb, 
    ["wav_data"], 
    ['labels_softmax'],
    {"wav_data":[1,160,160,3]}
)`

 Bypass and create a hybrid model with `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]`.

[Afterwards, use this notebook to convert your .pb file to a .tflite file.](https://colab.research.google.com/drive/1fB8HjfGWqtkqcsPliMVYqfmxvFRShm29)

### To Do

Get `streaming_test.py` working.
Argument Parser for `SaveTheWav.py` (including debug and verbose options)
`requirements.txt`


### Reference

This is forked from the basic speech recognition example by TensorFlow. For more information on the Tensorflow Implementation, see the
tutorial at https://www.tensorflow.org/tutorials/sequences/audio_recognition.
