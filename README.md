# Save the Wav Speech Commands

This is a tool that takes a .wav input of 16000 samples, in 16000 chunks, uses a queue to store the audio data in `numpy.int16` format.

I recommend you create a virtual environment before doing this, as per Python best practice.

You will have to install [PortAudio](http://www.portaudio.com/) on your system, and then

```
pip3 install requirements.txt
```

In the main loop, it writes to a `.wav` called `output.wav`, which is then passed to the model to run inference upon via calling the `label_wav` function in `label_wav.py`. This takes a 1 second, 16000 samples `.wav` input, converts it to a spectrogram, runs inference on the spectrogram, and returns a range of probabilities.

The aim of the tool was to provide pre-processing for raw audio input from a microphone, store the data in a data buffer, before passing the data to the model.

Post-Processing of information happens in `label_wav.py`. We take the prediction with the highest predicted probability, and return the prediction, score (predicted probability), and dictionary encoding of the highest predicted word in a tuple.

Key actors here are `label_wav.py` and `debugSaveTheWav.py`. Running `debugSaveTheWav.py` starts inference, taking a `.pb` as a graph input and `conv_labels.txt` as desired labels.

At every time step, we output how long the prediction time takes to run, and how long a main loop takes to run. I am currently in the process of exploring more sophisticated post-processing steps in order to increase the robustness.

### Using the Engine

```
python3 debugSaveTheWav.py
```

[![asciicast](https://asciinema.org/a/vqa3ENTwOwHpGOxBYj6b4IyWQ.svg)](https://asciinema.org/a/vqa3ENTwOwHpGOxBYj6b4IyWQ)

Ensure you have a microphone connected and the microphone is your default source for receiving audio. You can end the audio stream by pressing Ctrl + C.

The implementation is done using a Python queue so that no data is dropped. Incoming audio data is stored in a data buffer in frames of 1 second (containing 16000 samples each), and the callback fetches a . We print the time it takes for the prediction (from fetching the data from the data buffer to inference), the time for the main loop to run, and the size of the queue. It will print `detected with x% probability` if the predicted probability of the class exceeds >0.5.

You are able to modify the speech-commands it trains with by adjusting `--wanted-words` in `train.py`.


<Models and Architectures from the speech recognition example. You can find more credits there. I do not claim credit for these architectures.>

This was built to work with Raspberry Pi 3 (Tensorflow 1.15 version.)

Desired words -- words that have been trained for and are meant to output commands.

Detection ratio: Desired words detected divided by desired words spoken.
False alarm: Desired words detected when there is not a desired word being spoken.

#### ConvNet (36 classes) (default) (models/ConvNet_220220_37Classes.pb)

This Convolutional Neural Network has been trained on all the classes available in the [Google Speech Commands](https://arxiv.org/abs/1804.03209) v2 dataset. That is to say, all 35 classes and _silence_. You can adjust the words that it selects out of those 35 classes at `debugSaveTheWav.py` -- `if encoding > 8: encoding = 0`. This throws away the prediction if the encoding of the prediction is not one of your desired words. It is trivial to adjust accordingly.

#### Low Latency Conv (7 classes) (models/LowLatencyConv_7Classes_110220.pb)

This is the Low Latency Convolutional Neural Network example, trained on 7 classes. The label list can be found at `models/low_latency_conv_labels.txt`.

### Testing

Confusion matrix can be generated via running `python3 access_files.py`. This will test the model against the test set, and generate validation / recall scores. You should first run `train.py` in order to download the test files to your `/tmp/` folder. `access_files.py` will store the final confusion matrix from testing.

To get the accuracy of the model over time,
`tensorboard --logdir logs/retrain_logs`

### Running Debug

When running `python3 debugSaveTheWav.py`, it will write to `debug_log.txt` re: the time taken to run the inference.

### Converting to TFLite

Finding `input_array` and `output_array` is non trivial. Use `convert.py` while training to find the desired arrays. The current `input_name` is `wav_data`, `output_name` is `labels_softmax`, and you will need to specify the shape of `wav_data`. There are unsupported TensorFlow ops we attempt to convert.

```converter = tf.lite.TFLiteConverter.from_frozen_graph(
    localpb, 
    ["wav_data"], 
    ['labels_softmax'],
    {"wav_data":[1,160,160,3]}
)```

 Bypass and create a hybrid model with `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]`.

[Afterwards, use this notebook to convert your .pb file to a .tflite file.](https://colab.research.google.com/drive/1fB8HjfGWqtkqcsPliMVYqfmxvFRShm29)

### To Do

Implement Temporal Convolutional Network model in `models.py`.
Post processing to improve accuracy.

### Reference

This is forked from the basic speech recognition example by TensorFlow. For more information on the Tensorflow Implementation, see the
tutorial at https://www.tensorflow.org/tutorials/sequences/audio_recognition.
