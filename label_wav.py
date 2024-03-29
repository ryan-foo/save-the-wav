# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.io.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  with tf.compat.v1.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Dictionary Mapping for our labels
    # robot_arm_labels = {'_silence_': 0, '_unknown_': 0, 'one': 1, 'two': 2,
                        # 'three': 3, 'stop': 4, 'go': 5}
    robot_arm_labels = {'_silence_': 0, '_unknown_': 0, 'one': 1, 'two': 2, 
                        'three': 3, 'four': 4, 'on': 5, 'off': 6, 'stop': 7, 'go': 8}

    robot_arm_labels_37_classes = {'_silence_': 0, '_unknown_': 0, 'one': 1, 'two': 2, 
          'three': 3, 'four': 4, 'on': 5, 'off': 6, 'stop': 7, 'go': 8, 
          'backward': 9, 'bed': 10, 'bird': 11, 'cat': 12, 'dog': 13,
          'down': 14, 'eight': 15, 'five': 16, 'forward': 17, 'house': 18,
          'learn': 19, 'left': 20, 'marvin': 21, 'nine': 22, 'no': 23,
          'right': 24, 'seven': 25, 'sheila': 26, 'six': 27, 'tree': 28,
          'up': 29, 'visual': 30, 'wow': 31, 'yes': 32, 'zero': 33,
          'happy': 34, 'follow': 35}

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      score = predictions[node_id]
      human_string = labels[node_id]
      return(human_string, score, robot_arm_labels_37_classes[human_string])
      # print('Predicted word: %s' % (human_string))
      # print('Encoding: %s' % (robot_arm_labels[human_string]))
      # print('%s (score = %.5f)' % (human_string, score))
      
  return(0)


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.io.gfile.exists(wav):
    tf.compat.v1.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.io.gfile.exists(labels):
    tf.compat.v1.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.io.gfile.exists(graph):
    tf.compat.v1.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  return(wav_data, labels_list, input_name, output_name, how_many_labels)


def main(_):
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=1,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)