import tensorflow as tf

converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
    frozen_model_filename, INPUT_NODE, OUTPUT_NODE)

TFLITE_OUTPUT_FILE =

tflite_model = converter.convert()
open(TFLITE_OUTPUT_FILE, "wb").write(tflite_model)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
   parser.add_argument(
      '--input_file',
      type=str,
      default='ConvNet.pb',
      help="""\
      Path to input file.
      """)
  parser.add_argument(
      '--output_file',
      type=str,
      default='ConvNet.tflite',
      help='Path to output file.')
  parser.add_argument(
      '--input_node',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--background_volume',
      type=float,
      default=0.1,
      help="""\
      How loud the background noise should be, between 0 and 1.
      """)