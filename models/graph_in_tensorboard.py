import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.compat.v1.Session() as sess:
    model_filename ='ConvNet_220220_37Classes.pb'
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='../logs/retrain_logs'
train_writer = tf.compat.v1.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)

train_writer.flush()
train_writer.close()