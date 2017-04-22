import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, MultiRNNCell

n_neurons = 100
n_layers = 3

basic_cell = BasicRNNCell(num_units=n_neurons)
multi_layer_cell = MultiRNNCell([basic_cell] * n_layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
