import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, MultiRNNCell, DropoutWrapper

n_neurons = 100
n_layers = 3

keep_prop = 0.5

cell = BasicRNNCell(num_units=n_neurons)
cell_drop = DropoutWrapper(cell, input_keep_prob=keep_prop)

multi_layer_cell = MultiRNNCell([cell_drop] * n_layers)
outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
