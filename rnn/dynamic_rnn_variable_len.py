import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import BasicRNNCell

n_inputs = 3
n_neurons = 5
n_steps = 2

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
seq_length = tf.placeholder(tf.int32, shape=[None])

basic_cell = BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32, sequence_length=seq_length)
# dynamic_rnn accepts [None, n_steps, n_inputs] so we don't need to unstack, tack, transpose
# it also outputs [None, n_steps, n_neurons]
# dynamic_rnn use a while_loop function to run through steps, instead of initializing all steps at once.

init = tf.global_variables_initializer()

# X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
# X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 7, 8], [3, 2, 1]])
X_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],  # padding with 0, 0, 0
    [[6, 7, 8], [6, 7, 8]],
    [[9, 0, 1], [3, 2, 1]]
])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
    init.run()
    outputs_val = sess.run(outputs, feed_dict={X: X_batch, seq_length:seq_length_batch})

print(outputs_val)

