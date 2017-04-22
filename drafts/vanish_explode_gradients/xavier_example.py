import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

X = None  # tf.placeholder for your inputs
n_hidden1 = 100
# By default, fully_connected function uses Xavier initialization
# You can change to He initialization
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = fully_connected(X, n_hidden1, weights_initializer=he_init, scope="hidden1")