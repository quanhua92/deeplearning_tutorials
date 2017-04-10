import tensorflow as tf
from tensorflow.contrib.layers import dropout, fully_connected

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

keep_prob = 0.5

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

X_drop = dropout(X, keep_prob, is_training=is_training)

with tf.contrib.framework.arg_scope(
    [fully_connected], weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)):

    hidden1 = fully_connected(X_drop, n_hidden1, scope="hidden1")
    hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)

    hidden2 = fully_connected(hidden1_drop, n_hidden2, scope="hidden2")
    hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)

    logits = fully_connected(hidden2_drop, n_outputs, scope="outputs", activation_fn=None)


