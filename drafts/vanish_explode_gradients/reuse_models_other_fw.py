import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

original_w = []  # Load weights from other framework
original_b = []  # Load biases from other framework

n_inputs = 28 * 28
n_hidden1 = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = fully_connected(X, n_hidden1, scope="hidden1")


# Get a handle on vars in fully_connected
with tf.variable_scope("", default_name="", reuse=True):
    hidden1_weights = tf.get_variable("hidden1/weights")
    hidden1_biases = tf.get_variable("hidden1/biases")

# Create ops to assign values
ori_weights = tf.placeholder(tf.float32, shape=(n_inputs, n_hidden1))
ori_biases = tf.placeholder(tf.float32, shape=(n_hidden1))

assign_weights_op = tf.assign(hidden1_weights, ori_weights)
assign_biases_op = tf.assign(hidden1_biases, ori_biases)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run([assign_weights_op, assign_biases_op], feed_dict={
        ori_weights: original_w,
        ori_biases: original_b
    })