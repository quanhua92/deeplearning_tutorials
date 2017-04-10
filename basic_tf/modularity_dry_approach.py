import tensorflow as tf


def relu(X):
    # Method 01: create a threshold variable on first call
    # with tf.variable_scope("relu"):
    # if not hasattr(relu, "threshold"):
    #     relu.threshold = tf.Variable(0.0, name="threshold")
    # Method 02: use get_variable func in tf
    # with tf.variable_scope("relu", reuse=True):
    # threshold = tf.get_variable("threshold")
    # Method 03: create relu/threshold inside relu()
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))

    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, threshold, name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

# Method 02: create relu/threshold before call relu()
# with tf.variable_scope("relu"):
#     threshold = tf.get_variable("threshold", shape=(),
#                                 initializer=tf.constant_initializer(0.0))

# Method 01, 02:
# relus = [relu(X) for i in range(5)]

# Method 03:
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=not (relu_index == 0)) as scope:
        relus.append(relu(X))

output = tf.add_n(relus, name="output")

logdir = "/tmp/tf_modular/dry"
file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
file_writer.close()



