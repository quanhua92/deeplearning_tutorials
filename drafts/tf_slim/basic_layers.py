import tensorflow as tf
from tensorflow.contrib import slim
from utils.debug_print import *

print("Convolutional Networks")

g = tf.Graph()

with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="inputs")

    with tf.variable_scope("native-conv", [inputs]):
        with tf.variable_scope("conv1"):
            weights = tf.get_variable(trainable=True,
                                      name="weights",
                                      shape=[3, 3, 3, 64],
                                      initializer=tf.random_normal_initializer(stddev=0.1),
                                      dtype=tf.float32)
            conv = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding="SAME", name="conv")
            biases = tf.get_variable(trainable=True,
                                     name="biases",
                                     shape=[64],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            out = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(out, name="relu")

        with tf.variable_scope("conv2"):
            weights = tf.get_variable(trainable=True,
                                      name="weights",
                                      shape=[3, 3, 64, 64],
                                      initializer=tf.random_normal_initializer(stddev=0.1),
                                      dtype=tf.float32)
            conv = tf.nn.conv2d(conv1, weights, [1, 1, 1, 1], padding="SAME", name="conv")
            biases = tf.get_variable(trainable=True,
                                     name="biases",
                                     shape=[64],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            out = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(out, name="relu")

        with tf.variable_scope("pool3"):
            outputs = tf.nn.max_pool(conv2, name="max_pool", ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    print("Parameters")
    print_variables(slim.get_variables("native-conv"))
    print("Input / Output")
    print_variables([inputs, outputs])


g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="inputs")

    with tf.variable_scope("slim-conv", [inputs]):
        net = slim.conv2d(inputs, num_outputs=64, kernel_size=[3, 3], padding="SAME",
                          weights_initializer=tf.random_normal_initializer(stddev=0.1),
                          scope="conv1")
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], padding="SAME",
                          weights_initializer=tf.random_normal_initializer(stddev=0.1),
                          scope="conv2")
        outputs = slim.max_pool2d(net, [2, 2], scope="pool3")

    print("Parameters")
    print_variables(slim.get_variables("slim-conv"))
    print("Input / Output")
    print_variables([inputs, outputs])

g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="inputs")

    with tf.variable_scope("slim-repeat-conv", [inputs]):
        net = slim.repeat(inputs, 2, slim.conv2d, num_outputs=64, kernel_size=[3, 3], padding="SAME",
                          weights_initializer=tf.random_normal_initializer(stddev=0.1),
                          scope="conv")
        outputs = slim.max_pool2d(net, [2, 2], scope="pool3")

    print("Parameters")
    print_variables(slim.get_variables("slim-repeat-conv"))
    print("Input / Output")
    print_variables([inputs, outputs])

print("Fully Connected Layers")

g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 512], name="inputs")

    with tf.variable_scope("native-fc", [inputs]):
        with tf.variable_scope("fc1"):
            weights = tf.get_variable(trainable=True,
                                      name="weights",
                                      shape=[512, 1024],
                                      initializer=tf.random_normal_initializer(stddev=0.1),
                                      dtype=tf.float32)
            biases = tf.get_variable(trainable=True,
                                     name="biases",
                                     shape=[1024],
                                     initializer=tf.random_normal_initializer(stddev=1.0),
                                     dtype=tf.float32)
            fc1 = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
            fc1 = tf.nn.relu(fc1)

        with tf.variable_scope("fc2"):
            weights = tf.get_variable(trainable=True,
                                      name="weights",
                                      shape=[1024, 256],
                                      initializer=tf.random_normal_initializer(stddev=0.1),
                                      dtype=tf.float32)
            biases = tf.get_variable(trainable=True,
                                     name="biases",
                                     shape=[256],
                                     initializer=tf.random_normal_initializer(stddev=1.0),
                                     dtype=tf.float32)
            fc2 = tf.nn.bias_add(tf.matmul(fc1, weights), biases)
            fc2 = tf.nn.relu(fc2)

        with tf.variable_scope("fc3"):
            weights = tf.get_variable(trainable=True,
                                      name="weights",
                                      shape=[256, 10],
                                      initializer=tf.random_normal_initializer(stddev=0.1),
                                      dtype=tf.float32)
            biases = tf.get_variable(trainable=True,
                                     name="biases",
                                     shape=[10],
                                     initializer=tf.random_normal_initializer(stddev=1.0),
                                     dtype=tf.float32)
            fc3 = tf.nn.bias_add(tf.matmul(fc2, weights), biases)
            fc3 = tf.nn.relu(fc2)

        with tf.variable_scope("dropout4"):
            fc = tf.nn.dropout(fc3, 0.5, name="dropout")

        with tf.variable_scope("softmax"):
            outputs = tf.nn.softmax(fc, name="softmax")

    print("Parameters")
    print_variables(slim.get_variables("native-fc"))
    print("Input / Output")
    print_variables([inputs, outputs])


g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 512], name="inputs")

    with tf.variable_scope("slim-fc", [inputs]):
        net = slim.fully_connected(inputs, 1024,
                                   weights_initializer=tf.random_normal_initializer(stddev=0.1),
                                   scope="fc1")
        net = slim.fully_connected(inputs, 256,
                                   weights_initializer=tf.random_normal_initializer(stddev=0.1),
                                   scope="fc2")
        net = slim.fully_connected(inputs, 10,
                                   weights_initializer=tf.random_normal_initializer(stddev=0.1),
                                   scope="fc3")
        net = slim.dropout(net, 0.5, scope="dropout4")
        outputs = slim.softmax(net, scope="softmax")

    print("Parameters")
    print_variables(slim.get_variables("slim-fc"))
    print("Input / Output")
    print_variables([inputs, outputs])


g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 512], name="inputs")

    with tf.variable_scope("slim-stack-fc", [inputs]):
        net = slim.stack(inputs, slim.fully_connected, [1024, 256, 10],
                         weights_initializer=tf.random_normal_initializer(stddev=0.1),
                         scope="fc")

        net = slim.dropout(net, 0.5, scope="dropout4")
        outputs = slim.softmax(net, scope="softmax")

    print("Parameters")
    print_variables(slim.get_variables("slim-stack-fc"))
    print("Input / Output")
    print_variables([inputs, outputs])