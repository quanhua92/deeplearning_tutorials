import tensorflow as tf
from tensorflow.contrib import slim
from utils.debug_print import print_layers, print_variables

g = tf.Graph()
with g.as_default():
    global_step = tf.Variable(tf.constant(value=1, shape=[], dtype=tf.int32),
                              trainable=False, name="global_step", collections=["global_variables"])

    with tf.variable_scope("native"):
        with tf.device("/cpu:0"):
            weights = tf.Variable(trainable=True,
                                  name="weights",
                                  initial_value=tf.truncated_normal([3, 3, 64, 128], stddev=0.1, dtype=tf.float32),
                                  collections=["model_variables"])

            biases = tf.Variable(trainable=True,
                                 name="biases",
                                 initial_value=tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 collections=["model_variables"])

            regularizers = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)

        logits = tf.Variable(trainable=True,
                             name="logits",
                             initial_value=tf.constant(0.0, shape=[32, 10], dtype=tf.float32),
                             collections=["trainable_variables"])
        accuracy = tf.Variable(trainable=False,
                               name="accuracy",
                               initial_value=tf.constant(0.0, shape=[], dtype=tf.float32),
                               collections=["local_variables"])
    print_variables([global_step, weights, biases, logits, accuracy])


g = tf.Graph()
with g.as_default():
    global_step = slim.get_or_create_global_step()

    with tf.variable_scope("slim-weights"):
        weights = slim.model_variable(name="weights",
                                      shape=[3, 3, 64, 128],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01),
                                      regularizer=slim.l2_regularizer(0.05),
                                      device="/cpu:0")
        biases = slim.model_variable(name="biases",
                                     shape=[128],
                                     initializer=tf.zeros_initializer(),
                                     regularizer=slim.l2_regularizer(0.05),
                                     device="/cpu:0")

    logits = slim.variable("logits", shape=[32, 10], initializer=tf.zeros_initializer())
    accuracy = slim.local_variable(initial_value=0.0, name="accuracy")

    print_variables([global_step, weights, biases, logits, accuracy])
    print("- slim scope weights")
    print_variables(slim.get_variables(scope="slim-weights"))
    print("- collections: model_variables")
    print_variables(slim.get_variables(collection="model_variables"))
    print("- collections: trainable_variables")
    print_variables(slim.get_variables(collection="trainable_variables"))
    print("- Global Variables")
    print_variables([slim.get_global_step()])
    print("- Local Variables")
    print_variables(slim.get_variables(collection="local_variables"))
    print("- Regularization losses")
    print_variables(slim.get_variables(collection="regularization_losses"))

