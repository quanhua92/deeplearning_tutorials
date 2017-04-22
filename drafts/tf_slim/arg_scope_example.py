import tensorflow as tf
from tensorflow.contrib import slim
from utils.debug_print import *


def slim_model(inputs):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        biases_initializer=tf.constant_initializer(1.0),
                        weights_regularizer=slim.l2_regularizer(0.05)):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[3,3],
                            padding="SAME",
                            biases_initializer=tf.constant_initializer(0.0)):
            with slim.arg_scope([slim.max_pool2d],
                                kernel_size = [2,2],
                                padding="SAME"):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, scope="conv1")
                net = slim.max_pool2d(net, scope="poo1")
                net = slim.repeat(net, 2, slim.conv2d, 128, scope="conv2")
                net = slim.max_pool2d(net, scope="poo2")
                net = slim.flatten(net)
                net = slim.fully_connected(net, 1024, scope="fc3")
                net = slim.dropout(net, 0.5, scope="dropout3")
                net = slim.fully_connected(net, 256, scope="fc4")
                net = slim.dropout(net, 0.5, scope="dropout4")
                net = slim.fully_connected(net, 10, activation_fn=None, scope="linear")
                outputs = slim.softmax(net, scope="softmax4")
                return outputs

g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="images")

    with tf.variable_scope("slim", [inputs]):
        outputs = slim_model(inputs)

    print("Parameters")
    print_variables(slim.get_variables("slim"))
    print("Input / Output")
    print_variables([inputs, outputs])


print("---")
print("Argument Scoping for Collecting Layer Endpoints")
g = tf.Graph()
with g.as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="images")
    with tf.variable_scope("slim", [inputs]) as vs:
        end_points_collection = vs.original_name_scope + "_endpoints"
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            outputs = slim_model(inputs)

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

    print("Parameters")
    print_variables(slim.get_variables("slim"))
    print("Input / Output")
    print_variables([inputs, outputs])
    print("Layers")
    print_layers(end_points)
