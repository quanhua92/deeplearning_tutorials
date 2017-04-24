import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
from utils.debug_print import *

n_inputs = 28*28  # MNIST Size
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10


def slim_model(inputs):
    with tf.variable_scope("slim-fc", [inputs]) as vs:
        end_points_collection = vs.original_name_scope + "_endpoints"
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(stddev=0.1),
                            biases_initializer=tf.constant_initializer(1.0),
                            weights_regularizer=slim.l2_regularizer(0.05),
                            outputs_collections=end_points_collection):
            net = slim.fully_connected(inputs, num_outputs=n_hidden1,
                                       weights_initializer=tf.random_normal_initializer(stddev=0.1),
                                       scope="fc1")
            net = slim.fully_connected(net, num_outputs=n_hidden2,
                                       weights_initializer=tf.random_normal_initializer(stddev=0.1),
                                       scope="fc2")
            net = slim.fully_connected(net, num_outputs=n_outputs,
                                       weights_initializer=tf.random_normal_initializer(stddev=0.1),
                                       activation_fn=None, scope="outputs")
        logits = slim.softmax(net, scope="softmax")
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return logits, end_points

images, labels = mnist.LoadData()

logits, end_points = slim_model(inputs)
print_variables(slim.get_variables("slim-fc"))
print_variables([inputs, logits])
print_layers(end_points)

slim.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)

total_loss = slim.losses.get_total_loss()

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

train_op = slim.learning.create_train_op(total_loss, optimizer)

logdir="/tmp/logdir/mnist"

final_loss = slim.learning.train(train_op, logdir,
                                 number_of_steps=1000,
                                 save_summaries_secs=60,
                                 save_interval_secs=300)

print("Last batch loss is ", final_loss)

