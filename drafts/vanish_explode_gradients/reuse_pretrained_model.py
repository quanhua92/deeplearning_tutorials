import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

n_inputs = 28*28  # MNIST Size
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = fully_connected(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = fully_connected(hidden2, n_outputs, "outputs")

with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")

learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# Assume that you only want to restore hidden 1
reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[1]")
reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
original_saver = tf.Saver(reuse_vars_dict)  # saver to restore the original model

new_saver = tf.Saver()  # saver to save the new model


mnist = input_data.read_data_sets("/tmp/data/")
n_epochs = 50
batch_size = 50

with tf.Session() as sess:
    init.run()
    original_saver.restore("/tmp/tf_mnist/origin_model.ckpt")
    for epoch in range(n_epochs):
        for iter in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy: ", acc_train, " Test accuracy: ", acc_test)
    save_path = new_saver.save(sess, "/tmp/tf_mnist")
    print("Saved: ", save_path)



