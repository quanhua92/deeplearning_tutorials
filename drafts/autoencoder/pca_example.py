import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

n_inputs = 3
n_hidden = 2
n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=None)
outputs = fully_connected(hidden, n_outputs, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

X_train, X_test = []  # load dataset

n_iterations = 1000
codings = hidden

with tf.Session() as sess:
    init.run()
    for iter in range(n_iterations):
        train_op.run(feed_dict={X: X_train})
    codings_val = codings.eval(feed_dict={X: X_test})
