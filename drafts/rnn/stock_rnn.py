import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.rnn import BasicRNNCell, OutputProjectionWrapper

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
cell = OutputProjectionWrapper(basic_cell, output_size=n_outputs)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

logits = fully_connected(states, n_outputs, activation_fn=None)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

learning_rate = 0.001
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate)

train_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 10
iterations = 10000
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iter in range(iterations):
            # TODO: Add data
            X_batch, y_batch = [], []  # Feed some data here!
            sess.run([train_op], feed_dict={X: X_batch, y: y_batch})
        acc_train = sess.run([accuracy], feed_dict={X: X_batch, y: y_batch})

