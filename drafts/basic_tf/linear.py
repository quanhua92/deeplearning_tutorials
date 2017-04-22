import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from datetime import datetime

housing = fetch_california_housing()
m, n = housing.data.shape

scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)

housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
housing_target = housing.target.reshape(-1, 1)

n_epochs = 500
learning_rate = 0.0001
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
logdir = "/tmp/tf_logs/{}".format(now)
print("Logdir = ", logdir)

# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")  # Full batch
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")  # full batch

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")  # Mini batch
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")  # Mini batch


def fetch_batch(epoch, batch_index, batch_size):
    start_idx = batch_index * batch_size
    end_idx = start_idx + batch_size
    X_batch = housing_data_plus_bias[start_idx : end_idx]
    y_batch = housing_target[start_idx : end_idx]
    return X_batch, y_batch

# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)  # Normal Equation
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)  # Manual diff
gradients = tf.gradients(mse, [theta])[0]  # Auto diff
# train_op = tf.assign(theta, theta - learning_rate * gradients)  # Manual optimize
train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)  # Use optimizer

mse_summary = tf.summary.scalar("MSE", mse)
file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # saver.restore(sess, "/tmp/model-final.ckpt")

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(train_op, feed_dict={X: X_batch, y: y_batch})

            if epoch % 100 == 0 and batch_index == n_batches - 1:
                mse_value = sess.run(mse, feed_dict={X: X_batch, y: y_batch})
                print("Epoch", epoch, "MSE = ", mse_value)
                save_path = saver.save(sess, "/tmp/model.ckpt")
                print("Saved: ", save_path)
            if batch_index % 10 == 0:
                summary_str = sess.run(mse_summary, feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
    best_theta = theta.eval()
    print(best_theta)
    save_path = saver.save(sess, "/tmp/model-final.ckpt")
    print("Saved Final Model: ", save_path)

file_writer.close()
