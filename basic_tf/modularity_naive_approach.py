import tensorflow as tf

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.maximum(z1, 0, name="relu1")
relu2 = tf.maximum(z2, 0, name="relu2")

output = tf.add(relu1, relu2, name="output")

logdir = "/tmp/tf_modular/naive"
file_writer = tf.summary.FileWriter(logdir=logdir, graph=tf.get_default_graph())
file_writer.close()



