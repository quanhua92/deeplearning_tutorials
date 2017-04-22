import tensorflow as tf
import numpy as np
import cv2


n_inputs = 28*28  # MNIST Size
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")


def fully_connected(X, n_neurons, name, activation=None):
    """
    Fully connected layers. You can also use contrib.layers instead of this function
    from tensorflow.contrib.layers import fully_connected
    """
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z

with tf.name_scope("dnn"):
    hidden1 = fully_connected(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = fully_connected(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = fully_connected(hidden2, n_outputs, "outputs")


init = tf.global_variables_initializer()
saver = tf.train.Saver()


def load_and_preprocess(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = 255 - image  # Need invert because MNIST train data has white text, black background
    image = cv2.resize(image, (28, 28)) / 255.
    image = np.reshape(image, (1, 28*28))
    return image

test_images = []
img = load_and_preprocess("../images/test_mnist.png")
test_images.append(img)
img = load_and_preprocess("../images/test_mnist_2.png")
test_images.append(img)

with tf.Session() as sess:
    init.run()
    save_path = saver.restore(sess, "/tmp/tf_mnist")

    for image in test_images:
        logits_value = logits.eval(feed_dict={X: image})
        y_pred = np.argmax(logits_value, axis=1)
        print("logits \n",logits_value)
        print("y predicted: \n", y_pred)




