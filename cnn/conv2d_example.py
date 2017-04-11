import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt


# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

# Create graph
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
conv = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")

with tf.Session() as sess:
    output = sess.run(conv, feed_dict={X: dataset})

for i in range(filters.shape[3]):
    for j in range(output.shape[0]):
        fig = plt.figure()
        plt.imshow(output[j, :, :, i])  # jth image, ith filter
        fig.suptitle("Image %s Filter %s" % (j, i))
        plt.show()