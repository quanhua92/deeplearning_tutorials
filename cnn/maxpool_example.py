import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt


# Load sample images
dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create graph
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

for i in range(output.shape[0]):
    fig = plt.figure()
    print("Image ", i, " Original Size ", dataset[i].shape, " Output Size ", output[i].shape)
    plt.imshow(output[i].astype(np.uint8))  # ith image
    fig.suptitle("Image %s " % i)
    plt.show()