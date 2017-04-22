import tensorflow as tf
from tensorflow.contrib.layers import dropout

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

# Method 01: Gaussian noise
X_noisy = X + tf.random_normal(tf.shape(X))
# We must use tf.shape, not X.get_shape() because X is just partially defined [None, n_inputs]

# Method 02: Dropout
X_drop = dropout(X, keep_prob=0.7, is_training=is_training)

# define hidden and outputs

loss = tf.reduce_mean(tf.square(outputs - X))  # MSE
# optimize loss

