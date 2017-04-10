import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


def max_norm_regularizer(threshold, axes=1, name="max_norm", collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None  # No regularization loss
    return max_norm


max_norm_reg = max_norm_regularizer(threshold=1.0)
hidden1 = fully_connected(X, n_hidden1, scope="hidden1", weights_regularizer=max_norm_reg)

# Get handle to clip_weights

clip_all_weights = tf.get_collection("max_norm")

with tf.Session() as sess:
    sess.run(train_op, feed_dict={X: X_batch, y: y_batch})
    # Run the op to clip weights
    sess.run(clip_all_weights)