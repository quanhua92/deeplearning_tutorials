import tensorflow as tf

loss=None  # put your loss here

threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads_and_vars = optimizer.compute_gradients(loss=loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var) for grad, var in grads_and_vars]
train_op = optimizer.apply_gradients(capped_gvs)
