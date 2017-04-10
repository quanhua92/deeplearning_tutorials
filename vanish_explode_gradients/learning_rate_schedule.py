import tensorflow as tf


initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 1/10
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

loss = None  # Your loss 's here

# Adam, AdaGrad, RMSProp automatically reduce learning_rate so we don't need this schedule in those optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
train_op = optimizer.minimize(loss, global_step=global_step)