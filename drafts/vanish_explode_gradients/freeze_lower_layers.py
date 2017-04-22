import tensorflow as tf

# We will train all layers except hidden[12]. Therefore, Layers 1 and 2 are frozen
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden[34]|outputs")
loss = None  # Your loss is here

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss, var_list=train_vars)