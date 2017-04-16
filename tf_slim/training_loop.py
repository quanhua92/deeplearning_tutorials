import tensorflow as tf
from tensorflow.contrib import slim

images, labels = LoadData(...)

predictions = MyModel(images)

slim.losses.log_loss(predictions, labels)

total_loss = slim.losses.get_total_loss()

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train_op = slim.learning.create_train_op(total_loss, optimizer)

logdir = "/logdir/path"

final_loss = slim.learning.train(train_op, logdir,
                                 number_of_steps=1000,  # number of gradient steps
                                 save_summaries_secs=60,  # compute summaries every 60 seconds
                                 save_interval_secs=300)  # save model checkpoint every 300 seconds

print("Last batch loss is ", final_loss)


