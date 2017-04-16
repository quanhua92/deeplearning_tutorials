import tensorflow as tf
from tensorflow.contrib import slim


# Gradient norm clipping
# L2-norm greater than 4 will be clipped to avoid 'exploding gradients'
train_op = slim.learning.create_train_op(total_loss, optimizer,
                                         clip_gradient_norm=4)


# Gradient scaling
# scaling the gradient redistributes the weight importance

# map from variable name to scaling coefficent
gradient_multipliers = {
    'conv1/weights' : 2.4,
    'fc8/weights': 5.1
}

train_op = slim.learning.create_train_op(total_loss, optimizer,
                                         gradient_multipliers=gradient_multipliers)


# Modify the Update Operation
# - override the default update ops
train_op = slim.learning.create_train_op(total_loss, optimizer,
                                         update_ops=my_other_update_ops)
# - remove update op. for example, batch normalizing layer requires non-gradient updates during training (update moving
# mean / variance)
train_op = slim.learning.create_train_op(total_loss, optimizer,
                                         update_ops=[])



