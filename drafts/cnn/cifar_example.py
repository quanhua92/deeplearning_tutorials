import os
import sys
import tarfile
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from urllib.request import urlretrieve


batch_size = 128
output_every = 50
generations = 20000
eval_every = 100
image_height = 32
image_width = 32
crop_height = 24
crop_width = 24

num_channels = 3
num_targets = 10
data_dir = "/tmp/temp_cifar"
extract_folder = "cifar-10-batches-bin"

learning_rate = 0.1
lr_decay = 0.9
num_gens_to_wait = 250.

image_vec_length = image_height * image_width * num_channels
record_length = 1 + image_vec_length

# Download cifar images

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

cifar10_url = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
data_file = os.path.join(data_dir, "cifar-10-binary.tar.gz")
if not os.path.isfile(data_file):
    filepath, _ = urlretrieve(cifar10_url, data_file)
    tarfile.open(filepath, "r:gz").extractall(data_dir)


def read_cifar_files(filename_queue, distort_images=True):
    reader = tf.FixedLengthRecordReader(record_bytes=record_length)
    key, record_string = reader.read(filename_queue)
    record_bytes = tf.decode_raw(record_string, tf.uint8)
    image_label = tf.cast(tf.slice(record_bytes, [0], [1]), tf.int32)
    image_extracted = tf.reshape(tf.slice(record_bytes, [1], [image_vec_length]), [num_channels, image_height, image_width])
    image_uint8image = tf.transpose(image_extracted, [1, 2, 0])
    reshaped_image = tf.cast(image_uint8image, tf.float32)
    # randomly crop image
    final_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, crop_width, crop_height)
    if distort_images:
        # Random distort
        final_image = tf.image.random_flip_left_right(final_image)
        final_image = tf.image.random_brightness(final_image, max_delta=63)
        final_image = tf.image.random_contrast(final_image, lower=0.2, upper=1.8)
    # Normalize
    final_image = tf.image.per_image_standardization(final_image)
    return final_image, image_label


def input_pipeline(batch_size, is_training=True):
    if is_training:
        files = [os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(i)) for i in range(1, 6)]
    else:
        files = [os.path.join(data_dir, extract_folder, 'test_batch.bin')]
    filename_queue = tf.train.string_input_producer(files)
    image, label = read_cifar_files(filename_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity, min_after_dequeue)
    return image_batch, label_batch


def cifar_cnn_model(input_images, batch_size, is_training=True):
    def truncated_normal_var(name, shape, dtype):
        return tf.get_variable(name, shape, dtype, initializer=tf.truncated_normal_initializer(stddev=0.05))
    def zero_var(name, shape, dtype):
        return tf.get_variable(name, shape, dtype, initializer=tf.constant_initializer(0.0))

    # First conv layer
    with tf.variable_scope("conv1"):
        conv1_kernel = truncated_normal_var("conv_kernel1", shape=[5, 5, 3, 64], dtype=tf.float32)
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 1, 1, 1], padding="SAME")
        conv1_bias = zero_var(name="conv_bias1", shape=[64], dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        relu_conv1 = tf.nn.relu(conv1_add_bias)
    pool1 = tf.nn.max_pool(relu_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool_layer1")

    # Local response normalization
    norm1 = tf.nn.lrn(pool1, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name="norm1")
    with tf.variable_scope('conv2'):
        conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 64], dtype=tf.float32)
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        conv2_bias = zero_var(name='conv_bias2', shape=[64], dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        relu_conv2 = tf.nn.relu(conv2_add_bias)
    pool2 = tf.nn.max_pool(relu_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer2')
    norm2 = tf.nn.lrn(pool2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')
    reshaped_output = tf.reshape(norm2, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    with tf.variable_scope('full1') as scope:
        full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full_bias1 = zero_var(name='full_bias1', shape=[384], dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))
    with tf.variable_scope('full2') as scope:
        full_weight2 = truncated_normal_var(name='full_mult2', shape=[384, 192], dtype=tf.float32)
        full_bias2 = zero_var(name='full_bias2', shape=[192], dtype=tf.float32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))
    with tf.variable_scope('full3') as scope:
        full_weight3 = truncated_normal_var(name='full_mult3', shape=[192, num_targets], dtype=tf.float32)
        full_bias3 = zero_var(name='full_bias3', shape=[num_targets], dtype=tf.float32)
        final_output = tf.add(tf.matmul(full_layer2, full_weight3), full_bias3)
    return final_output


def cifar_loss(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def train_step(loss, global_step):
    model_learning_rate = tf.train.exponential_decay(learning_rate, global_step, num_gens_to_wait, lr_decay, staircase=True)
    optimizer = tf.train.AdamOptimizer(model_learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def accuracy_of_batch(logits, targets):
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy

images, targets = input_pipeline(batch_size, is_training=True)
test_images, test_targets = input_pipeline(batch_size, is_training=False)

with tf.variable_scope("model_def") as scope:
    model_output = cifar_cnn_model(images, batch_size)
    scope.reuse_variables()
    test_output = cifar_cnn_model(test_images, batch_size)

loss = cifar_loss(model_output, targets)
accuracy = accuracy_of_batch(test_output, test_targets)
global_step = tf.Variable(0, trainable=False)
train_op = train_step(loss, global_step)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    for i in tqdm(range(generations)):
        _, loss_value = sess.run([train_op, loss])
        if (i + 1) % output_every == 0:
            print("Generatation {}: Loss = {:.5f}".format(i + 1, loss_value))
        if (i + 1 ) % eval_every == 0:
            test_acc, loss_value = sess.run([accuracy, loss])
            print("---- Test accuracy {}: Loss = {:.5f}".format(test_acc, loss_value))

