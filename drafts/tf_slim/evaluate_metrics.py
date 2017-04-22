import tensorflow as tf
from tensorflow.contrib import slim


# Streaming metrics
value = ...
mean_value, update_op = slim.metrics.streaming_mean(values)
sess.run(tf.local_variables_initializer())

for i in range(num_of_batches):
    print("Mean at batch %d: %f" % (i, update_op.eval()))

print("Final Mean: %f" % mean_value.eval())

# Define multiple metrics
# - Load data
images, labels = LoadTestData(...)

predictions = MyModel(images)

mae_value_op, mae_update_op = slim.metrics.streaming_mean_absolute_error(predictions, labels)
mre_value_op, mre_update_op = slim.metrics.streaming_mean_relative_error(predictions, labels, labels)
p1_value_op, p1_update_op = slim.metrics.streaming_percentage_less(mre_value_op, 0.3)

# - Mean Aggregation
value_ops, update_ops = slim.metrics.aggregate_metrics(
    slim.metrics.streaming_mean_absolute_error(predictions, labels),
    slim.metrics.streaming_mean_squared_error(predictions, labels)
)

# Dictionary Aggregations
# - Load data
images, labels = load_data()
logits = MyModel(images)
predictions = tf.argmax(logits, 1)

names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/Accuracy": slim.metrics.streaming_accuracy(predictions, labels),
    "eval/Recall": slim.metrics.streaming_recall(predictions, labels),
    "eval/Recall@3": slim.metrics.streaming_recall_at_k(predictions, labels, 3),
    "eval/Precision": slim.metrics.streaming_precision(predictions, labels)
})

num_batches = 100
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for batch_id in range(num_batches):
        sess.run(names_to_updates.values())

    metric_values = sess.run(names_to_updates.values())
    for metric, value in zip(names_to_updates.keys, metric_values):
        print("Metric %s has value : %f" % (metric, value))