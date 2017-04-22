import tensorflow as tf
from tensorflow.contrib import slim

images, labels = load_data()
logits = MyModel(images)
predictions = tf.argmax(logits, 1)

names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    "eval/Accuracy": slim.metrics.streaming_accuracy(predictions, labels),
    "eval/Recall": slim.metrics.streaming_recall(predictions, labels),
    "eval/Recall@3": slim.metrics.streaming_recall_at_k(predictions, labels, 3),
    "eval/Precision": slim.metrics.streaming_precision(predictions, labels)
})

# define thet summaries to write
for metric_name, metric_value in names_to_values.iteritems():
    tf.summary.scalar(metric_name, metric_value)

tf.summary.scalar(..)
tf.summary.histogram(...)

checkpoint_dir = "/tmp/my_model_dir/"
log_dir = "/tmt/my_model_eval/"

num_evals = 1000

slim.get_or_create_global_step()

slim.evaluation.evaluation_loop(
    master='',
    checkpoint_dir=checkpoint_dir,
    logdir=log_dir,
    num_evals=num_evals,
    eval_op=names_to_updates.values(),
    summary_op=tf.summary.merge(summary_ops),
    eval_interval_secs=600
)

