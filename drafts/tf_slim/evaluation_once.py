checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

metric_values = slim.evaluation.evaluation_once(
    master='',
    checkpoint_path=checkpoint_path,
    logdir=log_dir,
    num_evals=num_evals,
    eval_op=names_to_updates.values(),
    final_op=names_to_values.values(),
)

for metric, value in zip(names_to_updates.keys, metric_values):
    print("Metric %s has value : %f" % (metric, value))