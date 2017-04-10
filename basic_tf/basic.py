import tensorflow as tf

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")

f = x * x * y + y + 2

# Basic usage
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)

result = sess.run(f)

print(result)

sess.close()

# DRY: avoid calling sess.run many times
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)

# DRY: use global_variables_initializer()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)

# Note: In Jupyter notebook or Python shell, use tf.InteractiveSession()
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

# Manage Graphs.
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())  # --> True: x1 in default_graph()

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is tf.get_default_graph())  # --> False: because x2 is in graph. not in default_graph()

# Lifecycle
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    # Eval y, z. Since we call eval 2 separated times, x, w must be computed twice
    print(y.eval())
    print(z.eval())
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)


