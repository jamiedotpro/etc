import tensorflow as tf

# tf Graph Input
x = [1, 2, 3]
y = [1, 2, 3]

# Set wrong model weights
w = tf.Variable(5.0)

# Linear model
hypothesis = x * w

# cost.loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Minimize: Gradient Descent Optimzier
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        _, w_val = sess.run([train, w])
        print(step, w_val)