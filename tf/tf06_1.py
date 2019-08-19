# Linear Regression
import tensorflow as tf
tf.set_random_seed(777) # for reproducibility

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

hypothesis = x * w + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        _, cost_val, w_val, b_val = sess.run(
            [train, cost, w, b], feed_dict={x: [1, 2, 3], y: [1, 2, 3]}
        )
        if step % 20 == 0:
            print(step, cost_val, w_val, b_val)

    # Testing our model
    print(sess.run(hypothesis, feed_dict={x: [5]}))
    print(sess.run(hypothesis, feed_dict={x: [2.5]}))
    print(sess.run(hypothesis, feed_dict={x: [1.5, 3.5]}))