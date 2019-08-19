# Minimizing Cost
import tensorflow as tf
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [1, 2, 3]

w = tf.placeholder(tf.float32)

# Our hypothesis for linear model x * w
hypothesis = x * w

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Variables for plotting cost function
w_history = []
cost_history = []

# Launch the graph in a session
with tf.Session() as sess:
    for i in range(-30, 50):
        curr_w = i * 0.1
        curr_cost = sess.run(cost, feed_dict={w: curr_w})

        w_history.append(curr_w)
        cost_history.append(curr_cost)

# Show the cost function
plt.plot(w_history, cost_history)
plt.show()