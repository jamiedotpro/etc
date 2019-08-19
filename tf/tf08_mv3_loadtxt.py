# Multi-variable linear regression 3
import tensorflow as tf
import numpy as np
import os

tf.set_random_seed(777)

dir_path = os.getcwd()
test_score_file = os.path.join(dir_path, 'etc/data/data-01-test-score.csv')
xy = np.loadtxt(test_score_file, delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data, '\nx_data shape:', x_data.shape)
print(x_data, '\ny_data shape:', y_data.shape)

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                    feed_dict={x: x_data, y: y_data})
    if step % 10 == 0:
        print(step, 'Cost: ', cost_val, '\nPrediction:\n', hy_val)
