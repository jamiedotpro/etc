import tensorflow as tf
import numpy as np
import os
tf.set_random_seed(777)

dir_path = os.getcwd()
x_data = np.load(os.path.join(dir_path, 'etc/data/boston_housing_x.npy'))
y_data = np.load(os.path.join(dir_path, 'etc/data/boston_housing_y.npy'))

y_data = y_data.reshape(-1, 1)

from sklearn.model_selection import train_test_split
x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

# layer 1
w1 = tf.get_variable('w1', shape=[13, 128], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([128]))
layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
# layer1 = tf.nn.dropout(layer1, keep_prob)

# layer 2
w2 = tf.get_variable('w2', shape=[128, 256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)

# layer 3
w3 = tf.get_variable('w3', shape=[256, 64], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([64]))
layer3 = tf.nn.sigmoid(tf.matmul(layer2, w3) + b3)

# layer 4
w4 = tf.get_variable('w4', shape=[64, 128], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([128]))
layer4 = tf.nn.sigmoid(tf.matmul(layer3, w4) + b4)

# layer 5
w5 = tf.get_variable('w5', shape=[128, 32], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([32]))
layer5 = tf.nn.sigmoid(tf.matmul(layer4, w5) + b5)

# output
w = tf.get_variable('w', shape=[128, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(layer4, w) + b)

cost = tf.reduce_mean(tf.square(hypothesis - y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([train, cost], feed_dict={x: x_data, y: y_data, keep_prob: 0.9})

        if step % 20 == 0:
            print(step, cost_val)

    # Accuracy report
    # h, c, a = sess.run([hypothesis, predicted, accuracy],
    #                     feed_dict={x: x_test, y: y_test, keep_prob: 0.9})
    # print('\nHypothesis: ', h, '\nCorrect (y): ', c, '\nAccuracy: ', a)
