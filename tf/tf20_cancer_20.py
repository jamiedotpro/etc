import tensorflow as tf
import random
import numpy as np
import os

tf.set_random_seed(777)

dir_path = os.getcwd()
cancer_data_file = os.path.join(dir_path, 'etc/data/cancer_data.npy')
cancer_d = np.load(cancer_data_file)
print('cancer',cancer_d.shape)

x_data = cancer_d[:, :30]
y_data = cancer_d[:, [-1]]

print('cancer_data', x_data.shape)
print('cancer_target', y_data.shape)

from sklearn.model_selection import train_test_split
x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)


x = tf.placeholder(tf.float32, [None, 30])
y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32)

layer1 = tf.layers.dense(x, 60, activation=tf.nn.relu)
layer1 = tf.layers.dense(layer1, 90, activation=tf.nn.relu)
layer1 = tf.layers.dense(layer1, 50, activation=tf.nn.relu)
layer1 = tf.layers.dropout(layer1, keep_prob)
layer1 = tf.layers.dense(layer1, 20, activation=tf.nn.relu)
layer1 = tf.layers.dense(layer1, 1, activation=tf.nn.relu)

hypothesis = tf.nn.sigmoid(layer1)


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _, acc_val = sess.run([cost, train, accuracy], feed_dict={x: x_data, y: y_data, keep_prob: 0.9})
        if step % 200 == 0:
            print(step, cost_val, acc_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x: x_test, y: y_test, keep_prob: 0.9})
    print('\nHypothesis: ', h, '\nCorrect (y): ', c, '\nAccuracy: ', a)