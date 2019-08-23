import tensorflow as tf
import random
import numpy as np
from keras.utils import np_utils
import os

tf.set_random_seed(777)

dir_path = os.getcwd()
iris2_file = os.path.join(dir_path, 'etc/data/iris2_data.npy')
iris2_data = np.load(iris2_file)
print('iris2_data.shape: ', iris2_data.shape)

x_data = iris2_data[:, :-1]
y_data = iris2_data[:, [-1]]
print(x_data.shape, y_data.shape)

y_data = np_utils.to_categorical(y_data)

from sklearn.model_selection import train_test_split
x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)


x = tf.placeholder(tf.float32, [None, 4])
y = tf.placeholder(tf.float32, [None, 3])
keep_prob = tf.placeholder(tf.float32)

layer1 = tf.layers.dense(x, 32, activation=tf.nn.leaky_relu)
layer1 = tf.layers.dropout(layer1, keep_prob)
layer1 = tf.layers.dense(layer1, 16, activation=tf.nn.leaky_relu)
layer1 = tf.layers.dense(layer1, 3, activation=tf.nn.leaky_relu)

hypothesis = tf.nn.softmax(layer1)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, _, acc_val = sess.run([cost, optimizer, accuracy],
                                feed_dict={x: x_data, y: y_data, keep_prob: 0.9})
        print(step, cost_val, acc_val)

    # predict
    print('Prediction: ', sess.run(prediction, feed_dict={x: x_test, keep_prob: 0.9}))
    # Calculate the accuracy
    print('Accuracy: ', sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 0.9}))