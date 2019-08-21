import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import os

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

tf.set_random_seed(777)

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 3])
keep_prob = tf.placeholder(tf.float32)

# layer 1
w1 = tf.get_variable('w1', shape=[4, 10], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([10]))
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob)

# output
w = tf.get_variable('w', shape=[10, 3], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([3]))
hypothesis = tf.nn.softmax(tf.matmul(layer1, w) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, w_val, _ = sess.run([cost, w, optimizer],
                                        feed_dict={x: x_data, y: y_data, keep_prob: 0.9})
        print(step, cost_val, w_val)

    # predict
    print('Prediction: ', sess.run(prediction, feed_dict={x: x_test, keep_prob: 0.9}))
    # Calculate the accuracy
    print('Accuracy: ', sess.run(accuracy, feed_dict={x: x_test, y: y_test, keep_prob: 0.9}))