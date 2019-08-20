import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import pandas as pd
import os
tf.set_random_seed(777)

dir_path = os.getcwd()
zoo_file = os.path.join(dir_path, 'etc/data/data-04-zoo.csv')
xy = np.loadtxt(zoo_file, delimiter=',', dtype=np.float32)  # 한가지 자료형일 때
# xy = pd.read_csv(zoo_file)    # 숫자 문자 등 자료형이 섞여 있을 때, 열이름이 있을 때
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

y_data = np_utils.to_categorical(y_data)

x = tf.placeholder(tf.float32, shape=[None, 16])
y = tf.placeholder(tf.float32, shape=[None, 7])

w = tf.Variable(tf.random_normal([16, 7]), name='weight')
b = tf.Variable(tf.random_normal([7]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict={x: x_data, y: y_data})

        if step % 200 == 0:
            print(step, cost_val)


    prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    print('Accuracy: ', accuracy.eval({x: x_data, y: y_data}))
