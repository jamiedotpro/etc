# 유방암 numpy 파일을 이용하여, 코딩하시오.
import numpy as np
import os

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


# Logistic Regression Classifier
import tensorflow as tf
tf.set_random_seed(777)

x = tf.placeholder(tf.float32, shape=[None, 30])
y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

# layer 1
w1 = tf.get_variable('w1', shape=[30, 10], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([10]))
layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
# layer1 = tf.nn.dropout(layer1, keep_prob)

# output
w = tf.get_variable('w', shape=[10, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(layer1, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # == model.compile(loss='binary_crossentropy'...
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data, keep_prob: 1})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x: x_test, y: y_test, keep_prob: 1})
    print('\nHypothesis: ', h, '\nCorrect (y): ', c, '\nAccuracy: ', a)