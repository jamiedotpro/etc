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
w1 = tf.get_variable('w1', shape=[13, 100], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([100]))
layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
# layer1 = tf.nn.dropout(layer1, keep_prob)

# # layer 2
# w2 = tf.get_variable('w2', shape=[50, 100], initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.Variable(tf.random_normal([100]))
# layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)

# # layer 3
# w3 = tf.get_variable('w3', shape=[100, 50], initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.Variable(tf.random_normal([50]))
# layer3 = tf.nn.sigmoid(tf.matmul(layer2, w3) + b3)

# # layer 4
# w4 = tf.get_variable('w4', shape=[64, 32], initializer=tf.contrib.layers.xavier_initializer())
# b4 = tf.Variable(tf.random_normal([32]))
# layer4 = tf.nn.sigmoid(tf.matmul(layer3, w4) + b4)

# # layer 5
# w5 = tf.get_variable('w5', shape=[64, 128], initializer=tf.contrib.layers.xavier_initializer())
# b5 = tf.Variable(tf.random_normal([128]))
# layer5 = tf.nn.sigmoid(tf.matmul(layer4, w5) + b5)

# # layer 6
# w6 = tf.get_variable('w6', shape=[256, 512], initializer=tf.contrib.layers.xavier_initializer())
# b6 = tf.Variable(tf.random_normal([512]))
# layer6 = tf.nn.sigmoid(tf.matmul(layer5, w6) + b6)

# # layer 7
# w7 = tf.get_variable('w7', shape=[512, 32], initializer=tf.contrib.layers.xavier_initializer())
# b7 = tf.Variable(tf.random_normal([32]))
# layer7 = tf.nn.sigmoid(tf.matmul(layer6, w7) + b7)

# output
w = tf.get_variable('w', shape=[100, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(layer1, w) + b)

cost = tf.reduce_mean(tf.square(hypothesis - y))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
train = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val = sess.run([train, cost], feed_dict={x: x_data, y: y_data, keep_prob: 0.9})

        if step % 20 == 0:
            print(step, 'cost: ', cost_val)


    pred = sess.run([hypothesis], feed_dict={x: x_data, keep_prob: 0.9})
    pred = np.array(pred)
    pred = pred.reshape(-1,)
    y_data = np.array(y_data)
    y_data = y_data.reshape(-1,)

    # RMSE 구하기
    from sklearn.metrics import mean_squared_error
    def RMSE(ytest, y_predict):
        return np.sqrt(mean_squared_error(y_data, y_predict))
    print('RMSE : ', RMSE(y_data, pred))   # 낮을수록 좋음

    # R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
    from sklearn.metrics import r2_score
    r2_y_predict = r2_score(y_data, pred)
    print('R2 : ', r2_y_predict)