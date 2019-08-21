import tensorflow as tf
import numpy as np
import os
tf.set_random_seed(777)

dir_path = os.getcwd()
x_data = np.load(os.path.join(dir_path, 'etc/data/boston_housing_x.npy'))
y_data = np.load(os.path.join(dir_path, 'etc/data/boston_housing_y.npy'))

y_data = y_data.reshape(-1, 1)

# ...Scaler 사용해서 정규화
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()

scaler.fit(x_data)
x_data = scaler.transform(x_data)


from sklearn.model_selection import train_test_split
x_data, x_test, y_data, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

# layer 1
w1 = tf.get_variable('w1', shape=[13, 50], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([50]))
layer1 = tf.nn.leaky_relu(tf.matmul(x, w1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob)

# layer 2
w2 = tf.get_variable('w2', shape=[50, 100], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([100]))
layer2 = tf.nn.leaky_relu(tf.matmul(layer1, w2) + b2)

# output
w = tf.get_variable('w', shape=[100, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.nn.leaky_relu(tf.matmul(layer2, w) + b)

cost = tf.reduce_mean(tf.square(hypothesis - y))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.006).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10501):
        _, cost_val = sess.run([train, cost], feed_dict={x: x_data, y: y_data, keep_prob: 0.9})

        if step % 100 == 0:
            print(step, 'cost: ', cost_val)


    pred = sess.run([hypothesis], feed_dict={x: x_test, keep_prob: 0.9})
    pred = np.array(pred)
    pred = pred.reshape(-1,)
    y_test = np.array(y_test)
    y_test = y_test.reshape(-1,)

    # RMSE 구하기
    from sklearn.metrics import mean_squared_error
    def RMSE(ytest, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))
    print('RMSE : ', RMSE(y_test, pred))   # 낮을수록 좋음

    # R2 구하기. R2는 결정계수로 1에 가까울수록 좋음
    from sklearn.metrics import r2_score
    r2_y_predict = r2_score(y_test, pred)
    print('R2 : ', r2_y_predict)