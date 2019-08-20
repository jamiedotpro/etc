import tensorflow as tf
import numpy as np

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
                [816, 820.958984, 1008100, 815.48999, 819.23999],
                [819.359985, 823, 1188100, 818.469971, 818.97998],
                [819, 823, 1198100, 816, 820.450012],
                [811.700012, 815.25, 1098100, 809.780029, 813.669983],
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[:, :-1]
y_data = xy[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([4, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
optimizer = tf.train.AdamOptimizer(learning_rate=0.3)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(5101):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={x: x_data, y: y_data})
    print(step, 'Cost: ', cost_val, '\nPrediction:\n', hy_val)


pred = sess.run([hypothesis], feed_dict={x: x_data})
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