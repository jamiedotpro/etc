import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# x, y, w, b, hypothesis, cost, train
# sigmoid 사용
# predict, accuracy

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

# 레이어1
w1 = tf.Variable(tf.random_normal([2, 100]), name='weight')  # [입력, 출력]
b1 = tf.Variable(tf.random_normal([100]), name='bias')   # [출력]
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

# 레이어2
w2 = tf.Variable(tf.random_normal([100, 200]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b2 = tf.Variable(tf.random_normal([200]), name='bias') # [출력]
layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)   # 이전 레이어를 넣어준다

# 레이어3
w3 = tf.Variable(tf.random_normal([200, 50]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b3 = tf.Variable(tf.random_normal([50]), name='bias') # [출력]
layer3 = tf.sigmoid(tf.matmul(layer2, w3) + b3)   # 이전 레이어를 넣어준다

# 레이어4
w4 = tf.Variable(tf.random_normal([50, 60]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b4 = tf.Variable(tf.random_normal([60]), name='bias') # [출력]
layer4 = tf.sigmoid(tf.matmul(layer3, w4) + b4)   # 이전 레이어를 넣어준다

# 레이어5
w5 = tf.Variable(tf.random_normal([60, 10]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b5 = tf.Variable(tf.random_normal([10]), name='bias') # [출력]
layer5 = tf.sigmoid(tf.matmul(layer4, w5) + b5)   # 이전 레이어를 넣어준다

# 레이어6
w6 = tf.Variable(tf.random_normal([10, 15]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b6 = tf.Variable(tf.random_normal([15]), name='bias') # [출력]
layer6 = tf.sigmoid(tf.matmul(layer5, w6) + b6)   # 이전 레이어를 넣어준다

# 레이어7
w7 = tf.Variable(tf.random_normal([15, 20]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b7 = tf.Variable(tf.random_normal([20]), name='bias') # [출력]
layer7 = tf.sigmoid(tf.matmul(layer6, w7) + b7)   # 이전 레이어를 넣어준다

# 레이어8
w8 = tf.Variable(tf.random_normal([20, 5]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b8 = tf.Variable(tf.random_normal([5]), name='bias') # [출력]
layer8 = tf.sigmoid(tf.matmul(layer7, w8) + b8)   # 이전 레이어를 넣어준다

# 레이어9
w9 = tf.Variable(tf.random_normal([5, 10]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b9 = tf.Variable(tf.random_normal([10]), name='bias') # [출력]
layer9 = tf.sigmoid(tf.matmul(layer8, w9) + b9)   # 이전 레이어를 넣어준다

# 레이어10
w10 = tf.Variable(tf.random_normal([10, 1]), name='weight')  # [입력 (이전 레이어의 출력), 출력]
b10 = tf.Variable(tf.random_normal([1]), name='bias') # [출력]
hypothesis = tf.sigmoid(tf.matmul(layer9, w10) + b10)   # 이전 레이어를 넣어준다


cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, w_val = sess.run([train, cost, w10], feed_dict={x: x_data, y: y_data})

        if step % 100 == 0:
            print(step, cost_val, w_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})

    print('\nHypothesis: ', h, '\nCorrect: ', c, '\nAccuracy: ', a)