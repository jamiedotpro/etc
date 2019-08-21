import tensorflow as tf
import matplotlib.pyplot as plt
import random

# w1 = tf.get_variable('w1', shape=[?, ?], initializer=tf.random_uniform_initializer())
# b1 = tf.Variable(tf.random_normal([512]))
# l1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# l1 = tf.nn.dropout(l1, keep_prop=keep_prob)

# 아래는 변수를 구성하는 시점에 초기화한다
# tf.constant_initializer()         # 상수
# tf.zeros_initializer()            # 0
# tf.random_uniform_initializer()   # 랜덤
# tf.random_normal_initializer()    # 랜덤
# tf.contrib.layers.xavier_initializer()

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

print(mnist.train.images)
print(mnist.test.labels)
print(mnist.train.images.shape)
print(mnist.test.labels.shape)
print(type(mnist.train.images))

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# layer 1
w1 = tf.get_variable('w1', shape=[784, 100], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([100]))
layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob)

# layer 2
w2 = tf.get_variable('w2', shape=[100, 40], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([40]))
layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)

# layer 3
w3 = tf.get_variable('w3', shape=[40, 200], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([200]))
layer3 = tf.nn.sigmoid(tf.matmul(layer2, w3) + b3)

# layer 4
w4 = tf.get_variable('w4', shape=[200, 40], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([40]))
layer4 = tf.nn.sigmoid(tf.matmul(layer3, w4) + b4)

# layer 5
w5 = tf.get_variable('w5', shape=[40, 30], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([30]))
layer5 = tf.nn.sigmoid(tf.matmul(layer4, w5) + b5)

# dropout
# drop = tf.nn.dropout(layer3, keep_prob)

w10 = tf.get_variable('w10', shape=[30, 10], initializer=tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([10]), name='bias')
hypothesis = tf.nn.softmax(tf.matmul(layer5, w10) + b10)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size) # 55000 / 100 = 550

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_val = sess.run([train, cost], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.9})
            avg_cost += cost_val / num_iterations

        print('Epoch: {:04d}, Cost: {:.9f}'.format(epoch + 1, avg_cost))

    print('Learning finished')

    # Test the model using test sets
    print(
        'Accuracy: ',
        accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 0.9}),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print('Label: ', sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print('Prediction: ', sess.run(tf.argmax(hypothesis, 1), feed_dict={x: mnist.test.images[r : r + 1], keep_prob: 0.9}),)

    plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28), cmap='Greys', interpolation='nearest',)
    plt.show()