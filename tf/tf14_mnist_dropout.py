import tensorflow as tf
import matplotlib.pyplot as plt
import random

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
w1 = tf.Variable(tf.random_normal([784, 10]), name='weight')
b1 = tf.Variable(tf.random_normal([10]), name='bias')
layer1 = tf.nn.sigmoid(tf.matmul(x, w1) + b1)

# layer 2
w2 = tf.Variable(tf.random_normal([10, 15]), name='weight')
b2 = tf.Variable(tf.random_normal([15]), name='bias')
layer2 = tf.nn.sigmoid(tf.matmul(layer1, w2) + b2)

# layer 3
w3 = tf.Variable(tf.random_normal([15, 20]), name='weight')
b3 = tf.Variable(tf.random_normal([20]), name='bias')
layer3 = tf.nn.sigmoid(tf.matmul(layer2, w3) + b3)

# dropout
drop = tf.nn.dropout(layer3, keep_prob)

w10 = tf.Variable(tf.random_normal([20, 10]), name='weight')
b10 = tf.Variable(tf.random_normal([10]), name='bias')
hypothesis = tf.nn.softmax(tf.matmul(drop, w10) + b10)

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