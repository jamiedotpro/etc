import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])  # img 28*28*1 (black/white)
y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape={?, 28, 28, 1}
w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))  # [커널사이즈,, 흑백/컬러, 아웃풋]
print('w1: ', w1)
#   Conv    -> (?, 28, 28, 32)
#   Pool    -> (?, 14, 14, 32)
l1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')  # strides 몇칸씩 자를 것인가. 가로 세로 동일하게. strides=[안씀, 가로, 세로, 안씀]
print('l1: ', l1)
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   # 맥스풀링은 겹쳐지지 않은 상태로 지정범위내 높은 특성값을 출력한다. 반으로 크기가 줄어듬
print('l1: ', l1)

# L2 ImgIn shape=(?, 14, 14, 32)
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # [커널사이즈:몇by몇으로 자를지,, 이전 아웃풋, 아웃풋]
#   Conv    -> (?, 14, 14, 64)
#   Pool    -> (?, 7, 7, 64)
l2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') # l1을 w2(3by3) 한칸씩 자르겠다.
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2_flat = tf.reshape(l2, [-1, 7 * 7 * 64])

# Final FC 7x7x64 inputs -> 10 outputs
w3 = tf.get_variable('w3', shape=[7 * 7 * 64, 10],
                        initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(l2_flat, w3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print('Prediction: ', sess.run(tf.argmax(logits, 1), feed_dict={x: mnist.test.images[r:r + 1]}))