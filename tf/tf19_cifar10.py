import tensorflow as tf
import random
import numpy as np
from keras.datasets import cifar10

tf.set_random_seed(777)

# CIFAR_10은 3채널로 구성된 32*32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x = tf.placeholder(tf.float32, [None, IMG_ROWS, IMG_COLS, IMG_CHANNELS])
y = tf.placeholder(tf.int32, [None, 1])

y_one_hot = tf.one_hot(y, 10)
y_one_hot = tf.reshape(y_one_hot, [-1, 10])

w1 = tf.Variable(tf.random_normal([3, 3, IMG_CHANNELS, 32]))
layer1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
layer1 = tf.nn.relu(layer1)
layer1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
layer2 = tf.nn.conv2d(layer1, w2, strides=[1, 1, 1, 1], padding='SAME')
layer2 = tf.nn.relu(layer2)

w3 = tf.Variable(tf.random_normal([3, 3, 64, 128]))
layer3 = tf.nn.conv2d(layer2, w3, strides=[1, 1, 1, 1], padding='SAME')
layer3 = tf.nn.relu(layer3)

flat = tf.reshape(layer3, [-1, 16 * 16 * 128])

w = tf.get_variable('w', shape=[16 * 16 * 128, 10],
                        initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(flat, w) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


def next_batch(num, data, labels):
    '''
    `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 10
batch_size = 100
num_examples = len(x_train)
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(x_train.shape[0] / batch_size)  # batch_size

    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)

        feed_dict = {x: batch_xs, y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    
    print('Epoch: ', '%04d' % (epoch + 1), 'cost = ', '{:9f}'.format(avg_cost))


# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

# Get one and predict
r = random.randint(0, len(x_test) - 1)
# print('Label: ', sess.run(tf.argmax(x_test[r:r + 1], 1)))
print('Prediction: ', sess.run(tf.argmax(logits, 1), feed_dict={x: x_test[r:r + 1]}))
