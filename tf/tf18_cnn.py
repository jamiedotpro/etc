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
print('w1: ', w1)   # w1:  <tf.Variable 'Variable:0' shape=(3, 3, 1, 32) dtype=float32_ref>
#   Conv    -> (?, 28, 28, 32)
#   Pool    -> (?, 14, 14, 32)
l1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')  # strides 몇칸씩 자를 것인가. 가로 세로 동일하게. strides=[안씀, 가로, 세로, 안씀]
print('l1: ', l1)   # l1:  Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# tf.nn.max_pool
    # 맥스풀링은 겹쳐지지 않은 상태로 지정범위 내 높은 특성값을 출력한다. ksize처럼 계산되어 크기가 줄어듬
    # ksize: [1, 2, 2, 1]은 2*2 범위에서 1개의 결과를 출력. 4개의 데이터 중 가장 큰 1개를 반환한다.
    # strides:
    # padding: 크기가 홀수인 경우 나누어 떨어지지 않기 때문에 필요. 이전 값이 홀수라면 +1된 상태에서 max_pool이 처리된다.
print('l1: ', l1)   # l1:  Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L2 ImgIn shape=(?, 14, 14, 32)
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # [커널사이즈:몇by몇으로 자를지,, 이전 아웃풋, 아웃풋]
#   Conv    -> (?, 14, 14, 64)
#   Pool    -> (?, 7, 7, 64)
l2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME') # l1을 w2(3by3) 한칸씩 자르겠다.
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
l2_flat = tf.reshape(l2, [-1, 7 * 7 * 64])