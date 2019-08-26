import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)

# 옵션 설정
learning_rate = 0.001
total_epoch = 30
batch_size = 128

# 가로 픽셀 수를 n_input 으로, 세로 픽셀 수를 입력 단계인 n_step 으로 설정
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

# 신경망 모델 구성
x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_class])

w = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

print(x)    # (?, 28, 28)
print(y)    # (?, 10)
print(w)    # (128, 10)
print(b)    # (10, )

# RNN 에 학습에 사용할 셀을 생성
# 다음 함수들을 사용해서 다른 구조의 셀로 간단하게 변경할 수 있다
# BasicRNNCell, BasicLSTMCell, GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)    # n_hidden = 10 == LSTM(10)

outputs, states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

print(outputs)  # (?, 28, 128)

# 결과를 y의 다음 형식과 바꿔야 하기 때문에
# y : [batch_size, n_class]
# outputs 의 형태를 이에 맞춰 변경해야함
# outputs : [batch_size, n_step, n_hidden]
#       -> [n_step, batch_size, n_hidden]
#       -> [batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
print(outputs)  # (28, ?, 128)
outputs = outputs[-1]
print(outputs)  # (?, 128)

model = tf.matmul(outputs, w) + b
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 신경망 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # x 데이터를 RNN 입력 데이터에 맞게 [batch_size, n_step, n_input] 형태로 변환
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch + 1),
            'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

# 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도: ', sess.run(accuracy, feed_dict={x: test_xs, y: test_ys}))
