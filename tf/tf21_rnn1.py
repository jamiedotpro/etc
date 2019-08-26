import tensorflow as tf
import numpy as np

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)
print(_data.shape)  # (7, 1)
print(_data)
print(_data.dtype)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')    # 사이킷런은 자동으로 알파벳 순서로 OneHotEncoder 처리해줌. float64로 리턴되기에 float32 로 형변환함

print(_data)
print(_data.shape)  # (7, 5)
print(type(_data))
print(_data.dtype)

x_data = _data[:6,] # (6, 5)
y_data = _data[1:,] # (6, 5)
y_data = np.argmax(y_data, axis=1)

print(x_data)
print(y_data)

x_data = x_data.reshape(1, 6, 5)    # (1, 6, 5)
y_data = y_data.reshape(1, 6)

print(x_data.shape) # (1, 6, 5)
print(x_data.dtype)
print(y_data.shape) # (6, ) ?? # (1, 6)

# 데이터 구성
# x : (batch_size, sequence_length, input_dim)  1, 6, 5
# 첫번째 아웃풋 : hidden_size = 2
# 첫번째 결과 : 1, 6, 5

num_classes = 5
batch_size = 1          # (전체행)
sequence_length = 6     # 컬럼
input_dim = 5           # 몇개씩 작업
hidden_size = 5         # 첫번째 노드 출력 개수
learning_rate = 0.1

# x = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # (?, 6, 5)
# y = tf.placeholder(tf.int32, [None, sequence_length]) # (?, 6)

x = tf.compat.v1.placeholder(tf.float32, [None, sequence_length, input_dim])    # (?, 6, 5)
y = tf.compat.v1.placeholder(tf.int32, [None, sequence_length]) # (?, 6)
print(x)
print(y)

# 2. 모델 구성
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)    #, state_is_tuple=True) # 출력 사이즈
                    # hidden_size 10 == model(10, lstm(...)
# initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)    # initial_state=initial_state,
# outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
print(outputs)
print(outputs.shape)    # (1, 6, 5)

# FC layer
# x_for_fc = tf.reshape(outputs, [-1, hidden_size])   # (6, 5)
# print(x_for_fc)
# outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# w, loss, train, prediction
weights = tf.ones([batch_size, sequence_length])    # 임의로 1을 넣는다

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
train = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2) # 원핫인코딩 원래대로 변환

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        loss2, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
        result = sess.run(prediction, feed_dict={x: x_data})
        print(i, 'loss: ', loss2, 'prediction: ', result, 'true Y: ', y_data)
        # print(sess.run(weights))
        
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print('\nPrediction str: ', ''.join(result_str))