import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# 노드의 형태를 출력해준다
print('node1: ', node1, 'node2: ', node2)
print('node3: ', node3)

# 1 스칼라
# (1, 2) 벡터 == input.dim=1, input_shape=(1,)
# [[1, 2]] 행렬 == input_shape=(2,3)
# ... 행렬보다 더 큰 거는 텐서

sess = tf.Session()
print('sess.run(node1, node2): ', sess.run([node1, node2])) # model.fit 과 비슷한 개념
print('sess.run(node3): ', sess.run(node3))
