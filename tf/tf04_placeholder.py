import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# 노드의 형태를 출력해준다
# print('node1: ', node1, 'node2: ', node2)
# print('node3: ', node3)

# 1 스칼라
# (1, 2) 벡터 == input.dim=1, input_shape=(1,)
# [[1, 2]] 행렬 == input_shape=(2,3)
# ... 행렬보다 더 큰 텐서는 그래프

sess = tf.Session()
# print('sess.run(node1, node2): ', sess.run([node1, node2]))
# print('sess.run(node3): ', sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, feed_dict={a: 3, b: 4.5}))
