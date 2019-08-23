import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

# 노드의 형태를 출력해준다
print('node1: ', node1, 'node2: ', node2)
print('node3: ', node3)

# 0차원: Scalar : s = 5 : shape[]
# 1차원: Vector : v = [1.1, 2.2, 3.3] : shape[d0]
# 2차원: Matrix : m = [[1,2,3], [4,5,6]] : shape[d0, d1]
# 3차원: 3-Tensor : t = [[[2], [4], [6]], [[8], [10], [12]]] : shape [d0, d1, d2]
# n차원: n-Tensor : ... : shape[d0, d1, ... dn-1]

sess = tf.Session()
print('sess.run(node1, node2): ', sess.run([node1, node2]))
print('sess.run(node3): ', sess.run(node3))
