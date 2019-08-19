# 랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력하시오.

import tensorflow as tf
tf.set_random_seed(777)

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

print(w)

w = tf.Variable([0.3], tf.float32)  # w = 0.3

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(w))
# sess.close()

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# aaa = w.eval()  # tf.InteractiveSession() 를 쓰면 sess.run(w) 대신에 w.eval()로 쓸 수 있다
# print(aaa)
# sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = w.eval(session=sess)
print(aaa)
sess.close()