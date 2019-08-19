# Logistic Regression Classifier
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2],
            [2, 3],
            [3, 1],
            [4, 3],
            [5, 3],
            [6, 2]]
y_data = [[0],
            [0],
            [0],
            [1],
            [1],
            [1]]

x = tf.placeholder(tf.float32, shape=[None, 2])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(x, w)))
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost/loss function 로지스틱 리그레션에서 cost에 -가 붙는다.
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1 - y) * tf.log(1 - hypothesis))   # == model.compile(loss='binary_crossentropy'...

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5 ,dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={x: x_data, y: y_data})
    print('\nHypothesis: ', h, '\nCorrect (y): ', c, '\nAccuracy: ', a)