# Lab 7 Learning rate and Evaluation
import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2, 1],    [1, 3, 2],
            [1, 3, 4],    [1, 5, 5],
            [1, 7, 5],    [1, 2, 5],
            [1, 6, 6],    [1, 7, 7]]
y_data = [[0, 0, 1],    [0, 0, 1],
            [0, 0, 1],    [0, 1, 0],
            [0, 1, 0],    [0, 1, 0],
            [1, 0, 0],    [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

x = tf.placeholder('float', [None, 3])
y = tf.placeholder('float', [None, 3])

w = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# Try to change learning_rate to small numbers
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) # Accuracy:  1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)    # Accuracy:  0.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)    # Accuracy:  0.33333334
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.06).minimize(cost)    # Accuracy:  0.33333334
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.07).minimize(cost)    # Accuracy:  0.6666667
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08).minimize(cost)    # Accuracy:  1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.09).minimize(cost)    # Accuracy:  1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost) # Accuracy:  1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)   # Accuracy:  1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.3).minimize(cost) # Accuracy:  1.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.4).minimize(cost) # Accuracy:  0.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=2).minimize(cost)   # Accuracy:  0.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=3).minimize(cost)   # Accuracy:  0.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=5).minimize(cost)   # Accuracy:  0.0
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=10).minimize(cost)  # Accuracy:  0.0

# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, w_val, _ = sess.run([cost, w, optimizer],
                                        feed_dict={x: x_data, y: y_data})
        print(step, cost_val, w_val)

    # predict
    print('Prediction: ', sess.run(prediction, feed_dict={x: x_test}))
    # Calculate the accuracy
    print('Accuracy: ', sess.run(accuracy, feed_dict={x: x_test, y: y_test}))